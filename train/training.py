from flax import nnx
import jax
import optax
import tensorflow as tf
import polars as pl
import orbax.checkpoint as ocp
import argparse
from omegaconf import OmegaConf
import os
import shutil
from models import MODEL

from util.losses import mse, euqlidian_distance, cosine_similarity, r2_score
from util.data_prep import BatchGenerator


def create_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    # Recreate the folder
    os.makedirs(path)
    print(f"[INFO] Created fresh folder at: {path}")


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Run experiment with OmegaConf config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration file (JSON or YAML).",
    )
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    experiment_name = "_".join(
        [
            conf.model,
            str(conf.model_params["num_layers"]),
            str(conf.model_params["hidden_ftrs"]),
            conf.optimizer,
            str(conf.optimizer_params["learning_rate"]),
            str(conf.lr_scheduler["init_value"]) + "-" + str(conf.lr_scheduler["end_value"]),
            str(conf.batch_size),
            str(conf.epochs),
        ]
    )

    conf.save_ckpts_path = conf.save_folder + "/" + experiment_name
    save_metrics_path = conf.save_folder + "/logs/" + experiment_name
    conf.train_log_dir = save_metrics_path + "/train"
    conf.test_log_dir = save_metrics_path + "/test"
    return conf


def save_ckpt(model, ckptr, save_path):
    _, state = nnx.split(model)
    ckptr.save(save_path / "ckpt", state)


def load_ckpt(model, ckptr, load_path):
    graphdef, state = nnx.split(model)
    state_restored = ckptr.restore(load_path, state)
    return nnx.merge(graphdef, state_restored)


def calc_metrics(
    metrics: nnx.MultiMetric, loss: jax.Array, preds: jax.Array, target: jax.Array
) -> None:
    dist = euqlidian_distance(preds, target)
    r2 = r2_score(preds, target)
    metrics.update(loss=loss, euqlid_dist=dist, r2_scr=r2)


def loss_fn(model: nnx.Module, X: jax.Array, Y: jax.Array):
    preds, _ = model(X)
    loss = mse(preds, Y)
    return loss, preds


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    X: jax.Array,
    Y: jax.Array,
):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, preds), grads = grad_fn(model, X, Y)

    calc_metrics(metrics, loss, preds, Y)
    optimizer.update(model, grads)


@nnx.jit
def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, X, Y):
    loss, preds = loss_fn(model, X, Y)
    calc_metrics(metrics, loss, preds, Y)


def train(config):

    rngs = jax.random.PRNGKey(config.seed)

    model = MODEL[config.model](rngs=nnx.Rngs(config.seed), **config.model_params)
    lr_scheduler = optax.linear_schedule(**config.lr_scheduler)
    # config.optimizer_params["learning_rate"] = lr_scheduler
    optimizer = nnx.Optimizer(
        model, optax.adam(learning_rate=lr_scheduler), wrt=nnx.Param
    )

    train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        euqlid_dist=nnx.metrics.Average("euqlid_dist"),
        r2_scr=nnx.metrics.Average("r2_scr"),
    )
    test_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        euqlid_dist=nnx.metrics.Average("euqlid_dist"),
        r2_scr=nnx.metrics.Average("r2_scr"),
    )
    train_summary_writer = tf.summary.create_file_writer(config.train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(config.test_log_dir)

    train_df = pl.read_parquet(config.data_path)
    train_df = train_df[:, 3:]

    if config.evaluate:
        test_df = pl.read_parquet(config.test_path)
        test_df = test_df[:, 3:]
        test_generator = BatchGenerator(rngs, test_df, config.batch_size)
    step = 0
    for epoch in range(config.epochs):
        rngs_data = jax.random.PRNGKey(config.seed + epoch)
        generator = BatchGenerator(rngs_data, train_df, config.batch_size)

        for X, Y in generator:
            model.train()

            train_step(model, optimizer, train_metrics, X, Y)
            with train_summary_writer.as_default():
                for (
                    metric,
                    value,
                ) in train_metrics.compute().items():
                    tf.summary.scalar(metric, value, step=step)
    
            if config.evaluate and step % 2000 == 0:
                model.eval()
                for X_test, Y_test in test_generator:
                    eval_step(model, test_metrics, X_test, Y_test)
                    with test_summary_writer.as_default():
                        for (
                            metric,
                            value,
                        ) in test_metrics.compute().items():
                            tf.summary.scalar(metric, value, step=step)
            step += 1

    with ocp.StandardCheckpointer() as ckptr:
        ckpt_dir = ocp.test_utils.erase_and_create_empty(config.save_ckpts_path)
        save_ckpt(model, ckptr, ckpt_dir)


if __name__ == "__main__":
    config = arg_parser()

    train(config)
