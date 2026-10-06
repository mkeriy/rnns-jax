import argparse
import os
import shutil

import jax
import optax
import orbax.checkpoint as ocp
import polars as pl
from flax import nnx
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter

from .data_prep import BatchGenerator
from .losses import (
    cosine_similarity,
    euqlidian_distance,
    mae,
    mse,
    mse_cosine,
    r2_score,
    rse,
)
from .models import MODEL

# TODO add noize while training
# TODO Different losses
# TODO

OPTIMIZERS = {
    "adam": optax.adam,
    "adamw": optax.adamw,
    "sgd": optax.sgd,
    "rmsprop": optax.rmsprop,
    "adagrad": optax.adagrad,
    "nadamw": optax.nadamw,
}

LOSSES = {
    "mse": mse,
    "mae": mae,
    "cosine": cosine_similarity,
    "euqlid": euqlidian_distance,
    "mse_cosine": mse_cosine,
    "rse": rse,
}

SCHEDULER = {
    "linear": optax.linear_schedule,
    "exponential": optax.exponential_decay,
}


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

    if conf.model == "blend":
        experiment_name = "_".join(
            [
                conf.experiment_name,
                conf.model,
                str(len(conf.models)),
                conf.loss,
                conf.optimizer,
                str(conf.lr_scheduler["init_value"])
                + "-"
                + str(conf.lr_scheduler["end_value"]),
                str(conf.batch_size),
                str(conf.epochs),
            ]
        )
    else:
        experiment_name = "_".join(
            [
                conf.experiment_name,
                conf.model,
                str(conf.model_params["num_layers"]),
                str(conf.model_params["hidden_ftrs"]),
                conf.loss,
                conf.optimizer,
                str(conf.optimizer_params["learning_rate"]),
                str(conf.lr_scheduler["init_value"])
                + "-"
                + str(conf.lr_scheduler["end_value"]),
                str(conf.batch_size),
                str(conf.epochs),
            ]
        )

    conf.save_ckpts_path = conf.save_folder + "/" + experiment_name
    save_metrics_path = conf.save_folder + "/logs/" + experiment_name
    conf.train_log_dir = save_metrics_path + "/train"
    conf.test_log_dir = save_metrics_path + "/test"
    os.makedirs(conf.save_ckpts_path, exist_ok=True)
    OmegaConf.save(config=conf, f=f"{conf.save_ckpts_path}/config.yaml")
    conf.save_ckpts_path += "/ckpts"
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
    if config.model == "blend":
        models = []
        for model_config in config.models:
            m = MODEL[model_config["model"]](
                rngs=nnx.Rngs(config.seed), **model_config["model_params"]
            )
            with ocp.StandardCheckpointer() as ckptr:
                m = load_ckpt(m, ckptr, model_config["ckpt"])
            models.append(m)
        model = MODEL[config.model](
            rngs=nnx.Rngs(config.seed), models=models, **config.model_params
        )
    else:
        model = MODEL[config.model](rngs=nnx.Rngs(config.seed), **config.model_params)

    lr_scheduler = SCHEDULER[config.scheduler](**config.lr_scheduler)

    optimizer_params = dict(config.optimizer_params)
    optimizer_params["learning_rate"] = lr_scheduler
    optimizer = nnx.Optimizer(
        model, OPTIMIZERS[config.optimizer](**optimizer_params), wrt=nnx.Param
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
    train_summary_writer = SummaryWriter(config.train_log_dir)
    if config.evaluate:
        test_summary_writer = SummaryWriter(config.test_log_dir)

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
            for metric, value in train_metrics.compute().items():
                train_summary_writer.add_scalar(metric, float(value), step)

            if config.evaluate and step % 2000 == 0:
                model.eval()
                test_metrics.reset()
                for X_test, Y_test in test_generator:
                    eval_step(model, test_metrics, X_test, Y_test)
                for metric, value in test_metrics.compute().items():
                    test_summary_writer.add_scalar(metric, float(value), step)
            step += 1

    train_summary_writer.close()
    if config.evaluate:
        test_summary_writer.close()

    with ocp.StandardCheckpointer() as ckptr:
        ckpt_dir = ocp.test_utils.erase_and_create_empty(config.save_ckpts_path)
        save_ckpt(model, ckptr, ckpt_dir)


if __name__ == "__main__":
    config = arg_parser()

    train(config)
