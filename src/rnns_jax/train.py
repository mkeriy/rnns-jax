import os
from dataclasses import dataclass, field

import jax
import optax
from flax import nnx
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm

from .data_prep import ArrayLoader
from .losses import (
    cosine_similarity,
    euqlidian_distance,
    mae,
    mse,
    mse_cosine,
    r2_score,
    rse,
)

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


@dataclass
class TrainConfig:
    optimizer: str = "adam"
    optimizer_params: dict = field(default_factory=dict)
    lr: float = 1e-3
    scheduler: str | None = None
    scheduler_params: dict = field(default_factory=dict)
    batch_size: int = 64
    epochs: int = 10
    seed: int = 0
    eval_every: int | None = None
    log_dir: str | None = None


def _new_metrics() -> nnx.MultiMetric:
    return nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        euqlid_dist=nnx.metrics.Average("euqlid_dist"),
        r2_scr=nnx.metrics.Average("r2_scr"),
    )


def _record(history, writer, prefix, metrics, step):
    for name, value in metrics.compute().items():
        history.setdefault(f"{prefix}_{name}", []).append(float(value))
        if writer is not None:
            writer.add_scalar(name, float(value), step)


def fit(
    model: nnx.Module,
    X,
    y,
    X_val=None,
    y_val=None,
    config: TrainConfig | None = None,
) -> tuple[nnx.Module, dict[str, list[float]]]:
    """Train `model` on (X, y) and return the model and a history of metrics.

    X: (N, window, F_in) array or WindowView, y: (N, F_out). The model must
    return (preds, _) with preds shaped like a batch of y. Validation runs at
    the end of each epoch, or every `config.eval_every` steps if set.
    """
    config = config or TrainConfig()
    has_val = X_val is not None and y_val is not None

    train_loader = ArrayLoader(
        X, y, config.batch_size, shuffle=True, seed=config.seed, drop_last=True
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"{len(y)} samples is fewer than batch_size={config.batch_size}"
        )
    val_loader = (
        ArrayLoader(X_val, y_val, config.batch_size, shuffle=False) if has_val else None
    )

    if config.scheduler is None:
        lr = config.lr
    else:
        lr = SCHEDULER[config.scheduler](**config.scheduler_params)
    optimizer = nnx.Optimizer(
        model,
        OPTIMIZERS[config.optimizer](learning_rate=lr, **config.optimizer_params),
        wrt=nnx.Param,
    )

    train_metrics, val_metrics = _new_metrics(), _new_metrics()
    history: dict[str, list[float]] = {}
    train_writer = val_writer = None
    if config.log_dir is not None:
        train_writer = SummaryWriter(os.path.join(config.log_dir, "train"))
        if has_val:
            val_writer = SummaryWriter(os.path.join(config.log_dir, "val"))

    xb, yb = next(iter(ArrayLoader(X, y, config.batch_size, shuffle=False)))
    model.eval()
    preds, _ = model(xb)
    if preds.shape != yb.shape:
        raise ValueError(
            f"model output shape {preds.shape} does not match target batch shape "
            f"{yb.shape}; check out_ftrs and the target columns"
        )

    def validate(step):
        model.eval()
        val_metrics.reset()
        for xb, yb in val_loader:
            eval_step(model, val_metrics, xb, yb)
        _record(history, val_writer, "val", val_metrics, step)
        history.setdefault("val_step", []).append(step)
        model.train()

    step = 0
    for epoch in range(config.epochs):
        model.train()
        train_metrics.reset()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{config.epochs}")
        for xb, yb in progress:
            train_step(model, optimizer, train_metrics, xb, yb)
            step += 1
            if step % 50 == 0:
                progress.set_postfix(loss=float(train_metrics.compute()["loss"]))
            if has_val and config.eval_every and step % config.eval_every == 0:
                validate(step)

        _record(history, train_writer, "train", train_metrics, step)
        if has_val and not config.eval_every:
            validate(step)

    for writer in (train_writer, val_writer):
        if writer is not None:
            writer.close()
    model.eval()
    return model, history
