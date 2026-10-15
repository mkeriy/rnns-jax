from dataclasses import dataclass, field

import jax
import optax
from flax import nnx

from .losses import (
    cosine_similarity,
    euqlidian_distance,
    mae,
    mse,
    mse_cosine,
    rse,
)

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


def loss_fn(model: nnx.Module, batch: tuple[jax.Array, jax.Array]):
    """Default loss: MSE between `model(X)` predictions and `Y`; returns (loss, preds)."""
    X, Y = batch
    preds, _ = model(X)

    loss = mse(preds, Y)
    return loss, preds


@dataclass
class TrainConfig:
    """Optimizer settings; turn them into an optimizer with `make_optimizer`."""

    optimizer: str = "adam"
    optimizer_params: dict = field(default_factory=dict)
    lr: float = 1e-3
    scheduler: str | None = None
    scheduler_params: dict = field(default_factory=dict)


def make_optimizer(model: nnx.Module, config: TrainConfig) -> nnx.Optimizer:
    """Build an nnx.Optimizer for `model` from the optimizer/lr/scheduler fields of `config`."""
    if config.scheduler is None:
        lr = config.lr
    else:
        lr = SCHEDULER[config.scheduler](**config.scheduler_params)
    return nnx.Optimizer(
        model,
        OPTIMIZERS[config.optimizer](learning_rate=lr, **config.optimizer_params),
        wrt=nnx.Param,
    )


def make_train_step(loss_fn=loss_fn, metrics_fn=None):
    """Return a jitted `train_step(model, optimizer, batch, metrics=None) -> (loss, aux)`.

    loss_fn(model, batch) -> (loss, aux): `batch` is whatever your loader yields,
    `aux` is anything you want back (e.g. predictions). Defaults to the MSE
    `loss_fn` above, which expects `batch = (X, Y)`.
    metrics_fn(loss, aux, batch) -> dict of scalars, used to update `metrics`
    (an nnx.MultiMetric) if one is passed; without it only `loss` is recorded.
    """
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    @nnx.jit
    def train_step(model, optimizer, batch, metrics=None):
        (loss, aux), grads = grad_fn(model, batch)
        optimizer.update(model, grads)
        if metrics is not None:
            values = metrics_fn(loss, aux, batch) if metrics_fn else {"loss": loss}
            metrics.update(**values)
        return loss, aux

    return train_step


def make_eval_step(loss_fn=loss_fn, metrics_fn=None):
    """Like `make_train_step`, without gradients: `eval_step(model, batch, metrics=None)`."""

    @nnx.jit
    def eval_step(model, batch, metrics=None):
        loss, aux = loss_fn(model, batch)
        if metrics is not None:
            values = metrics_fn(loss, aux, batch) if metrics_fn else {"loss": loss}
            metrics.update(**values)
        return loss, aux

    return eval_step
