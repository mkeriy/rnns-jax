from dataclasses import asdict, dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm.auto import tqdm

from .checkpoint import save_model
from .loggers import DictLogger, make_logger
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
    logger: str | None = None
    logger_params: dict = field(default_factory=dict)
    monitor: str = "val_loss"
    monitor_mode: str = "min"
    min_delta: float = 0.0
    ckpt_dir: str | None = None


def _batches(X, y, batch_size: int, rng=None, drop_last: bool = False):
    """Yield (X, y) float32 batches, shuffled with `rng` if given."""
    n = len(y)
    order = np.arange(n) if rng is None else rng.permutation(n)
    end = n - n % batch_size if drop_last else n
    for start in range(0, end, batch_size):
        idx = order[start : start + batch_size]
        yield (
            jnp.asarray(X[idx], dtype=jnp.float32),
            jnp.asarray(y[idx], dtype=jnp.float32),
        )


METRIC_NAMES = ("loss", "euqlid_dist", "r2_scr")


def _new_metrics() -> nnx.MultiMetric:
    return nnx.MultiMetric(**{name: nnx.metrics.Average(name) for name in METRIC_NAMES})


def _improved(value: float, best: float | None, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best - min_delta
    return value > best + min_delta


def _computed(prefix: str, metrics: nnx.MultiMetric) -> dict[str, float]:
    return {f"{prefix}/{name}": float(v) for name, v in metrics.compute().items()}


def fit(
    model: nnx.Module,
    X,
    y,
    X_val=None,
    y_val=None,
    config: TrainConfig | None = None,
) -> tuple[nnx.Module, dict[str, list[float]]]:
    """Train `model` on (X, y) and return the model and a history of metrics.

    X: (N, window, F_in) array, y: (N, F_out). The model must return
    (preds, _) with preds shaped like a batch of y. Validation runs at the end
    of each epoch, or every `config.eval_every` steps if set.

    With validation data, the weights with the best `config.monitor` value are
    returned (and saved to `config.ckpt_dir` on every improvement, if set).
    history["best_step"] and history["best_<monitor>"] list each improvement.
    """
    config = config or TrainConfig()
    has_val = X_val is not None and y_val is not None

    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} samples, y has {len(y)}")
    if has_val and len(X_val) != len(y_val):
        raise ValueError(f"X_val has {len(X_val)} samples, y_val has {len(y_val)}")
    steps_per_epoch = len(y) // config.batch_size
    if steps_per_epoch == 0:
        raise ValueError(
            f"{len(y)} samples is fewer than batch_size={config.batch_size}"
        )
    if config.ckpt_dir is not None and not has_val:
        raise ValueError("ckpt_dir needs validation data (X_val, y_val)")
    monitor_keys = [f"val_{name}" for name in METRIC_NAMES]
    if config.monitor not in monitor_keys:
        raise ValueError(
            f"monitor must be one of {monitor_keys}, got {config.monitor!r}"
        )
    if config.monitor_mode not in ("min", "max"):
        raise ValueError(
            f"monitor_mode must be 'min' or 'max', got {config.monitor_mode!r}"
        )
    rng = np.random.default_rng(config.seed)

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

    xb, yb = next(_batches(X, y, config.batch_size))
    model.eval()
    preds, _ = model(xb)
    if preds.shape != yb.shape:
        raise ValueError(
            f"model output shape {preds.shape} does not match target batch shape "
            f"{yb.shape}; check out_ftrs and the target columns"
        )

    dict_logger = DictLogger()
    loggers = [dict_logger]
    extra = make_logger(config.logger, config.logger_params, asdict(config))
    if extra is not None:
        loggers.append(extra)

    def log(metrics, step):
        for logger in loggers:
            logger.log(metrics, step)

    best = best_state = None

    def validate(step):
        nonlocal best, best_state
        model.eval()
        val_metrics.reset()
        for xb, yb in _batches(X_val, y_val, config.batch_size):
            eval_step(model, val_metrics, xb, yb)
        metrics = _computed("val", val_metrics)
        log(metrics, step)

        value = metrics[config.monitor.replace("_", "/", 1)]
        if _improved(value, best, config.monitor_mode, config.min_delta):
            best = value
            best_state = jax.tree.map(jnp.copy, nnx.state(model))
            if config.ckpt_dir is not None:
                save_model(model, config.ckpt_dir)
            log({f"best/{config.monitor}": best}, step)
        model.train()

    try:
        step = 0
        for epoch in range(config.epochs):
            model.train()
            train_metrics.reset()
            progress = tqdm(
                _batches(X, y, config.batch_size, rng=rng, drop_last=True),
                total=steps_per_epoch,
                desc=f"epoch {epoch + 1}/{config.epochs}",
            )
            for xb, yb in progress:
                train_step(model, optimizer, train_metrics, xb, yb)
                step += 1
                if step % 50 == 0:
                    progress.set_postfix(loss=float(train_metrics.compute()["loss"]))
                if has_val and config.eval_every and step % config.eval_every == 0:
                    validate(step)

            log(_computed("train", train_metrics), step)
            if has_val and not config.eval_every:
                validate(step)
    finally:
        for logger in loggers:
            logger.close()

    if best_state is not None:
        nnx.update(model, best_state)
    model.eval()
    return model, dict_logger.history
