"""Losses and metrics as pure functions `f(preds, targets) -> scalar`.

`preds` and `targets` have the same shape (..., features), e.g. (batch, features)
or (batch, time, features). Each function reduces over features first, then
averages over all leading axes, unless its docstring says otherwise.
"""

import jax
import jax.numpy as jnp

EPS = 1e-8


def _norm(x: jax.Array) -> jax.Array:
    # L2 norm over features; the eps keeps its gradient finite at x == 0,
    # where jnp.linalg.norm's gradient is NaN.
    return jnp.sqrt(jnp.sum(x**2, axis=-1) + EPS**2)


def mse(preds: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean squared error."""
    return jnp.mean(jnp.mean((preds - targets) ** 2, axis=-1))


def mae(preds: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean absolute error."""
    return jnp.mean(jnp.mean(jnp.abs(preds - targets), axis=-1))


def euclidean_distance(preds: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean L2 distance between prediction and target vectors."""
    return jnp.mean(_norm(preds - targets))


def cosine_distance(preds: jax.Array, targets: jax.Array) -> jax.Array:
    """1 - mean cosine similarity: 0 when every prediction points the same way
    as its target, 2 when opposite. Ignores magnitude. A zero vector counts as
    similarity 0."""
    cos = jnp.sum(preds * targets, axis=-1) / (_norm(preds) * _norm(targets))
    return 1 - jnp.mean(cos)


def mse_cosine(preds: jax.Array, targets: jax.Array, alpha: float = 0.1) -> jax.Array:
    """mse + alpha * cosine_distance: fits magnitude and rewards the right direction."""
    return mse(preds, targets) + alpha * cosine_distance(preds, targets)


def rse(preds: jax.Array, targets: jax.Array) -> jax.Array:
    """Relative squared error: squared error divided by the targets' variance,
    per feature, averaged. 0 is perfect, 1 is no better than predicting the mean.

    The mean and variance are taken over axis 0 (the batch), so the value
    depends on the batch: averaging it over validation batches is not the
    same as computing it on the whole set.
    """
    sso = jnp.sum((targets - jnp.mean(targets, axis=0)) ** 2, axis=0)
    sse = jnp.sum((targets - preds) ** 2, axis=0)
    return jnp.mean(sse / (sso + EPS))


def r2_score(preds: jax.Array, targets: jax.Array) -> jax.Array:
    """Coefficient of determination, 1 - rse: 1 is perfect, 0 is no better than
    predicting the mean. Per batch, like `rse`."""
    return 1 - rse(preds, targets)


# name -> loss, for choosing a loss from a config string
LOSSES = {
    "mse": mse,
    "mae": mae,
    "euclidean": euclidean_distance,
    "cosine": cosine_distance,
    "mse_cosine": mse_cosine,
    "rse": rse,
}
