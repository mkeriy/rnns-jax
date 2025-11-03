import jax
import jax.numpy as jnp


def euqlidian_distance(preds: jax.Array, targets: jax.Array) -> jax.Array:
    return jnp.mean(jnp.linalg.norm(preds - targets, axis=-1))


def mse(preds: jax.Array, targets: jax.Array) -> jax.Array:
    return jnp.mean(jnp.mean((preds - targets) ** 2, axis=-1))


def cosine_similarity(preds: jax.Array, targets: jax.Array) -> jax.Array:
    return 1 - jnp.mean(jnp.sum(preds * targets, axis=-1) / (
        jnp.linalg.norm(preds, axis=-1) * jnp.linalg.norm(targets, axis=-1)
    ))


def r2_score(preds: jax.Array, targets: jax.Array) -> jax.Array:
    mean_targets = jnp.mean(targets, axis=0)
    sso = jnp.sum((targets - mean_targets)**2, axis=0)
    sse = jnp.sum((targets - preds)**2, axis=0)
    r2 = jnp.mean(1 - sse / sso)
    return r2
    