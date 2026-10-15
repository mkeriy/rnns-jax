import jax
import jax.numpy as jnp
from flax import nnx


class BlendModel(nnx.Module):
    """Learned blend of several models' predictions.

    Every model in `models` must map x to (preds, _) with preds of shape
    (batch, out_features). Their predictions are concatenated and mapped back to
    out_features by a Linear head that starts as their plain average, so before
    training the blend equals the mean of the models.

    The models' own weights are trained too. To train only the head, build the
    optimizer with `wrt=nnx.All(nnx.Param, nnx.PathContains("head"))`.
    Returns (preds, None).
    """

    def __init__(self, models: list[nnx.Module], out_features: int, *, rngs: nnx.Rngs):
        self.out_features = out_features
        self.models = nnx.List(models)

        n = len(models)

        def average_init(key, shape, dtype=jnp.float32):
            # (n * out, out): n stacked identities / n -> mean of the n predictions
            return jnp.tile(jnp.eye(out_features, dtype=dtype), (n, 1)) / n

        self.head = nnx.Linear(
            n * out_features,
            out_features,
            kernel_init=average_init,
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, None]:
        preds = [model(x)[0] for model in self.models]
        return self.head(jnp.concatenate(preds, axis=-1)), None
