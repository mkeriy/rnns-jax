import jax
import jax.numpy as jnp
from flax import nnx

from .rnn import RNN, RNNCell


class GRUCell(RNNCell):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        concat = in_features + hidden_features
        self.z_t = nnx.Linear(concat, hidden_features, use_bias=bias, rngs=rngs)
        self.r_t = nnx.Linear(concat, hidden_features, use_bias=bias, rngs=rngs)
        self.h_t = nnx.Linear(concat, hidden_features, use_bias=bias, rngs=rngs)

    def __call__(self, x: jax.Array, h_t: jax.Array) -> tuple[jax.Array, jax.Array]:
        in_s = jnp.concat([x, h_t], axis=1)
        r_t = nnx.sigmoid(self.r_t(in_s))
        z_t = nnx.sigmoid(self.z_t(in_s))

        # candidate state from the input and the reset-gated previous state
        c_h = nnx.tanh(self.h_t(jnp.concat([x, r_t * h_t], axis=1)))

        out = (1 - z_t) * h_t + z_t * c_h
        return out, out

    def initialize_carry(self, batch_size: int) -> jax.Array:
        return jnp.zeros((batch_size, self.hidden_features))


class GRU(RNN):
    cell_type = GRUCell
