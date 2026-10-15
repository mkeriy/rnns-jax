import jax
import jax.numpy as jnp
from flax import nnx

from .rnn import RNN, RNNCell


class LSTMCell(RNNCell):
    def __init__(
        self, in_features: int, hidden_features: int, bias: bool = True, *, rngs: nnx.Rngs
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.f = nnx.Linear(
            in_features + hidden_features, 4 * hidden_features, use_bias=bias, rngs=rngs
        )

    def __call__(
        self, x: jax.Array, hidden: tuple[jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        c, h = hidden
        gates = self.f(jnp.concat([x, h], axis=1))
        f_g, i_g, c_g, o_g = jnp.split(gates, indices_or_sections=4, axis=-1)

        new_cell = c * nnx.sigmoid(f_g) + nnx.sigmoid(i_g) * nnx.tanh(c_g)
        new_h = nnx.sigmoid(o_g) * nnx.tanh(new_cell)
        return (new_cell, new_h), new_h

    def initialize_carry(self, batch_size: int) -> tuple[jax.Array, jax.Array]:
        zeros = jnp.zeros((batch_size, self.hidden_features))
        return (zeros, zeros)


class LSTM(RNN):
    cell_type = LSTMCell
