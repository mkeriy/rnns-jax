import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers


class LSTMCell(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, in_ftrs: int, out_ftrs: int, bias: bool):
        self.in_features = in_ftrs
        self.out_features = out_ftrs
        self.f = nnx.Linear(in_ftrs + out_ftrs, 4 * out_ftrs, use_bias=bias, rngs=rngs)
        self.rngs = rngs
        self.param_dtype = jnp.float32

    def __call__(
        self, x: jax.Array, hidden: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        c, h = hidden
        in_s = jnp.concat([x, h], axis=1)

        gates = self.f(in_s)
        f_g, i_g, c_g, o_g = jnp.split(gates, indices_or_sections=4, axis=-1)

        new_cell = c * nnx.sigmoid(f_g) + nnx.sigmoid(i_g) * nnx.tanh(c_g)
        new_h = nnx.sigmoid(o_g) * nnx.tanh(new_cell)
        return (new_cell, new_h), new_h

    def initialize_carry(
        self, input_shape: tuple[int, ...]
    ) -> tuple[jax.Array, jax.Array]:  # type: ignore[override]
        batch_dims = input_shape
        carry_init = initializers.zeros_init()

        mem_shape = batch_dims + (self.out_features,)
        c = carry_init(self.rngs(), mem_shape, self.param_dtype)
        h = carry_init(self.rngs(), mem_shape, self.param_dtype)
        return (c, h)


class LSTM(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        in_ftrs: int,
        hidden_ftrs: int,
        out_ftrs: int,
        num_layers: int,
        bias: bool = True,
    ):
        self.in_features = in_ftrs
        self.out_features = out_ftrs
        self.hidden_features = hidden_ftrs
        self.num_layers = num_layers
        self.rngs = rngs
        self.param_dtype = jnp.float32

        self.cells = nnx.List(
            [
                LSTMCell(rngs, in_ftrs if i == 0 else hidden_ftrs, hidden_ftrs, bias)
                for i in range(num_layers)
            ]
        )

        self.ff = nnx.Linear(hidden_ftrs, out_ftrs, use_bias=bias, rngs=rngs)

    def __call__(self, x: jax.Array, carry: jax.Array = None) -> jax.Array:
        scan_fn = lambda carry, cell, x: cell(x, carry)
        if carry is None:
            carry = self.cells[0].initialize_carry(x.shape[:-2])

        for cell in self.cells:
            carry, x = nnx.scan(
                scan_fn, in_axes=(nnx.Carry, None, 1), out_axes=(nnx.Carry, 1)
            )(carry, cell, x)

        y_last = x[:, -1, :]

        out = self.ff(y_last.squeeze())

        return out, carry
