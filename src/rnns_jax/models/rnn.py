from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import nnx

# Every cell state shape in use, each made of (batch, hidden_features) arrays.
# Add a variant here when a new cell keeps a differently shaped state.
State = (
    jax.Array  # GRU: h
    | tuple[jax.Array, jax.Array]  # LSTM: (c, h)
)


class RNNCell(nnx.Module, ABC):
    """One time step of a recurrent layer.

    Subclasses build their weights in `__init__(in_features, hidden_features,
    bias, *, rngs)` and implement `__call__` and `initialize_carry`.
    """

    in_features: int
    hidden_features: int

    @abstractmethod
    def __call__(self, x: jax.Array, state: State) -> tuple[State, jax.Array]:
        """x (batch, in_features), state -> (new state, output (batch, hidden_features))."""

    @abstractmethod
    def initialize_carry(self, batch_size: int) -> State:
        """The state at the start of a sequence (zeros)."""


class RNN(nnx.Module, ABC):
    """Stacked RNN over x of shape (batch, time, in_features).

    Subclasses only set `cell_type`. Layers run one after another, each over
    the whole sequence, and each starts from its own state.

    Returns (preds, state):
    - preds: (batch, out_features) from the last time step, or
      (batch, time, out_features) with `return_sequences=True`;
    - state: the cell state of every layer after the last time step, stacked on
      axis 1, e.g. (batch, num_layers, hidden_features) for GRU. Pass it back as
      `carry` to continue a sequence; by default every layer starts at zeros.
    """

    cell_type: type[RNNCell]

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 1,
        bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        if not hasattr(type(self), "cell_type"):
            raise TypeError(
                f"{type(self).__name__} needs a `cell_type`; subclass RNN and set "
                "cell_type = YourCell, or use GRU / LSTM"
            )
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers

        self.cells = nnx.List(
            [
                self.cell_type(
                    in_features if i == 0 else hidden_features,
                    hidden_features,
                    bias,
                    rngs=rngs,
                )
                for i in range(num_layers)
            ]
        )
        self.ff = nnx.Linear(hidden_features, out_features, use_bias=bias, rngs=rngs)

    def initialize_carry(self, batch_size: int) -> State:
        """Zero state for every layer, stacked on axis 1."""
        states = [cell.initialize_carry(batch_size) for cell in self.cells]
        return jax.tree.map(lambda *s: jnp.stack(s, axis=1), *states)

    def __call__(
        self, x: jax.Array, carry: State | None = None, return_sequences: bool = False
    ) -> tuple[jax.Array, State]:
        if carry is None:
            carry = self.initialize_carry(x.shape[0])

        def step(state, cell, x_t):
            return cell(x_t, state)

        scan = nnx.scan(step, in_axes=(nnx.Carry, None, 1), out_axes=(nnx.Carry, 1))

        states = []
        for i, cell in enumerate(self.cells):
            layer_carry = jax.tree.map(lambda s, i=i: s[:, i], carry)
            state, x = scan(layer_carry, cell, x)  # x becomes this layer's outputs
            states.append(state)

        out = x if return_sequences else x[:, -1, :]
        new_carry = jax.tree.map(lambda *s: jnp.stack(s, axis=1), *states)
        return self.ff(out), new_carry
