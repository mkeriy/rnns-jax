import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers



class GRUCell(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, in_ftrs: int, out_ftrs: int, bias: bool):
        self.in_features = in_ftrs
        self.out_features = out_ftrs
        self.z_t = nnx.Linear(in_ftrs + out_ftrs,  out_ftrs, use_bias=bias, rngs=rngs)
        self.r_t = nnx.Linear(in_ftrs + out_ftrs,  out_ftrs, use_bias=bias, rngs=rngs)
        self.h_t = nnx.Linear(in_ftrs + out_ftrs, out_ftrs, use_bias=bias, rngs=rngs)
        self.rngs = rngs
        self.param_dtype = jnp.float32
    

    # In GRUCell class

    def __call__(
    self, x: jax.Array, h_t: jax.Array
) -> tuple[jax.Array, jax.Array]:
        
    # Concatenate for gate calculations
        in_s = jnp.concat([x, h_t], axis=1)
        r_t_linear = self.r_t(in_s)
        z_t_linear = self.z_t(in_s)

        r_t = nnx.sigmoid(r_t_linear)
        z_t = nnx.sigmoid(z_t_linear)
        rht = r_t * h_t

        h = jnp.concat([x, rht], axis=1)

    # Calculate linear transformation for candidate state
        c_h_linear = self.h_t(h)
    
    # --- FIX 2: Apply tanh to candidate state ---
        c_h = nnx.tanh(c_h_linear)

    # Final update: combine previous and candidate states
        out = (1 - z_t) * h_t + z_t * c_h
    
    # Return the new state as carry and output
        return out, out
        

    def initialize_carry(
        self, input_shape: tuple[int, ...]
    ) -> tuple[jax.Array, jax.Array]:  # type: ignore[override]
        batch_dims = input_shape
        carry_init = initializers.zeros_init()

        mem_shape = batch_dims + (self.out_features,)
        h = carry_init(self.rngs(), mem_shape, self.param_dtype)
       
        return h


class GRU(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        in_ftrs: int,
        hidden_ftrs: int,
        out_ftrs: int,
        num_layers: int,
        bias: bool = True
    ):
        self.in_features = in_ftrs
        self.out_features = out_ftrs
        self.hidden_features = hidden_ftrs
        self.num_layers = num_layers
        self.rngs = rngs
        self.param_dtype = jnp.float32
        
        
        self.cells = nnx.List([
            GRUCell(rngs, in_ftrs if i == 0 else  hidden_ftrs, hidden_ftrs, bias)
            for i in range(num_layers)
        ])             
        
        self.ff = nnx.Linear(hidden_ftrs, out_ftrs, use_bias=bias, rngs=rngs)

    def __call__(self, x: jax.Array, carry: jax.Array = None) -> jax.Array:
        scan_fn = lambda carry, fn, x: fn(x, carry)
        if carry is None:
            carry = self.cells[0].initialize_carry(x.shape[:1])
        for cell in self.cells:
            carry, x = nnx.scan(
                scan_fn, in_axes=(nnx.Carry, None, 1), out_axes=(nnx.Carry, 1)
            )(carry, cell, x)

        out = self.ff(x[:, -1, :].squeeze())

        return out, carry
