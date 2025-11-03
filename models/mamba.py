import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers


class SelectiveSSM(nnx.Module):
    def __init__(
        self, rngs: nnx.Rngs, in_ftrs: int, hidden_ftrs: int
    ):  # in_ftrs = D, hidden_ftrs=N
        self.in_ftrs = in_ftrs
        self.hidden_ftrs = hidden_ftrs
        initializer = initializers.kaiming_normal()
        
        self.A = nnx.Param(
            initializer(jax.random.key(42), (in_ftrs, hidden_ftrs))
        )  # (D, N)
        self.ll = nnx.Linear(
            in_features=in_ftrs, out_features=2 * hidden_ftrs, rngs=rngs
        )
        self.delta = nnx.Linear(
            in_features=in_ftrs, out_features=in_ftrs, use_bias=True, rngs=rngs
        )

    def _discretize(self, delta, A, B):
        deltaA = delta[..., None] * A[None, None, ...] # (B, L, D, N)
        deltaB = delta[..., None] * delta[..., None, :]
        
        assert deltaA.shape == (delta.shape[0], delta.shape[1], delta.shape[2], A.shape[-1])
        A_disc = jnp.exp(deltaA)  # (B, L, D, N)
        
        I = jnp.eye(A.shape[0], A.shape[1])
      
        B_disc = (deltaA**-1) * (A_disc - I[None, None, ...]) * deltaB

        return A_disc, B_disc

    def __call__(self, x: jax.Array):
        sBsC = self.ll(x)  # B, L, 2N
        B, C = jnp.split(sBsC, indices_or_sections=2, axis=-1)  # (B, L, N), (B, L, N)
        delta = nnx.softplus(self.delta(x))  # (B, L, D)
        Ab, Bb = self._discretize(delta, self.A, B)  # (B, L, D, N), (B, L, D, N)
        
        h = jnp.zeros((x.shape[0], self.hidden_ftrs, self.in_ftrs))# (B, N, D)
        
        @nnx.scan(in_axes=(1, 1, 1, 1, nnx.Carry), out_axes=(1, nnx.Carry))
        def scan_SSM(Ab, Bb, C, x, h):
            new_h = (
                Ab[:,...] * h[:, ...] + Bb[:, ...] @ x[:,:, None]
            )  # (B, D, N) @ (B, N, D) + (B, D, N) @ (B, D) = (B, N, D)
            y = C[:, None,] @ new_h  # (B, D) = (B, N) @ B, N, D

            return y, new_h
        
        y, _ = scan_SSM(Ab, Bb, C, x, h)
        y = y.squeeze()
        assert y.shape == x.shape
        return y


class MambaBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, in_ftrs: int, hidden_ftrs: int, kernel_size: int):

        self.up_projection = nnx.Linear(
            in_features=in_ftrs, out_features=2*hidden_ftrs, rngs=rngs
        )
        self.ssm = SelectiveSSM(rngs=rngs, in_ftrs=hidden_ftrs, hidden_ftrs=hidden_ftrs)
        
        self.conv = nnx.Conv(in_features=hidden_ftrs, out_features=hidden_ftrs, kernel_size=kernel_size, rngs=rngs)
        
        self.down_projection = nnx.Linear(in_features=hidden_ftrs, out_features=in_ftrs, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        expanding = self.up_projection(x)
        left, right = jnp.split(expanding, indices_or_sections=2, axis=-1)
        
        right = nnx.silu(right)
        
        left = self.conv(left)
        left = nnx.silu(left)
        left = self.ssm(left)
        
        out = left * right
        out = self.down_projection(out)
        return out


class Mamba(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, num_layers: int, in_ftrs: int, hidden_ftrs: int, kernel_size: int):
        
        self.norm = nnx.LayerNorm(in_ftrs, rngs=rngs)
        self.blocks = nnx.List([
            MambaBlock(rngs, in_ftrs, hidden_ftrs, kernel_size)
            for i in range(num_layers)
        ])
        
        
    def __call__(self, x: jax.Array):
        
        for block in self.blocks:
            x = self.norm(x)
            y = block(x)
            x = x + y
            
        out = x[:, -1, :]
        
        return out, None