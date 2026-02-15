import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers
from einops import rearrange, repeat, einsum



class DepthwiseConv1D(nnx.Module):
    """Depthwise 1D Convolution with specified features and kernel size."""
    def __init__(self, features: int, kernel_size: int, *, rngs: nnx.Rngs):
        self.features = features
        self.kernel_size = kernel_size

        # nnx.Conv replaces nn.Conv, using kernel_init and bias_init arguments.
        # feature_group_count=features makes it depthwise.
        # We use a custom 'Conv_0' name for the Conv module to potentially match state dict keys
        # from the original implementation's naming conventions, but nnx uses instance attributes.
        self.Conv_0 = nnx.Conv(
            in_features=features, # in_features is usually required by nnx.Conv
            out_features=features,
            kernel_size=(kernel_size,),
            feature_group_count=features,
            strides=(1,),
            padding='SAME', # Use 'SAME' padding for convenience if input and output dims should match
            rngs=rngs
        )
    
    @nnx.jit
    def __call__(self, x):
        # The original code's padding logic was `padding=self.kernel_size - 1` which
        # implies a 'valid' convolution on a padded input.
        # Here we use 'SAME' padding for simplicity which often approximates the effect.
        # For a full match, explicit padding and a 'VALID' conv would be needed.
        # For now, let's stick to the simplest interpretation of 'same' padding, which might
        # behave differently than the original's explicit padding and slicing.

        # The original implementation uses an explicit pad then a valid conv, and slices.
        # Here we mimic the padding used in the original
        pad_size = 0
        x = jnp.pad(x, ((0, 0), (pad_size, 0), (0, 0)))

        return self.Conv_0(x)


class SelectiveSSM(nnx.Module):
    def __init__(
        self, rngs: nnx.Rngs, in_ftrs: int, hidden_ftrs: int
    ):  # in_ftrs = D, hidden_ftrs=N
        self.in_ftrs = in_ftrs
        self.hidden_ftrs = hidden_ftrs
        # initializer = initializers.kaiming_normal()

        self.A_log = nnx.Param(self.a_log_initiazer((in_ftrs, hidden_ftrs)))
        self.D = nnx.Param(jnp.ones(hidden_ftrs, dtype=jnp.float32))
        # (D, N)
        self.ll = nnx.Linear(
            in_features=in_ftrs, out_features=3 * hidden_ftrs, rngs=rngs
        )
    
    @staticmethod
    def a_log_initiazer(shape: tuple[int, int]) -> jax.Array:
        return jnp.log(repeat(jnp.arange(1, shape[0]  + 1), 'n -> d n', d=shape[1]))
    
    @nnx.jit
    @staticmethod
    def run_parallel_scan(Ab, Bb_u, Cb):
        # Associative operation inspired from "Annotated Mamba" by S.Rush
        def combine_parallel(state1, state2):
            # (a_1​,b_1​)⊕(a_2​,b_2​)=(a_1 * ​a_2​,a_2 *​ b_1​ + b_2​)
            fl, xl = state1
            fr, xr = state2
            f = fr * fl
            x = fr * xl + xr
            return f, x

        # Perform associative scan
        results = jax.lax.associative_scan(combine_parallel, (Ab, Bb_u))
        return einsum(results[1], Cb, 'l b d_in n, l b n -> l b d_in')

    @nnx.jit
    def __call__(self, x: jax.Array):
        delta_sBsC = self.ll(x)  # B, L, 3N
        delta, B, C = jnp.split(delta_sBsC, indices_or_sections=3, axis=-1)  # (B, L, N), (B, L, N)
        delta = nnx.softplus(delta)  
        A = -jnp.exp(self.A_log.astype(float))
        deltaA = jnp.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_x = einsum(delta, B, x, 'b l d_in, b l n, b l d_in  -> b l d_in n') # (B, L, D, N), (B, L, D, N)

        ys = SelectiveSSM.run_parallel_scan(deltaA.swapaxes(0, 1), deltaB_x.swapaxes(0, 1), C.swapaxes(0, 1))
        y = ys.swapaxes(0, 1)
        
        y = y + x * self.D 

        y = y.squeeze()
        assert y.shape == x.shape
        return y



class MambaBlock(nnx.Module):
    def __init__(
        self, rngs: nnx.Rngs, in_ftrs: int, hidden_ftrs: int, kernel_size: int
    ):

        self.up_projection = nnx.Linear(
            in_features=in_ftrs, out_features=2 * hidden_ftrs, rngs=rngs
        )
        self.ssm = SelectiveSSM(rngs=rngs, in_ftrs=hidden_ftrs, hidden_ftrs=hidden_ftrs)

        self.conv = DepthwiseConv1D(features=hidden_ftrs, kernel_size=kernel_size, rngs=rngs)

        self.down_projection = nnx.Linear(
            in_features=hidden_ftrs, out_features=in_ftrs, rngs=rngs
        )

    @nnx.jit
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
    def __init__(
        self,
        rngs: nnx.Rngs,
        num_layers: int,
        in_ftrs: int,
        hidden_ftrs: int,
        kernel_size: int,
    ):

        self.norm = nnx.RMSNorm(in_ftrs, rngs=rngs)
        self.blocks = nnx.List(
            [
                MambaBlock(rngs, in_ftrs, hidden_ftrs, kernel_size)
                for i in range(num_layers)
            ]
        )
        self.f = nnx.Linear(in_features=in_ftrs, out_features=in_ftrs, use_bias=True, rngs=rngs)

    @nnx.jit
    def __call__(self, x: jax.Array):

        for block in self.blocks:
            y = self.norm(x)
            y = block(y)
            x = x + y

        out = x[:, -1, :]
        out = self.f(out.squeeze())

        return out, None
