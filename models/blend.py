import jax
import jax.numpy as jnp 
from flax import nnx 



class BlendModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, out_ftrs: int, models: list[nnx.Module]):
        self.models = nnx.List(models)
    
    @nnx.jit
    def __call__(self, x: jax.Array):
        preds = []
        for model in self.models:
            model.eval()
            p, _ = model(x)
            preds.append(p)
        
        in_p = jnp.concat(jnp.array(preds), axis=1)


        return out, out