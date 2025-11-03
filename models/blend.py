import jax
import jax.numpy as jnp 
from flax import nnx 



class BlendModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, out_ftrs: int, models: list[nnx.Models]):
        self.models = models
        self.ff = nnx.Linear(in_features=len(models) * out_ftrs, out_features=out_ftrs, rngs=rngs)
    
    def __call__(self, x: jax.Array):
        preds = []
        for model in self.models:
            model.eval()
            p = model(x)
            preds.append(p)
        
        in_p = jnp.concat(preds, axis=1)
        out = nnx.tanh(self.ff(in_p))

        return out