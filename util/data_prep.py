import jax
import jax.numpy as jnp
from tqdm import trange
import polars as pl

class BatchGenerator:
    def __init__(self, rngs: jax.Array, data: pl.DataFrame, batch_size: int = 16):
        self.rngs = rngs
        self.data = data
        self.data_idxs = self.shuffled_idxs(rngs, data.height)
        self.batch_size = batch_size

    @staticmethod
    def shuffled_idxs(rngs, data_height):
        data_idxs = []
        for i in range(0, data_height, 1000):
            for j in range(899):
                data_idxs.append((i + j, i + j + 100))
        data_idxs = jnp.array(data_idxs)
        return jax.random.permutation(rngs, data_idxs)

    def __iter__(self):
        """
        Returns an iterator that yields (X, Y) batches.
        """
        for i in trange(0, len(self.data_idxs), self.batch_size):
            X, Y = [], []
            batch = self.data_idxs[i : i + self.batch_size]
            for el in batch:
                elem = el.tolist() if hasattr(el, "tolist") else el
                X.append(self.data[elem[0] : elem[1]].to_numpy())
                Y.append(self.data[elem[1]].to_numpy())
            yield jnp.array(X), jnp.array(Y).squeeze()
