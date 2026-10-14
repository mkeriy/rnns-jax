from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
import jax


class Dataset(ABC):
    # Return host (numpy) arrays: the loader stacks them on the host and moves
    # the whole batch to device once.
    @abstractmethod
    def __getitem__(self, index: int) -> tuple[np.ndarray, ...]:
        pass

    # Optional fast path: a subclass may define
    #     def __getitems__(self, indexes: np.ndarray) -> tuple[np.ndarray, ...]
    # returning an already-batched (X, y), e.g. `return self.X[indexes], self.y[indexes]`.
    # DataLoader then skips per-sample collation. Don't declare it here: the
    # loader detects it with `hasattr`, so a stub would always be found.

    @abstractmethod
    def __len__(self) -> int:
        pass


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 0,
        collate_fn: callable | None = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        self.collate_fn = collate_fn

        self.has_bulk_support = hasattr(dataset, "__getitems__")

    def _get_iterator(self) -> DataLoaderIterator:
        return DataLoaderIterator(self)

    def __iter__(self) -> DataLoaderIterator:
        return self._get_iterator()

    def __len__(self) -> int:
        # number of batches per epoch
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return -(-n // self.batch_size)  # ceil division


class DataLoaderIterator:
    def __init__(self, loader: DataLoader):
        self._dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.collate_fn = loader.collate_fn
        data_len = len(self._dataset)

        self.pos = 0

        self.order = np.arange(data_len, dtype=int)
        if loader.shuffle:
            loader.rng.shuffle(self.order)

        self.end = data_len
        if loader.drop_last:
            self.end = data_len - data_len % self.batch_size

        self.bulk_support = loader.has_bulk_support

    def __iter__(
        self,
    ) -> DataLoaderIterator:
        return self

    def __next__(self) -> tuple[jax.Array, jax.Array]:
        if self.pos >= self.end:
            raise StopIteration

        idxs = self.order[self.pos : self.pos + self.batch_size]
        self.pos += self.batch_size

        if self.bulk_support:
            # already batched: only move each field to device
            batch = self._dataset.__getitems__(idxs)
            return tuple(jnp.asarray(field) for field in batch)

        samples = [self._dataset[idx] for idx in idxs]
        return self._collate(samples)

    def _collate(self, batch_samples) -> tuple[jax.Array, ...]:
        if self.collate_fn:
            return self.collate_fn(batch_samples)

        # stack matching fields of all samples on the host, e.g. B samples of
        # (x, y) -> (X[B, ...], Y[B, ...]); works for any pytree sample
        batch = jax.tree.map(lambda *fields: np.stack(fields), *batch_samples)
        # one host-to-device copy per field
        return jax.tree.map(jnp.asarray, batch)
