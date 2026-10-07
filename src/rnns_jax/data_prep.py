from itertools import pairwise

import jax.numpy as jnp
import numpy as np


class WindowView:
    """Lazy (N, window, F) array of windows cut from one (rows, F) array.

    Windows are only copied when indexed, so a batch costs batch_size * window
    rows of memory instead of the whole windowed dataset.
    """

    def __init__(self, data: np.ndarray, starts: np.ndarray, window: int):
        self.data = data
        self.starts = starts
        self.window = window
        self._offsets = np.arange(window)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.starts), self.window, self.data.shape[1])

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx) -> np.ndarray:
        starts = self.starts[idx]
        return self.data[np.asarray(starts)[..., None] + self._offsets]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        out = self[:]
        return out if dtype is None else out.astype(dtype)


def _window_starts(length: int, window: int, horizon: int, stride: int) -> np.ndarray:
    n = (length - window - horizon) // stride + 1
    return np.arange(max(n, 0)) * stride


def _check_window_args(window: int, horizon: int, stride: int) -> None:
    if window < 1 or horizon < 1 or stride < 1:
        raise ValueError(
            f"window, horizon and stride must be >= 1, got {window=}, {horizon=}, {stride=}"
        )


def make_windows(
    series, window: int, horizon: int = 1, stride: int = 1, target=None
) -> tuple[WindowView, np.ndarray]:
    """Cut one time series into (input window, target) training samples.

    series: (T, F) array. Sample i is series[s : s + window] with target
    target[s + window + horizon - 1], where s = i * stride. target defaults to
    series; pass a (T, F_out) array to predict other columns.

    Returns X as a lazy (N, window, F) WindowView and y as an (N, F_out) array.
    """
    _check_window_args(window, horizon, stride)
    data = np.asarray(series, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(
            f"series must be 2-D (T, F), got shape {data.shape}; use series[:, None] for one feature"
        )
    target = data if target is None else np.asarray(target, dtype=np.float32)
    if len(target) != len(data):
        raise ValueError(f"target has {len(target)} rows, series has {len(data)}")

    starts = _window_starts(len(data), window, horizon, stride)
    return WindowView(data, starts, window), target[starts + window + horizon - 1]


def windows_from_frame(
    df,
    feature_cols: list[str],
    target_cols: list[str] | None = None,
    id_col: str | None = None,
    *,
    window: int,
    horizon: int = 1,
    stride: int = 1,
) -> tuple[WindowView, np.ndarray]:
    """Turn a Polars/Pandas table of one or many sequences into (X, y).

    Rows must be sorted by time, and the rows of each id_col sequence must be
    contiguous. Windows never cross from one sequence into the next, and
    sequences shorter than window + horizon rows are skipped.
    """
    _check_window_args(window, horizon, stride)
    target_cols = feature_cols if target_cols is None else target_cols
    data = np.asarray(df[list(feature_cols)].to_numpy(), dtype=np.float32)
    target = np.asarray(df[list(target_cols)].to_numpy(), dtype=np.float32)

    if id_col is None:
        bounds = np.array([0, len(data)])
    else:
        ids = np.asarray(df[id_col].to_numpy())
        changes = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        bounds = np.concatenate([[0], changes, [len(ids)]])

    starts = np.concatenate(
        [
            begin + _window_starts(end - begin, window, horizon, stride)
            for begin, end in pairwise(bounds)
        ]
    )
    return WindowView(data, starts, window), target[starts + window + horizon - 1]


class ArrayLoader:
    """Iterate over (X, y) in shuffled float32 jnp batches.

    Each pass uses a new order seeded with seed + epoch, so runs are
    reproducible. X can be a NumPy array or a WindowView.
    """

    def __init__(
        self,
        X,
        y,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} samples, y has {len(y)}")
        self.X = X
        self.y = np.asarray(y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __len__(self) -> int:
        n = len(self.y)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.y)
        if self.shuffle:
            order = np.random.default_rng(self.seed + self.epoch).permutation(n)
        else:
            order = np.arange(n)
        self.epoch += 1

        for b in range(len(self)):
            idx = order[b * self.batch_size : (b + 1) * self.batch_size]
            yield (
                jnp.asarray(self.X[idx], dtype=jnp.float32),
                jnp.asarray(self.y[idx], dtype=jnp.float32),
            )
