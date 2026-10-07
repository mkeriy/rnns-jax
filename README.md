# RNNs on JAX

Recurrent and other sequence models written in [JAX](https://github.com/jax-ml/jax) and [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html). Train them on your own sequence data from Python.

## Install

Requires Python 3.11+.

```bash
uv add git+https://github.com/mkeriy/rnns-jax.git
pip install git+https://github.com/mkeriy/rnns-jax.git
```

This installs the CPU build of JAX. For GPU, also install `jax[cuda12]`.

## Use as a library

```python
import numpy as np
from flax import nnx
from rnns_jax import GRU, TrainConfig, fit, make_windows, save_model

series = np.random.randn(5000, 4)                  # (time steps, features)
X, y = make_windows(series, window=50)             # X: (N, 50, 4), y: (N, 4)
X_val, y_val = make_windows(np.random.randn(1000, 4), window=50)

model = GRU(rngs=nnx.Rngs(0), in_ftrs=4, hidden_ftrs=64, out_ftrs=4, num_layers=1)
model, history = fit(model, X, y, X_val, y_val, TrainConfig(epochs=5, lr=1e-3))

preds, _ = model(np.asarray(X_val[:32]))           # (32, 4)
save_model(model, "checkpoints/gru")
```

- For tables with many sequences (Polars or Pandas), use `windows_from_frame(df, feature_cols, id_col=..., window=...)`.
- [docs/data.md](docs/data.md) explains how to prepare data: format, splitting, scaling, and a checklist.
- [examples/quickstart.ipynb](examples/quickstart.ipynb) is a full walkthrough: data → windows → training → plots → save/load → swapping models.

## Models

| Model | File                                            |
|-------|-------------------------------------------------|
| GRU   | [models/gru.py](src/rnns_jax/models/gru.py)     |
| LSTM  | [models/lstm.py](src/rnns_jax/models/lstm.py)   |
| Mamba | [models/mamba.py](src/rnns_jax/models/mamba.py) |
| Blend | [models/blend.py](src/rnns_jax/models/blend.py) |

Every model returns `(predictions, state)` from `model(X)`, so they can be swapped freely. `Mamba` has no `out_ftrs`: it predicts `in_ftrs` values. `Blend` is unfinished and doesn't run yet.

## Development

```bash
git clone https://github.com/mkeriy/rnns-jax.git && cd rnns-jax
uv sync                    # or: poetry install --no-root
uv run ruff check src
```
