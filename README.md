# RNNs on JAX

Recurrent and other sequence models written in [JAX](https://github.com/jax-ml/jax) and [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html). Train them on your own sequence data from Python, or from a YAML config with the `rnns-jax` command.

## Install

Requires Python 3.11+.

```bash
# library only
uv add git+https://github.com/mkeriy/rnns-jax.git
pip install git+https://github.com/mkeriy/rnns-jax.git

# library + rnns-jax command (adds polars to read Parquet)
uv add "rnns-jax[cli] @ git+https://github.com/mkeriy/rnns-jax.git"
pip install "rnns-jax[cli] @ git+https://github.com/mkeriy/rnns-jax.git"
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

## Use from the command line

Requires the `[cli]` extra.

1. Put your data in Parquet files: one row per time step, plus an optional sequence id column.
2. Copy a config from [configs/](configs/) and set at least:
   - the `data:` paths and columns;
   - `in_ftrs` / `out_ftrs`, which must match the number of feature and target columns.
3. Train:

```bash
rnns-jax train --config configs/gru.yaml
```

The checkpoint and the config used are saved to `<save_folder>/<run name>/`, and TensorBoard logs to `<save_folder>/logs/<run name>/`.

## Models

| Model | File                                                           | Config                                         |
|-------|----------------------------------------------------------------|------------------------------------------------|
| GRU   | [models/gru.py](src/rnns_jax/models/gru.py)                    | [configs/gru.yaml](configs/gru.yaml)           |
| LSTM  | [models/lstm.py](src/rnns_jax/models/lstm.py)                  | [configs/lstm.yaml](configs/lstm.yaml)         |
| Mamba | [models/mamba.py](src/rnns_jax/models/mamba.py)                | [configs/mamba.yaml](configs/mamba.yaml)       |
| Blend | [models/blend.py](src/rnns_jax/models/blend.py)                | [configs/blending.yaml](configs/blending.yaml) |

Every model returns `(predictions, state)` from `model(X)`, so they can be swapped freely. `Mamba` has no `out_ftrs`: it predicts `in_ftrs` values. `Blend` is unfinished and doesn't run yet.

## Development

```bash
git clone https://github.com/mkeriy/rnns-jax.git && cd rnns-jax
uv sync --extra cli        # or: poetry install --no-root --extras cli
uv run ruff check src
```
