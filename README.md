# RNNs on JAX

Recurrent and other sequence models written in [JAX](https://github.com/jax-ml/jax) and [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html). You prepare the data and write the pipeline; the library gives you the models and a `fit` function that trains on your arrays.

## Install

Requires Python 3.11+.

```bash
uv add git+https://github.com/mkeriy/rnns-jax.git
pip install git+https://github.com/mkeriy/rnns-jax.git
```

This installs the CPU build of JAX. For GPU, also install `jax[cuda12]`.

## Usage

`fit` takes NumPy arrays:

- `X`: `(N, window, in_ftrs)` — N input windows;
- `y`: `(N, out_ftrs)` — one target per window;
- `X_val`, `y_val` (optional): the same shapes, for validation.

```python
import numpy as np
from flax import nnx
from numpy.lib.stride_tricks import sliding_window_view
from rnns_jax import GRU, TrainConfig, fit, save_model

def make_windows(series, window):
    # every `window`-step slice of `series` -> the step right after it
    X = sliding_window_view(series[:-1], window, axis=0).transpose(0, 2, 1)
    return X, series[window:]

X, y = make_windows(np.random.randn(5000, 4).astype(np.float32), window=50)  # (4950, 50, 4), (4950, 4)
X_val, y_val = make_windows(np.random.randn(1000, 4).astype(np.float32), window=50)

model = GRU(rngs=nnx.Rngs(0), in_ftrs=4, hidden_ftrs=64, out_ftrs=4, num_layers=1)
model, history = fit(model, X, y, X_val, y_val, TrainConfig(epochs=5, lr=1e-3))

preds, _ = model(X_val[:32])                       # (32, 4)
save_model(model, "checkpoints/gru")
```

- `TrainConfig` sets `optimizer` (`adam`, `adamw`, `sgd`, `rmsprop`, `adagrad`, `nadamw`) and `optimizer_params`, `lr` or a `scheduler` (`linear`, `exponential`) with `scheduler_params`, `batch_size`, `epochs`, `seed`, and `eval_every` (validate every N steps instead of every epoch).
- `history` is a dict of lists: `train_loss`, `val_loss`, `train_r2_scr`, ..., plus `train_step` / `val_step` — plot it however you like.
- `load_model(GRU(...same args...), "checkpoints/gru")` loads saved weights.

### Logging, checkpoints and early stopping

```python
config = TrainConfig(
    epochs=100,
    logger="wandb",                                # or "tensorboard"; None = only the history dict
    logger_params={"project": "my-project"},       # tensorboard: {"log_dir": "runs/gru"}
    monitor="val_loss", monitor_mode="min",        # what "best" means, e.g. "val_r2_scr" + "max"
    ckpt_dir="checkpoints/gru_best",               # saved on every new best validation result
    early_stopping_patience=5,                     # stop after 5 validations without improvement
)
model, history = fit(model, X, y, X_val, y_val, config)  # model has the best weights
```

- Loggers need an extra: `pip install "rnns-jax[tensorboard]"`, `"rnns-jax[wandb]"` or `"rnns-jax[all]"`. W&B also gets the whole `TrainConfig` as the run config.
- With validation data, `fit` always returns the best weights (by `monitor`), even without `ckpt_dir`. `history["best_step"][-1]` and `history["best_val_loss"][-1]` say where and how good; `history["stopped_step"]` is set if training stopped early.
- `min_delta` sets how much better a result must be to count as an improvement. Patience counts validations: epochs by default, or `eval_every` steps.
- To log somewhere else, subclass `rnns_jax.Logger` (`log(metrics, step)`, `close()`).

### Walkthrough

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
