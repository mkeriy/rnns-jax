# rnns-jax

Sequence models (GRU, LSTM, Mamba) in [JAX](https://github.com/jax-ml/jax) + [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html), and small building blocks to train them. There is no `fit`: you write the loop, so data, loss, validation and logging are yours.

## Install

Python 3.11+. CPU JAX by default; for GPU also install `jax[cuda12]`.

```bash
pip install git+https://github.com/mkeriy/rnns-jax.git
```

## Quickstart

```python
import numpy as np
from flax import nnx
from rnns_jax import GRU, DataLoader, Dataset, TrainConfig, make_eval_step, make_optimizer, make_train_step


class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X.astype(np.float32), y.astype(np.float32)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __getitems__(self, idx):  # optional: whole batch in one indexing call
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


X, y = np.random.randn(5000, 50, 4), np.random.randn(5000, 4)  # (N, time, features), (N, targets)
train = DataLoader(ArrayDataset(X[:4000], y[:4000]), batch_size=64, shuffle=True, drop_last=True)
val = DataLoader(ArrayDataset(X[4000:], y[4000:]), batch_size=64)

model = GRU(in_features=4, hidden_features=64, out_features=4, rngs=nnx.Rngs(0))
optimizer = make_optimizer(model, TrainConfig(optimizer="adam", lr=1e-3))
train_step, eval_step = make_train_step(), make_eval_step()  # default loss: MSE
metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

for epoch in range(5):
    model.train()
    for batch in train:
        train_step(model, optimizer, batch)
    model.eval()
    metrics.reset()
    for batch in val:
        eval_step(model, batch, metrics)
    print(epoch, float(metrics.compute()["loss"]))

predict = nnx.jit(lambda model, X: model(X)[0])  # jit inference at the call site
preds = predict(model, X[4000:])  # (1000, 4)
```

[examples/quickstart.ipynb](https://github.com/mkeriy/rnns-jax/blob/main/examples/quickstart.ipynb) has the full loop: early stopping, best weights, checkpoints, metrics, a custom loss.

## API

| Name | What it does |
| --- | --- |
| `Dataset`, `DataLoader` | `__getitem__`, `__len__`, optional `__getitems__`. Reshuffles each epoch from `seed`. Any iterable of batches works instead. |
| `TrainConfig`, `make_optimizer` | `optimizer`, `optimizer_params`, `lr` or `scheduler` + `scheduler_params` → `nnx.Optimizer`. |
| `make_train_step(loss_fn, metrics_fn)` | Jitted `step(model, optimizer, batch, metrics=None) -> (loss, aux)`. |
| `make_eval_step(loss_fn, metrics_fn)` | Same without gradients: `step(model, batch, metrics=None)`. |
| `loss_fn(model, batch) -> (loss, aux)` | Default is MSE on `batch = (X, y)`; pass your own to the step factories. |
| `rnns_jax.losses` | `mse`, `mae`, `euclidean_distance`, `cosine_distance`, `mse_cosine`, `rse`, `r2_score`: `f(preds, targets) -> scalar`. |
| `save_model`, `load_model` | Save weights; load into a model built with the same arguments. |

## Models

`model(X)` with `X` of shape `(batch, time, in_features)` returns `(predictions, state)`.

| Model | Arguments and notes |
| --- | --- |
| `GRU`, `LSTM` | `(in_features, hidden_features, out_features, num_layers=1, bias=True, *, rngs)`. `state`: each layer's final state, `(batch, num_layers, hidden_features)`; pass it back as `carry` to continue. `return_sequences=True` predicts at every step. |
| `RNN`, `RNNCell` | Base classes: subclass `RNNCell`, then `class MyRNN(RNN): cell_type = MyCell`. |
| `BlendModel` | `(models, out_features, *, rngs)`: learned blend of several models' predictions, initialised as their average. |
| `Mamba` | `(rngs, num_layers, in_ftrs, hidden_ftrs, kernel_size)`, predicts `in_ftrs` values. Interface will be aligned with the RNNs. |

## Development

```bash
git clone https://github.com/mkeriy/rnns-jax.git && cd rnns-jax
uv sync && uv run ruff check src
```

MIT License.
