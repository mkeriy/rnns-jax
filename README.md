# RNNs on JAX

Recurrent and other sequence models written in [JAX](https://github.com/jax-ml/jax) and [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html).

## Models

| Model       | File                                           | Config                                         |
|-------------|------------------------------------------------|------------------------------------------------|
| GRU         | [models/gru.py](models/gru.py)                 | [configs/gru.json](configs/gru.json)           |
| LSTM        | [models/lstm.py](models/lstm.py)               | [configs/lstm.json](configs/lstm.json)         |                                          |
| Mamba       | [models/mamba.py](models/mamba.py)             | [configs/mamba.json](configs/mamba.json)       |                                           |
| Blend       | [models/blend.py](models/blend.py)             | [configs/blending.json](configs/blending.json) |

## Setup

Requires Python 3.11+. Dependencies are declared in [pyproject.toml](pyproject.toml). Install them with any of these:

### uv

```bash
uv sync
```

### Poetry

```bash
poetry install --no-root
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

This installs the CPU build of JAX. For GPU, also install `jax[cuda12]`.

## Training

1. Put the training and test data (Parquet) where the config's `data_path` and `test_path` point (by default `./datasets/`).
2. Set the model, optimizer, scheduler and training parameters in a config. There are examples in [configs/](configs/).
3. Run training:

```bash
python training.py --config configs/<name>.json
```

Checkpoints and the resolved config are saved to `<save_folder>/<experiment_name>/`. Logs are saved to `<save_folder>/logs/`.
