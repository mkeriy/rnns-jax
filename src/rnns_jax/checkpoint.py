from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx


def save_ckpt(model, ckptr, save_path):
    _, state = nnx.split(model)
    ckptr.save(save_path / "ckpt", state)


def load_ckpt(model, ckptr, load_path):
    graphdef, state = nnx.split(model)
    state_restored = ckptr.restore(load_path, state)
    return nnx.merge(graphdef, state_restored)


def save_model(model: nnx.Module, path) -> Path:
    """Save the model's weights to the folder `path`, overwriting it if it exists."""
    path = Path(path).resolve()
    _, state = nnx.split(model)
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(path, state, force=True)
    return path


def load_model(model: nnx.Module, path) -> nnx.Module:
    """Load weights saved by `save_model` into a model built with the same arguments."""
    with ocp.StandardCheckpointer() as ckptr:
        return load_ckpt(model, ckptr, Path(path).resolve())
