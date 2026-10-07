import argparse
import os

from flax import nnx
from omegaconf import DictConfig, OmegaConf

from .checkpoint import load_model, save_model
from .data_prep import windows_from_frame
from .models import MODEL
from .train import TrainConfig, fit


def _read_frame(path):
    try:
        import polars as pl
    except ImportError as e:
        raise ImportError(
            'The CLI reads Parquet with polars: pip install "rnns-jax[cli]"'
        ) from e
    return pl.read_parquet(path)


def experiment_name(conf: DictConfig) -> str:
    lr = (
        f"{conf.scheduler_params.init_value}-{conf.scheduler_params.end_value}"
        if conf.get("scheduler")
        else str(conf.get("lr", 1e-3))
    )
    if conf.model == "blend":
        size = [str(len(conf.models))]
    else:
        size = [
            str(conf.model_params.num_layers),
            str(conf.model_params.hidden_ftrs),
        ]
    parts = [
        conf.get("experiment_name", ""),
        conf.model,
        *size,
        conf.optimizer,
        lr,
        str(conf.batch_size),
        str(conf.epochs),
    ]
    return "_".join(p for p in parts if p)


def build_model(conf: DictConfig) -> nnx.Module:
    rngs = nnx.Rngs(conf.seed)
    if conf.model != "blend":
        return MODEL[conf.model](rngs=rngs, **conf.model_params)

    models = [
        load_model(MODEL[m.model](rngs=nnx.Rngs(conf.seed), **m.model_params), m.ckpt)
        for m in conf.models
    ]
    return MODEL["blend"](rngs=rngs, models=models, **conf.model_params)


def load_windows(conf: DictConfig, path: str):
    df = _read_frame(path)
    data = conf.data
    exclude = set(data.get("drop_cols") or [])
    if data.get("id_col"):
        exclude.add(data.id_col)
    feature_cols = data.get("feature_cols") or [
        c for c in df.columns if c not in exclude
    ]
    return windows_from_frame(
        df,
        list(feature_cols),
        target_cols=list(data.target_cols) if data.get("target_cols") else None,
        id_col=data.get("id_col"),
        window=data.window,
        horizon=data.get("horizon", 1),
        stride=data.get("stride", 1),
    )


def train(conf: DictConfig) -> None:
    name = experiment_name(conf)
    run_dir = os.path.join(conf.save_folder, name)
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(config=conf, f=os.path.join(run_dir, "config.yaml"))

    X, y = load_windows(conf, conf.data.train_path)
    X_val = y_val = None
    if conf.data.get("val_path"):
        X_val, y_val = load_windows(conf, conf.data.val_path)

    model = build_model(conf)
    train_config = TrainConfig(
        optimizer=conf.optimizer,
        optimizer_params=dict(conf.get("optimizer_params") or {}),
        lr=conf.get("lr", 1e-3),
        scheduler=conf.get("scheduler"),
        scheduler_params=dict(conf.get("scheduler_params") or {}),
        batch_size=conf.batch_size,
        epochs=conf.epochs,
        seed=conf.seed,
        eval_every=conf.get("eval_every"),
        log_dir=os.path.join(conf.save_folder, "logs", name),
    )
    model, _ = fit(model, X, y, X_val, y_val, train_config)
    path = save_model(model, os.path.join(run_dir, "ckpt"))
    print(f"Saved checkpoint to {path}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="rnns-jax")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train a model from a config")
    train_parser.add_argument(
        "--config", required=True, help="Path to a YAML or JSON config"
    )
    args = parser.parse_args(argv)

    if args.command == "train":
        train(OmegaConf.load(args.config))


if __name__ == "__main__":
    main()
