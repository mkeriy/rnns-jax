class Logger:
    """Receives metrics from `fit`. Keys look like "train/loss" or "val/r2_scr"."""

    def log(self, metrics: dict[str, float], step: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class DictLogger(Logger):
    """Collects metrics in `history`: "train/loss" -> history["train_loss"].

    Also records the step of every call under "<prefix>_step", e.g. "val_step".
    """

    def __init__(self):
        self.history: dict[str, list[float]] = {}

    def log(self, metrics: dict[str, float], step: int) -> None:
        prefixes = set()
        for name, value in metrics.items():
            prefixes.add(name.split("/")[0])
            self.history.setdefault(name.replace("/", "_"), []).append(float(value))
        for prefix in prefixes:
            self.history.setdefault(f"{prefix}_step", []).append(step)


class TensorBoardLogger(Logger):
    def __init__(self, log_dir: str = "runs"):
        try:
            from tensorboardX import SummaryWriter
        except ImportError as e:
            raise ImportError(
                'TensorBoard logging needs tensorboardX: pip install "rnns-jax[tensorboard]"'
            ) from e
        self.writer = SummaryWriter(log_dir)

    def log(self, metrics: dict[str, float], step: int) -> None:
        for name, value in metrics.items():
            self.writer.add_scalar(name, float(value), step)

    def close(self) -> None:
        self.writer.close()


class WandbLogger(Logger):
    """Starts a W&B run; init_kwargs (project, name, entity, ...) go to wandb.init."""

    def __init__(self, config: dict | None = None, **init_kwargs):
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                'W&B logging needs wandb: pip install "rnns-jax[wandb]"'
            ) from e
        self.run = wandb.init(config=config, **init_kwargs)

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.run.log({name: float(value) for name, value in metrics.items()}, step=step)

    def close(self) -> None:
        self.run.finish()


def make_logger(name: str | None, params: dict, run_config: dict) -> Logger | None:
    """Build the logger named in TrainConfig.logger, or None for dict-only logging."""
    if name is None:
        return None
    if name == "tensorboard":
        return TensorBoardLogger(**params)
    if name == "wandb":
        return WandbLogger(config=run_config, **params)
    raise ValueError(f"unknown logger {name!r}; use 'tensorboard', 'wandb' or None")
