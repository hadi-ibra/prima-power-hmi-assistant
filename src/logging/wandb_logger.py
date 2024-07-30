import json
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo
from datetime import datetime as dt
from overrides import overrides
import wandb

from src.logging.logger import BaseDecorator, Logger


class WandbLoggerDecorator(BaseDecorator):

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        wandb.init(
            project=self.args.project, name=self.args.exp_name, config=vars(self.args)
        )

    @overrides
    def save(self, trainer) -> None:
        snapshot_tmp_path = Path(wandb.run.dir).joinpath("tmp_model_folder")
        trainer.save_model(snapshot_tmp_path)
        wandb.save(snapshot_tmp_path.as_posix(), policy="now")
        self.logger.save(trainer)

    @overrides
    def summary(self, data: Dict[str, Any]) -> None:
        for k, v in data.items():
            wandb.summary[k] = v
        self.logger.summary(data)

    @overrides
    def finish(self) -> None:
        wandb.finish()
        self.logger.finish()
