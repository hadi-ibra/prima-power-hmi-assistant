import json
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo
from datetime import datetime as dt
from overrides import overrides

from src.logging.logger import BaseDecorator, Logger


class LocalLoggerDecorator(BaseDecorator):

    root = Path("models_saved/")

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        if not LocalLoggerDecorator.root.exists():
            LocalLoggerDecorator.root.mkdir()
        self.exp_dir = self._create_dir()
        self._save_params(vars(self.args))

    @overrides
    def save(self, trainer) -> None:
        trainer.save_model(self.exp_dir.joinpath(self.args.best_finetune_weight_path))
        self.logger.save(trainer)

    @overrides
    def save_results(self, data: Dict[str, Any]) -> None:
        # uglyest name ever given to an argument but it's in the original project so that's it
        file_path = self.exp_dir.joinpath(self.args.test_output_file_name)
        with open(file_path, "a") as outfile:
            json.dump(data, outfile)
        self.logger.save_results(data)

    def _create_dir(self) -> Path:
        timezone = ZoneInfo("Europe/Rome")
        time = dt.now(timezone).isoformat(timespec="minutes")
        exp_dir = LocalLoggerDecorator.root.joinpath(f"{time}-{self.args.exp_name}")
        if not exp_dir.exists():
            exp_dir.mkdir()
        elif not exp_dir.is_dir():
            raise Exception("LocalLogger: Already exist a folder with the same name")
        return exp_dir

    def _save_params(self, data: Dict[str, Any]) -> None:
        file_path = self.exp_dir.joinpath(f"params.json")
        with open(file_path, "a") as outfile:
            json.dump(data, outfile)
