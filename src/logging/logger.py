from argparse import Namespace
from datetime import datetime as dt
import enum
import json
from zoneinfo import ZoneInfo
from abc import ABC, abstractmethod
from overrides import overrides
from typing import Any, Dict, Literal, Optional
from pathlib import Path

# from experiment.snapshot import Snapshot


class Logger(ABC, object):

    def __init__(self, args: Namespace) -> None:
        self._args = args

    @property
    def args(self) -> Namespace:
        return self._args

    @abstractmethod
    def log(self, data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def save(self, trainer) -> None:
        pass

    @abstractmethod
    def summary(self, data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def save_results(self, data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass


class DummyLogger(Logger):

    @overrides
    def log(self, data: Dict[str, Any]) -> None:
        pass

    @overrides
    def save(self, trainer) -> None:
        pass

    @overrides
    def save_results(self, data: Dict[str, Any]) -> None:
        pass

    @overrides
    def summary(self, data: Dict[str, Any]) -> None:
        pass

    @overrides
    def finish(self) -> None:
        pass


class BaseDecorator(Logger):

    _logger: Logger = None

    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def args(self) -> Namespace:
        return self._logger.args

    @overrides
    def log(self, data: Dict[str, Any]) -> None:
        self._logger.log(data)

    @overrides
    def save(self, trainer) -> None:
        self._logger.save(trainer)

    @overrides
    def save_results(self, data: Dict[str, Any]) -> None:
        self._logger.save_results(data)

    @overrides
    def summary(self, data: Dict[str, Any]) -> None:
        self._logger.summary(data)

    @overrides
    def finish(self) -> None:
        self._logger.finish()
