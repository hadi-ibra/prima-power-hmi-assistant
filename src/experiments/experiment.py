from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from overrides import overrides
from rouge import Rouge
from bert_score import score

from experiments.logging.logger import Logger


class Experiment(ABC):

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def test(self, **kwargs) -> None:
        pass

    @abstractmethod
    def save(self) -> Any:
        pass


class BasicExperiment(Experiment):

    def __init__(
        self,
        model,
        train_ds,
        eval_ds,
        test_ds,
        gen: np.random.Generator,
        device,
        logger: Logger,
    ) -> None:
        self.model = model
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.test_ds = test_ds
        self.gen = gen
        self.device = device
        self.logger = logger
