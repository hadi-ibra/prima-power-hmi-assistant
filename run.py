from typing import List

import numpy as np
import copy

from tqdm import tqdm

from langchain_groq import ChatGroq

import numpy as np
import pandas as pd

import time
from src.config.args import get_parser
from argparse import Namespace
import numpy as np
import torch
import json

from src.logging.local_logger import LocalLoggerDecorator
from src.logging.logger import DummyLogger, Logger
from src.logging.wandb_logger import WandbLoggerDecorator

from src.experiments.few_shot import FewShotLearning


def get_args() -> Namespace:
    parser = get_parser()
    args = parser.parse_args()
    return args


def is_cuda_available() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("Current cuda device:", torch.cuda.current_device())
        print("Count of using GPUs:", torch.cuda.device_count())
        print(torch.cuda.get_device_name())
    return device


def get_model(args: Namespace, tokenizer, device):
    print(f"Initializing Model")
    # if args.load_checkpoint:
    #     return load_checkpoint(args)
    # else:
    #     return load_pretrained_model(args, tokenizer, device)


def get_logger(args: Namespace) -> Logger:
    logger = DummyLogger(args)

    if not args.not_use_local_logging:
        logger = LocalLoggerDecorator(logger)
    if not args.not_use_wandb:
        logger = WandbLoggerDecorator(logger)

    return logger


def main():

    start_time = time.time()

    # Create a new experiment
    np.random.seed(42)
    device = is_cuda_available()
    args = get_args()
    print(args)
    # args = {
    #     "framework": "few_shot_learning",
    #     "phase": "all",
    #     "seed": 42,
    # }

    gen = np.random.default_rng(args.seed)

    try:
        logger = get_logger(args)
        if args.framework == "few_shot_learning":

            experiment = FewShotLearning(
                args.model_name,
                args.train_dataset,
                "",
                args.test_dataset,
                0,
                2,
                gen,
                device,
                logger,
                args.groq_api_key,
            )

        if args.phase == "all":
            experiment.train()
            saving_object = experiment.save()
            if saving_object is not None:
                logger.save(saving_object)
            experiment.test()
        elif args.phase == "train":
            experiment.train()
            saving_object = experiment.save()
            logger.save(saving_object)
        elif args.phase == "test":
            experiment.test()
        # NOTE: use this settings ONLY with few-shot experiment
        elif args.phase == "metric":
            assert (
                args["framework"] == "few_shot_learning"
            ), "Use metric phase only with few-shot framework"
            assert (
                args.answers_folder is not None
            ), "Can't use metric phase without setting answers_folder param"
            with open(args.answers_folder) as f:
                result_file = json.load(f)
            answers = result_file["answers"]
            metrics = experiment._compute_metrics(answers)  # type: ignore
            logger.save_results({"answers": answers})
            logger.save_results(metrics)
            logger.summary(metrics)
        else:
            raise NotImplementedError("The phase chosen is not implemented")
    finally:
        logger.finish()
        torch.cuda.empty_cache()
        end_time = time.time()
        print(f"Elapsed time: {round(end_time - start_time, 2)}")


if __name__ == "__main__":
    main()
