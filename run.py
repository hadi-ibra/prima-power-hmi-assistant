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
import pickle
from src.experiments.logging.local_logger import LocalLoggerDecorator
from src.experiments.logging.logger import DummyLogger, Logger
from src.experiments.logging.wandb_logger import WandbLoggerDecorator

from src.experiments.few_shot import FewShotLearning
from src.experiments.rag import RAGModel
from src.experiments.combined_rag_fewshot import CombinedRAGAndFewShot


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
    # Convert strings to boolean

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
                args.temperature,
                args.k_few_shot,
                gen,
                device,
                logger,
                args.groq_api_key,
            )
        elif args.framework == "rag":
            args.reranking = args.reranking.lower() == "true"
            args.refine_query = args.refine_query.lower() == "true"
            pickle_file = args.docs
            with open(pickle_file, "rb") as file:
                docs = pickle.load(file)
            dataset = pd.read_csv(args.test_dataset)
            print(
                "ARGUMENTS:",
                args.k_rag,
                args.vector_store_type,
                args.reranking,
                args.method,
                args.refine_query,
                args.embedding_model,
                args.model_name,
                args.groq_api_key,
                args.temperature,
                args.seed,
            )
            experiment = RAGModel(
                docs,
                dataset,
                args.k_rag,
                "Groq",
                args.vector_store_type,
                args.reranking,
                args.method,
                args.refine_query,
                args.embedding_model,
                args.model_name,
                args.groq_api_key,
                args.temperature,
                args.seed,
                logger,
                args.exp_name,
            )
        elif args.framework == "combined_rag_fewshot":
            args.reranking = args.reranking.lower() == "true"
            args.refine_query = args.refine_query.lower() == "true"
            logger = get_logger(args)
            pickle_file = args.docs
            with open(pickle_file, "rb") as file:
                docs = pickle.load(file)
            dataset = pd.read_csv(args.test_dataset)
            experiment = CombinedRAGAndFewShot(
                args.train_dataset,
                args.k_few_shot,
                docs,
                dataset,
                args.k_rag,
                "Groq",
                args.vector_store_type,
                args.reranking,
                args.method,
                args.refine_query,
                args.embedding_model,
                args.model_name,
                args.groq_api_key,
                args.temperature,
                args.seed,
                logger,
                args.exp_name,
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
