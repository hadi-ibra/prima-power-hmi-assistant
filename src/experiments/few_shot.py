import sys
import time
from typing import List

import numpy as np
import copy

from overrides import overrides
from rouge import Rouge
from bert_score import score
from tqdm import tqdm

from src.experiments.experiment import BasicExperiment
from langchain_groq import ChatGroq

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import score as bert_score
import pandas as pd

import evaluate


class FewShotLearning(BasicExperiment):

    def __init__(
        self,
        model,
        train_ds,
        eval_ds,
        test_ds,
        temperature,
        k,
        gen,
        device,
        logger,
        groq_api_key,
    ) -> None:
        super().__init__(model, train_ds, eval_ds, test_ds, gen, device, logger)
        self.temperature = temperature
        self.k = k
        self.use_temperature = True if temperature > 0 else False
        self.gen = gen
        self.device = device
        self.train_ds = pd.read_csv(self.train_ds)
        self.test_ds = pd.read_csv(self.test_ds)
        self.groq_api_key = groq_api_key
        if self.k != 0:
            examples = self._get_training_samples()
            self.base_prompt = self._gen_examples_template(examples)
        else:
            self.base_prompt = "Answer the following Question.\nQuestion:"

        self.model = ChatGroq(
            groq_api_key=self.groq_api_key,
            temperature=self.temperature,
            model_name=self.model,
            max_tokens=2048,
            verbose=True,
        )

    @overrides
    def train(self) -> None:
        pass
        # raise NotImplementedError(
        #     "Training for an LLM is not implemented since the required hardware capabilieties are too high"
        # )

    @overrides
    def test(self, **kwargs) -> None:
        answers = []

        # Assuming df3 is your dataframe
        for index, row in tqdm(self.test_ds.iterrows(), total=self.test_ds.shape[0]):
            # Access the value in the 'question' column
            prompt = self.base_prompt + row["question"] + "\Answer:"
            output_answer = None

            while output_answer is None:
                try:
                    output_answer = self.model.invoke(prompt)
                except Exception as e:
                    print(e)
                    print("Error generating answer. Retrying in 3 minutes...")
                    time.sleep(300)
            if not output_answer:
                output_answer = "model empty generation"
            answers.append((output_answer.content, row["ground_truth"]))
            if index != 0 and index % 100 == 0:
                print("Sleeping for 5 minutes")
                wait_time = 300
                time.sleep(wait_time)

        self.logger.save_results({"answers": answers})
        metrics = self._compute_metrics(answers)
        self.logger.save_results(metrics)
        self.logger.summary(metrics)
        return answers

    @overrides
    def save(self):
        pass

    def _get_training_samples(self):
        idxes = self.gen.integers(0, len(self.train_ds), size=self.k)
        print("TRAIN DS IS:", self.train_ds)
        examples = []
        for idx in idxes:
            question, contexts, ground_truth, evolution_type, metadata, episode_done = (
                self.train_ds.iloc[idx]
            )
            examples.append((question, ground_truth))
        return examples

    def _gen_examples_template(self, training_examples: List[str]) -> str:
        header = "You are a helpful assistant that answer mostly questions about Tulus Manual software from the company Prima Power. Answer the question below. But first to have a better understanding Here you can find some examples:\n"
        tail = "Answer the following question.\nQuestion:"
        examples = []
        for question, answer in training_examples:
            template_example = f"Question: {question}\nAnswer: {answer}\n"
            examples.append(template_example)
        return header + " ".join(examples) + tail

    def _compute_metrics(self, answers) -> dict:
        """
        Calculate various metrics for generated answers compared to ground truth answers.

        Parameters:
        answers (list of tuples): List of (generated_answer, correct_answer) tuples.

        Returns:
        dict: Dictionary containing metric scores.
        """
        # Initialize metrics

        # Load the metrics
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        bert_score = evaluate.load("bertscore")
        result = {}
        score_tot = {}

        generated_answers = []
        correct_answers = []

        for generated_answer, correct_answer in answers:
            generated_answers.append(generated_answer)
            correct_answers.append(correct_answer)

        score_tot["bleu_result"] = bleu.compute(
            predictions=generated_answers, references=[[ans] for ans in correct_answers]
        )

        # Compute the ROUGE score
        score_tot["rouge_result"] = rouge.compute(
            predictions=generated_answers, references=correct_answers
        )

        bert_result = bert_score.compute(
            predictions=generated_answers, references=correct_answers, lang="en"
        )

        mean_precision = np.mean(bert_result["precision"])
        mean_recall = np.mean(bert_result["recall"])
        mean_f1 = np.mean(bert_result["f1"])
        bert_result_main = {}
        # Add the mean scores to the bert_result dictionary
        bert_result_main["mean_precision"] = mean_precision
        bert_result_main["mean_recall"] = mean_recall
        bert_result_main["mean_f1"] = mean_f1

        # Compute the BERT score
        score_tot["bert_result"] = bert_result_main
        result[f"summary"] = copy.deepcopy(score_tot)
        return result


# def main():

#     try:
#         # Create a new experiment
#         np.random.seed(42)
#         args = get_arg()
#         print(args)
#         args = {
#             "phase": "all",
#         }

#         experiment = FewShotLearning(
#             "meta-llama/Meta-Llama-3-70B",
#             "/Users/hadiibrahim/Dev/prima-power-hmi-assistant/src/data/trainset_manual.csv",
#             "/Users/hadiibrahim/Dev/prima-power-hmi-assistant/src/data/testset_manual.csv",
#             0,
#             2,
#         )

#         if args.phase == "all":
#             experiment.train()
#             saving_object = experiment.save()
#             if saving_object is not None:
#                 logger.save(saving_object)
#             experiment.test(**test_kwargs)
#         elif args.phase == "train":
#             experiment.train()
#             saving_object = experiment.save()
#             logger.save(saving_object)
#         elif args.phase == "test":
#             experiment.test(**test_kwargs)
#         # NOTE: use this settings ONLY with few-shot experiment
#         elif args.phase == "metric":
#             assert (
#                 args.framework == "few_shot_learning"
#             ), "Use metric phase only with few-shot framework"
#             assert (
#                 args.answers_folder is not None
#             ), "Can't use metric phase without setting answwers_folder param"
#             with open(args.answwers_folder) as f:
#                 result_file = json.load(f)
#             summaries = result_file["summaries"]
#             metrics = experiment._compute_metrics(summaries)  # type: ignore
#             logger.save_results({"summaries": summaries})
#             logger.save_results(metrics)
#             logger.summary(metrics)
#         else:
#             raise NotImplementedError("The phase chosen is not implemented")
#     finally:
#         logger.finish()
#         torch.cuda.empty_cache()
#         end_time = time.time()
#         print(f"Elapsed time: {round(end_time - start_time, 2)}")


# if __name__ == "__main__":
#     main()
