# rag_model.py
import sys
import os
import streamlit as st

import os
import random
import time
import numpy as np
import logging
import json
import pickle
from dotenv import load_dotenv
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_groq import ChatGroq
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity,
)
import re
import traceback

# from ragas.metrics.critique import harmfulness
from ragas import evaluate
from langchain.retrievers import ContextualCompressionRetriever
import toml

from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
)
import rank_bm25
from langchain.retrievers import BM25Retriever, EnsembleRetriever

from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    LLMListwiseRerank,
    LLMChainFilter,
    EmbeddingsFilter,
)

from overrides import overrides
from experiments.experiment import BasicExperiment
import evaluate as evaluate_module

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import copy

import os

# RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
# compressor = FlashrankRerank(model_name="ms-marco-MultiBERT-L-12")

with open("config.toml", "r") as f:
    config = toml.load(f)

# Access values from the config

os.environ["OPENAI_API_KEY"] = config["env"]["OPENAI_API_KEY"]


class RAGModel(BasicExperiment):
    def __init__(
        self,
        docs,
        dataset,
        k,
        llm_type,
        vector_store_type,
        reranking=False,
        method="llm_listwise_rerank",
        refine_query=False,
        embedding_model="hf_embeddings",
        model_name="llama-3.1-70b-Versatile",
        groq_api_key=None,
        temperature=0,
        seed=None,
        logger2=None,
        exp_name="exp",
    ):
        load_dotenv()
        self.docs = docs
        self.dataset = dataset
        self.k = k
        self.llm_type = llm_type
        self.vector_store_type = vector_store_type
        self.reranking = reranking
        self.method = method
        self.refine_query = refine_query
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.temperature = temperature
        self.huggingface_token = config["env"]["HUGGINGFACEHUB_API_TOKEN"]
        self.groq_api_key = groq_api_key
        self.cohere_api_key = config["env"]["COHERE_API_KEY"]
        self.openai_api_key = config["env"]["OPENAI_API_KEY"]
        self.vector_store = None
        self.embeddings = None
        self.retriever = None
        self.retrievalQA = None
        self.prompt_template = None
        self.seed = seed
        self.logger2 = logger2
        self.exp_name = exp_name

        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed):
        logger.info(f"Setting random seed: {seed}.")
        random.seed(seed)
        np.random.seed(seed)
        # Set seed for other libraries if necessary
        # Example for TensorFlow: tf.random.set_seed(seed)
        # Example for PyTorch: torch.manual_seed(seed)
        # If using CUDA: torch.cuda.manual_seed_all(seed)

    def setup_embeddings(self):
        if self.embedding_model == "openai":
            self.embeddings = OpenAIEmbeddings()

        elif self.embedding_model == "hf_embeddings":
            #  HuggingFaceEmbeddings(
            #  model_name="mixedbread-ai/mxbai-embed-large-v1",
            #  model_kwargs={"device":"cpu"},
            #  encode_kwargs={"normalize_embeddings":True}
            # )

            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",  # or sentence-trainsformers/all-MiniLM-L6-v2
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        logger.info("Setting up embeddings.")

    def setup_vector_store(self):
        logger.info(f"Setting up vector store: {self.vector_store_type}.")
        if self.vector_store_type == "FAISS":
            self.vector_store = FAISS.from_documents(self.docs, self.embeddings)
            logger.info(f"Setting up vector store: {self.vector_store_type}.")
        elif self.vector_store_type == "Chroma":
            self.vector_store = Chroma.from_documents(self.docs, self.embeddings)
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")

    def setup_prompt_template(self):
        logger.info("Setting up prompt template.")
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=""" Using the information contained in the context, give a comprehensive answer to the question.
            Respond only to the question asked, response should be concise and relevant to the question.
            Given the following context: {context},\n answer the question: {question}
            If the answer cannot be deduced from the context, use your own knowledge.""",
        )

    def setup_retriever(self):
        # TODO: use this ensemble retrieval  个

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )
        logger.info("Setting up normal retriever.")

        if self.reranking:
            self.retriever = self.retriever_with_reranking()
            logger.info("Setting up advanced retriever.")

        else:
            logger.info("Not set up with reranker.")

    def setup_llm(self):
        logger.info(f"Setting up LLM.")
        if self.llm_type == "HuggingFace":
            self.llm = HuggingFaceHub(
                repo_id="meta-llama/Meta-Llama-3-70B",
                # model_kwargs={"temperature":0.1}
            )
        elif self.llm_type == "Groq":
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model=self.model_name,
                temperature=self.temperature,
            )

    def refine_query_with_llm(self, original_query):
        logger.info(f"Refining query: {original_query}.")
        refined_query = self.llm.invoke(
            f"Refine the following query for better retrieval. Only return the refined query, without any additional text or explanation:\n\n{original_query}"
        )
        logger.info(f"Refined query: {refined_query}.")

        return refined_query.content

        # Function to refine the query using an LLM

    def compute_weighted_embedding(self, text, alarm_id=None):
        """Compute embedding and apply extra weight to Alarm ID if present."""
        # Compute the normal embedding first
        embedding = np.array(
            self.embeddings.embed_query(text)
        )  # Convert to NumPy array

        if alarm_id:
            # Emphasize the alarm ID by increasing its contribution to the overall embedding
            alarm_embedding = np.array(
                self.embeddings.embed_query(alarm_id)
            )  # Convert to NumPy array

            # Apply a higher weight to the alarm ID embedding (e.g., 2x)
            weighted_embedding = 0.5 * embedding + 1.5 * alarm_embedding
            return weighted_embedding

        return embedding

    def retrieve_with_weighted_embedding(self, query, alarm_id=None):
        """Retrieve documents using embeddings, giving weight to Alarm ID if applicable."""
        if alarm_id:
            # Compute a weighted embedding to emphasize the Alarm ID
            embedding = self.compute_weighted_embedding(query, alarm_id)
        else:
            # Use the default query embedding
            embedding = self.embeddings.embed_query(query)

        # Perform the retrieval using the computed embedding
        results = self.vector_store.similarity_search_by_vector(embedding, k=self.k)
        return results

    def refine_query_with_alarm(self, query, alarm_id):
        llm_prompt = f"""
            You are an intelligent assistant refining queries for retrieving specific alarm information. The user has provided an Alarm ID, and it is critical to emphasize this ID in the refined query to ensure correct retrieval.
            
            Alarm ID: {alarm_id}
            
            Please rewrite the following query, making the Alarm ID the most prominent aspect of the refined query. The retrieval system will prioritize this Alarm ID to find the relevant information, so ensure it is explicitly highlighted and given higher importance than other parts of the query.
            
            Original query: "{query}"
            
            Refined query (with Alarm ID emphasized):
        """
        refined_query = self.llm.invoke(llm_prompt)
        logger.info(f"Refined query: {refined_query}.")

        return refined_query.content

    # Function to check if the query is alarm-related and contains an Alarm ID

    # Function to check if the query is alarm-related and contains a valid Alarm ID
    def is_alarm_query(self, query):
        # Check if the word "alarm" is in the query
        if "alarm" not in query.lower():
            return False, None

        # Use regex to find a possible Alarm ID (including formats like "324/322")
        alarm_id_pattern = r"\b\d+(?:/\d+)?\b"
        alarm_ids = re.findall(alarm_id_pattern, query)

        # Return True if any Alarm ID is found, otherwise return False
        return bool(alarm_ids), alarm_ids[0] if alarm_ids else None

    def retriever_with_reranking(
        self,
    ):

        if self.method == "llm_chain_extractor":
            print("llm_chain_extractor")
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=self.retriever
            )
        elif self.method == "ensemble":
            print("ensemble")
            bm25_retriever = BM25Retriever.from_documents(self.docs)
            bm25_retriever.k = self.k

            retriever = self.vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": self.k}
            )
            compression_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, retriever]
            )
            logger.info("Setting up ensemble retriever.")

        elif self.method == "llm_listwise_rerank":
            print("llm_listwise_rerank")
            _filter = LLMListwiseRerank.from_llm(self.llm, top_n=4)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=_filter, base_retriever=self.retriever
            )

        elif self.method == "llm_chain_filter":
            print("llm_chain_filter")
            _filter = LLMChainFilter.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=_filter, base_retriever=self.retriever
            )
        elif self.method == "embeddings_filter":
            print("embeddings_filter")
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.embeddings, similarity_threshold=0.76
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter, base_retriever=self.retriever
            )

        return compression_retriever

    @overrides
    def train(self) -> None:
        pass
        # raise NotImplementedError(
        #     "Training for an LLM is not implemented since the required hardware capabilieties are too high"
        # )

    @overrides
    def save(self):
        pass

    def setup_retrieverQA(self):
        logger.info("Setting up RetrievalQA chain.")
        self.retrievalQA = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template},
        )

    def generate_rag_answer(self, query):
        logger.info(f"Processing the query: {query}")

        # Check if the query is about an alarm and extract the Alarm ID
        is_alarm, alarm_id = self.is_alarm_query(query)
        refined_query = query

        if is_alarm and alarm_id:
            logger.info(f"Query is about an alarm: {alarm_id}")
            refined_query = self.retry_until_success(
                self.refine_query_with_alarm,
                query,
                error_msg="Error refining query with LLM.",
            )
            logger.info(f"Refined query: {refined_query}")
        else:
            logger.info(f"Query is not about an alarm: {query}")

            # Refine the query using LLM if required
            if self.refine_query:
                refined_query = self.retry_until_success(
                    self.refine_query_with_llm,
                    query,
                    error_msg="Error refining query with LLM.",
                )

            else:
                logger.info("Query refinement not required.")

        result = self.retry_until_success(
            self.retrievalQA.invoke,
            {"query": refined_query},
            error_msg="Error generating RAG answer.",
        )

        # Return the answer and source documents
        if not result:
            return "model empty generation", None

        return result["result"], result["source_documents"]

    def retry_until_success(self, func, *args, error_msg="", sleep_time=60, **kwargs):
        result = None
        while result is None:
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{error_msg} {str(e)}")
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
        return result

    def create_ragas_dataset(self, column_name="ground_truth"):
        logger.info("Creating RAGAS dataset.")
        file_name = f"ragas_dataset_{self.exp_name}.csv"
        file_created = False
        self.dataset = self.dataset[759:]

        rag_dataset = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
            "original_contexts": [],
            "retrieved_contexts": [],
        }
        # self.dataset = self.dataset[763:]

        for index, row in tqdm(
            self.dataset.iterrows(),
            total=self.dataset.shape[0],
            desc="Processing questions",
        ):
            answer, source_docs = self.generate_rag_answer(row.question)

            # Append the extracted elements to the rag_dataset
            rag_dataset["question"].append(row.question)
            rag_dataset["answer"].append(answer)
            rag_dataset["contexts"].append(
                [context.page_content for context in source_docs]
            )
            rag_dataset["ground_truth"].append(row["ground_truth"])
            rag_dataset["original_contexts"].append(row.contexts)
            rag_dataset["retrieved_contexts"].append(source_docs)
            # Convert to DataFrame after each iteration
            rag_df = pd.DataFrame(
                {
                    "question": [row.question],
                    "answer": [answer],
                    "contexts": [[context.page_content for context in source_docs]],
                    "ground_truth": [row["ground_truth"]],
                    "original_contexts": [row.contexts],
                }
            )
            # Save to CSV after every iteration
            if not file_created:
                rag_df.to_csv(
                    file_name, index=False, mode="w"
                )  # Write the header on the first save
                file_created = True
            else:
                rag_df.to_csv(
                    file_name, index=False, mode="a", header=False
                )  # Append without headers

        # Convert to DataFrame once after loop completion
        rag_df = pd.DataFrame(rag_dataset)

        rag_df.to_csv(file_name, index=False)
        # delete original_contexts key from the dictionary
        del rag_dataset["original_contexts"]
        del rag_dataset["retrieved_contexts"]
        answers = list(
            zip(
                rag_df["answer"], rag_df["ground_truth"]
            ),  # (Generated answer, Correct answer)# (Retrieved contexts, Original contexts)
        )

        self.logger2.save_results({"answers": answers})

        rag_eval_dataset = Dataset.from_dict(rag_dataset)
        metrics = self._compute_metrics(answers, rag_eval_dataset)
        self.logger2.save_results(metrics)

        # Convert the dictionary to a Dataset directly

        return rag_eval_dataset

        # Function to save progress

    def save_progress(self, result, save_path="evaluation_progress.csv"):
        df = pd.DataFrame(result)
        df.to_csv(save_path, index=False)

    # Function to load progress
    def load_progress(self, save_path="evaluation_progress.csv"):
        try:
            df = pd.read_csv(save_path)
            return df.to_dict(orient="list")
        except FileNotFoundError:
            return {
                metric: []
                for metric in [
                    "answer_relevancy",
                    "faithfulness",
                    "context_recall",
                    "context_precision",
                    "answer_correctness",
                    "answer_similarity",
                ]
            }

    # Define a function for evaluation with retry mechanism and progress saving
    def evaluate_with_resume(
        self,
        dataset,
        metrics,
        llm,
        embeddings,
        retries=5,
        sleep_time=300,
        save_path="evaluation_progress.csv",
    ):
        progress = self.load_progress(save_path)
        start_index = len(progress)  # Start from where we left off

        while True:
            try:
                for i in range(start_index, len(dataset)):
                    # Evaluate a single item
                    result = evaluate(
                        Dataset.from_dict(
                            dataset[i : i + 10]
                        ),  # Evaluate one item at a time
                        metrics=metrics,
                        llm=llm,
                        embeddings=embeddings,
                    )

                    # Append each metric result to the corresponding list in the progress dictionary
                    for metric, value in result.items():
                        progress[metric].append(value)
                    self.save_progress(
                        progress, save_path
                    )  # Save progress after each step

                return progress  # Return the full results if successful

            except Exception as e:
                print(f"Sleeping for {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
                print("Retrying...")

    def _compute_metrics(self, answers, ragas_dataset) -> dict:
        """
        Calculate various metrics for generated answers compared to ground truth answers
        and for retrieved contexts compared to original contexts.

        Parameters:
        answers_and_contexts (list of tuples): List of ((generated_answer, correct_answer), (retrieved_contexts, original_contexts)) tuples.

        Returns:
        dict: Dictionary containing metric scores.
        """
        # Load the metrics
        bleu = evaluate_module.load("bleu")
        rouge = evaluate_module.load("rouge")
        bert_score = evaluate_module.load("bertscore")
        precision_metric = evaluate_module.load("precision")
        recall_metric = evaluate_module.load("recall")

        result = {}
        score_tot = {}

        generated_answers = []
        correct_answers = []
        retrieved_contexts = []
        original_contexts = []

        for generated_answer, correct_answer in answers:
            generated_answers.append(generated_answer)
            correct_answers.append(correct_answer)
            # retrieved_contexts.append(
            #     " ".join(retrieved_context)
            # )  # Join the retrieved context passages
            # original_contexts.append(
            #     " ".join(original_context)
            # )  # Join the original context passages

        # Compute BLEU score for answers
        score_tot["bleu_result"] = bleu.compute(
            predictions=generated_answers, references=[ans for ans in correct_answers]
        )

        # Compute ROUGE score for answers
        score_tot["rouge_result"] = rouge.compute(
            predictions=generated_answers, references=correct_answers
        )

        # Compute BERT score for answers
        bert_result = bert_score.compute(
            predictions=generated_answers, references=correct_answers, lang="en"
        )
        score_tot["bert_result"] = {
            "mean_precision": np.mean(bert_result["precision"]),
            "mean_recall": np.mean(bert_result["recall"]),
            "mean_f1": np.mean(bert_result["f1"]),
        }

        # # Compute Precision for retrieved contexts
        # score_tot["context_precision"] = precision_metric.compute(
        #     predictions=retrieved_contexts, references=original_contexts
        # )["precision"]

        # # Compute Recall for retrieved contexts
        # score_tot["context_recall"] = recall_metric.compute(
        #     predictions=retrieved_contexts, references=original_contexts
        # )["recall"]
        logger.info("Evaluating RAGAS dataset.")

        # List of metrics to be used for evaluation
        metrics = [
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
            answer_correctness,
            answer_similarity,
        ]

        # Run the evaluation with the resume and retry mechanism
        file_name = f"evaluation_results_{self.exp_name}.csv"
        # result = self.evaluate_with_resume(
        #     ragas_dataset,
        #     metrics=metrics,
        #     llm=self.llm,
        #     embeddings=self.embeddings,
        #     save_path=file_name,
        # )
        print("Evaluation completed successfully.")

        # score_tot["ragas_scores"] = result
        logger.info("Evaluation complete.")
        result["summary"] = copy.deepcopy(score_tot)
        return result

    def test_set_generation(self):
        logger.info("Generating test set.")
        generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
        critic_llm = ChatOpenAI(model="gpt-4o")
        embeddings = OpenAIEmbeddings()
        generator = TestsetGenerator.from_langchain(
            generator_llm, critic_llm, embeddings
        )
        distributions = {simple: 0.5, multi_context: 0.4, reasoning: 0.1}
        testset = generator.generate_with_langchain_docs(self.docs, 10, distributions)
        logger.info("Test set generation complete.")
        return pd.DataFrame(testset.to_pandas())

    def save_results(self, results, filepath):
        logger.info(f"Saving results to {filepath}.")
        with open(filepath, "w") as file:
            json.dump(results, file, indent=4)

    def get_model_params(self):
        params = {
            "k": self.k,
            "llm_type": self.llm_type,
            "vector_store_type": self.vector_store_type,
            "reranking": self.reranking,
            "refine": self.refine_query,
            "seed": self.seed,
            "huggingface_token": self.huggingface_token,
            "groq_api_key": self.groq_api_key,
        }
        return params

    # Main function for setup and testing
    @overrides
    def test(self, **kwargs) -> None:
        logger.info("Starting RAG model setup.")

        self.setup_embeddings()
        self.setup_vector_store()
        self.setup_prompt_template()
        self.setup_llm()
        self.setup_retriever()
        self.setup_retrieverQA()
        ragas_dataset = self.create_ragas_dataset()
        # evaluation_result = self.evaluate(ragas_dataset)
        logger.info("RAG Model setup and evaluation complete.")
        # print(evaluation_result)

        # evaluation_result_with_params = {
        #     "evaluation_result": evaluation_result,
        #     "model_params": self.get_model_params(),
        # }
        # self.save_results(
        #     evaluation_result_with_params, "evaluation_results_reranking.json"
        # )
