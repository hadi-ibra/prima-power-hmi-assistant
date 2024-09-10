# rag_model.py

import os
import random
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
from ragas.metrics.critique import harmfulness
from ragas import evaluate
from langchain.retrievers import ContextualCompressionRetriever
import toml

from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
)

from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    LLMListwiseRerank,
    LLMChainFilter,
    EmbeddingsFilter,
)
from overrides import overrides
from src.experiments.experiment import BasicExperiment
import evaluate as evaluate_module
from src.experiments.rag import RAGModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
# compressor = FlashrankRerank(model_name="ms-marco-MultiBERT-L-12")

with open("config.toml", "r") as f:
    config = toml.load(f)

# Access values from the config


class CombinedRAGAndFewShot(RAGModel):
    def __init__(self, train_ds, k_few_shot, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_ds = pd.read_csv(train_ds)
        self.k_few_shot = k_few_shot

    def setup_combined_prompt_template(self):
        logger.info("Setting up combined prompt template.")
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "examples"],
            template="""Here are some example questions and answers to help you provide a better response:
            {examples}

            Using the following context, give a comprehensive answer to the question.
            Respond only to the question asked, and be concise and relevant.
            Given the following context: {context},\n answer the question: {question}
            If the answer cannot be deduced from the context, use your own knowledge.""",
        )

    def _get_training_samples(self):
        idxes = np.random.randint(0, len(self.train_ds), size=self.k_few_shot)
        logger.info("Generating training samples for few-shot learning.")
        examples = []
        for idx in idxes:
            question, contexts, ground_truth, evolution_type, metadata, episode_done = (
                self.train_ds.iloc[idx]
            )
            examples.append((question, ground_truth))
        return examples

    def _gen_examples_template(self, training_examples: list):
        logger.info("Generating examples template for few-shot learning.")
        header = "You are a helpful assistant that answers questions about Tulus Manual software from Prima Power. Here are some examples:\n"
        tail = "Now, answer the following question.\nQuestion:"
        examples = []
        for question, answer in training_examples:
            template_example = f"Question: {question}\nAnswer: {answer}\n"
            examples.append(template_example)
        return header + " ".join(examples) + tail

    @overrides
    def generate_rag_answer(self, query):
        logger.info("Refining the query {query}.")
        if self.refine_query:
            query = self.refine_query_with_llm(query)
        else:
            logger.info(f"Not Refined.")

        # Adding few-shot examples
        training_examples = self._get_training_samples()
        examples_template = self._gen_examples_template(training_examples)

        logger.info(f"Generating RAG answer for query: {query}.")
        result = self.retrievalQA.invoke(
            {"query": query, "examples": examples_template}
        )
        answer = result["result"]
        source_documents = result["source_documents"]
        return answer, source_documents
