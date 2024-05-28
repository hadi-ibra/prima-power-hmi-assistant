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
    context_relevancy,
    answer_correctness,
    answer_similarity,
)
from ragas.metrics.critique import harmfulness
from ragas import evaluate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
import toml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
# compressor = FlashrankRerank(model_name="ms-marco-MultiBERT-L-12")

with open('config.toml', 'r') as f:
    config = toml.load(f)
 
# Access values from the config



class RAGModel:
    def __init__(
        self,
        docs,
        dataset,
        k,
        llm_type,
        vector_store_type,
        reranking=False,
        refine_query=False,
        seed=None,
    ):
        load_dotenv()
        self.docs = docs
        self.dataset = dataset
        self.k = k
        self.llm_type = llm_type
        self.vector_store_type = vector_store_type
        self.reranking = reranking
        self.refine_query = refine_query
        self.huggingface_token = config['env']['HUGGINGFACEHUB_API_TOKEN']
        self.groq_api_key =  config['env']['GROQ_API_KEY']
        self.cohere_api_key = config['env']['COHERE_API_KEY']
        self.vector_store = None
        self.embeddings = None
        self.retriever = None
        self.retrievalQA = None
        self.prompt_template = None
        self.seed = seed

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
        logger.info("Setting up embeddings.")
        self.embeddings = OpenAIEmbeddings()

    def setup_vector_store(self):
        logger.info(f"Setting up vector store: {self.vector_store_type}.")
        if self.vector_store_type == "FAISS":
            self.vector_store = FAISS.from_documents(self.docs, self.embeddings)
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
            If the answer cannot be deduced from the context, use your own knowledge.
            Given the following context: {context}, answer the question: {question}""",
        )

    def setup_retriever(self):
        logger.info("Setting up retriever.")
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )

        if self.reranking:
            self.retriever = self.retriever_with_reranking()

    def setup_llm(self, name):
        logger.info(f"Setting up LLM: {name}.")
        if self.llm_type == "HuggingFace":
            self.llm = HuggingFaceHub(
                repo_id="meta-llama/Meta-Llama-3-70B",
                # model_kwargs={"temperature":0.1}
            )
        elif self.llm_type == "Groq":
            self.llm = ChatGroq(groq_api_key=self.groq_api_key, model_name=name)

    def refine_query_with_llm(self, original_query):
        logger.info(f"Refining query: {original_query}.")
        refined_query = self.llm.invoke(
            f"Refine the following query for better retrieval. Only return the refined query, without any additional text or explanation:\n\n{original_query}"
        )
        logger.info(f"Refined query: {refined_query}.")

        return refined_query.content

    # def rerank_results_colbert(self, query, source_docs):

    #     docs = self.retriever.invoke(query)
    #     docs=[doc.page_content for doc in docs]
    #     # Integrate the reranker
    #     logger.info("Reranking results.")
    #     reranked_results = RERANKER.rerank(query, docs, self.k)
    #     return reranked_results[: self.k]

    def retriever_with_reranking(
        self,
    ):
        compressor = CohereRerank(cohere_api_key=self.cohere_api_key)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.retriever
        )
        return compression_retriever

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
        logger.info("Refining the query {query}.")
        if self.refine_query:
            query = self.refine_query_with_llm(query)
        else:
            logger.info(f"Not Refined.")

        logger.info(f"Generating RAG answer for query: {query}.")
        result = self.retrievalQA.invoke({"query": query})
        answer = result["result"]
        source_documents = result["source_documents"]
        return answer, source_documents

    def create_ragas_dataset(self, column_name="ground_truth"):
        logger.info("Creating RAGAS dataset.")
        rag_dataset = []
        for index, row in tqdm(
            self.dataset.iterrows(),
            total=self.dataset.shape[0],
            desc="Processing questions",
        ):
            answer, source_docs = self.generate_rag_answer(row["question"])

            rag_dataset.append(
                {
                    "question": row["question"],
                    "answer": answer,
                    "contexts": [context.page_content for context in source_docs],
                    "ground_truths": [row[column_name]],
                }
            )
        rag_df = pd.DataFrame(rag_dataset)
        rag_eval_dataset = Dataset.from_pandas(rag_df)
        return rag_eval_dataset

    def evaluate(self, ragas_dataset):
        logger.info("Evaluating RAGAS dataset.")
        result = evaluate(
            ragas_dataset,
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
                context_relevancy,
                answer_correctness,
                answer_similarity,
            ],
        )
        logger.info("Evaluation complete.")
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
def main():
    logger.info("Starting RAG model setup.")
    pickle_file = "../data/docs.pkl"
    with open(pickle_file, "rb") as file:
        docs = pickle.load(file)
    dataset = pd.read_csv("../data/testset.csv")
    model = RAGModel(
        docs,
        dataset,
        k=5,
        llm_type="Groq",  # 'HuggingFace
        vector_store_type="FAISS",
        reranking=True,
        refine_query=True,
        seed=42,
    )
    model.setup_embeddings()
    model.setup_vector_store()
    model.setup_prompt_template()
    model.setup_retriever()
    model.setup_llm("llama3-70b-8192")
    model.setup_retrieverQA()
    ragas_dataset = model.create_ragas_dataset()
    evaluation_result = model.evaluate(ragas_dataset)
    logger.info("RAG Model setup and evaluation complete.")
    print(evaluation_result)

    evaluation_result_with_params = {
        "evaluation_result": evaluation_result,
        "model_params": model.get_model_params(),
    }
    model.save_results(
        evaluation_result_with_params, "evaluation_results_reranking.json"
    )


if __name__ == "__main__":
    main()
