from models.rag import RAGModel

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings
import time
import pickle
import pandas as pd
import numpy as np
import pickle
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

from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from dotenv import load_dotenv

load_dotenv()
## load the Groq API key


# pickle_file = "/Users/hadiibrahim/Dev/langchain-tutorial/groq/prima_power/chunks_qa.pkl"
# pickle_file1 = (
#     "/Users/hadiibrahim/Dev/langchain-tutorial/groq/prima_power/alarm_info_list.pkl"
# )
# with open(pickle_file, "rb") as file:
#     chunks = pickle.load(file)

# with open(pickle_file1, "rb") as file:
#     alarms = pickle.load(file)

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

pickle_file = "/workspaces/prima-power-hmi-assistant/src/data/docs.pkl"
with open(pickle_file, "rb") as file:
    docs = pickle.load(file)
dataset = pd.read_csv("/workspaces/prima-power-hmi-assistant/src/data/testset.csv")


if "model" not in st.session_state:
    model = RAGModel(
        docs,
        dataset,
        k=5,
        llm_type="Groq",  # 'HuggingFace' or 'Groq'
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
    st.session_state.model = model


st.title("Prima Power Demo")
# Prompt for user input
prompt = st.text_input("Input your prompt here")


if prompt:
    start = time.process_time()
    answer, source_documents = st.session_state.model.generate_rag_answer(prompt)
    response_time = time.process_time() - start
    st.write(f"Response time: {response_time:.2f} seconds")
    st.write(answer)

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(source_documents):
            st.write(doc.page_content)
            st.write("--------------------------------")


if st.button("Generate Test Set"):
    test_set_df = model.test_set_generation()
    st.write(test_set_df)

if st.button("Evaluate Model"):
    ragas_dataset = model.create_ragas_dataset()
    evaluation_result = model.evaluate(ragas_dataset)
    st.write(evaluation_result)
