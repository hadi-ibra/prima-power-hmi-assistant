from experiments.rag import RAGModel

import streamlit as st
import os
import time
import pickle
import pandas as pd
import pickle
import logging
import toml

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
# logger.info("LANGHCAIN API KEY IS:",os.getenv("LANGCHAIN_API_KEY"))


with open("../config.toml", "r") as f:
    config = toml.load(f)

# Access values from the config

os.environ["LANGCHAIN_TRACING_V2"] = "true"

os.environ["LANGCHAIN_API_KEY"] = config["env"]["LANGCHAIN_API_KEY"]
os.environ["OPENAI_API_KEY"] = config["env"]["OPENAI_API_KEY"]


pickle_file = "docs.pkl"
with open(pickle_file, "rb") as file:
    docs = pickle.load(file)
dataset = pd.read_csv("testset.csv")


if "model" not in st.session_state:
    model = RAGModel(
        docs,
        dataset,
        k=5,
        llm_type="Groq",  # 'HuggingFace' or 'Groq'
        vector_store_type="FAISS",
        reranking=True,
        method = "ensemble",
        refine_query=True,
        embedding_model="hf_embeddings",
        model_name="llama-3.1-70b-Versatile",
        groq_api_key= os.environ["LANGCHAIN_API_KEY"],
        temperature = 0,
        seed=42,
    )


    model.setup_embeddings()
    model.setup_vector_store()
    model.setup_prompt_template()
    model.setup_llm()
    model.setup_retriever()
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
            st.write(f"Document {i}")
            st.write(doc.page_content)
            st.write("--------------------------------")


# if st.button("Generate Test Set"):
#     test_set_df = model.test_set_generation()
#     st.write(test_set_df)

# if st.button("Evaluate Model"):
#     ragas_dataset = model.create_ragas_dataset()
#     evaluation_result = model.evaluate(ragas_dataset)
#     st.write(evaluation_result)
