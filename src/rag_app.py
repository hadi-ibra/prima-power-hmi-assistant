from experiments.rag import RAGModel
import streamlit as st
import os
import time
import pickle
import pandas as pd
import logging
import toml
from dotenv import load_dotenv
from langsmith import Client


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

with open("config.toml", "r") as f:
    config = toml.load(f)


st.set_page_config(
    page_title="Prima Power Demo",  # Title displayed on the browser tab
    page_icon="src/prima_power.ico",  # Favicon for the tab, you can also use a URL to an image
    # layout="centered"  # Optional: "centered" or "wide" layout
)

# st.markdown(
#     """
#     <style>
#     .main {
#         max-width: 1200px;
#         margin: 0 auto;
#         padding-top: 50px;
#     }
#     .stTextInput {
#         max-width: 600px; /* Set max width for input box */
#         margin: 0 auto;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = config["env"]["LANGCHAIN_API_KEY"]
os.environ["OPENAI_API_KEY"] = config["env"]["OPENAI_API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]=config["env"]["LANGCHAIN_PROJECT"]
os.environ["GROQ_API_KEY"]=config["env"]["GROQ_API_KEY"]

# Initialize a client



client = Client()

try:
    # Attempt to list existing projects
    existing_projects = client.list_projects()  
    project_name = "Prima Power Demo"
    
    # Attempt to find the project with the specified name
    project = next((proj for proj in existing_projects if proj.name == project_name), None)

    if project is None:
        # If the project doesn't exist, create a new one
        project = client.create_project(
            project_name=project_name,
            description="A project that answers questions about Tulus manual software",
        )
        logger.info(f"Created new project: {project_name}")
    else:
        logger.info(f"Using existing project: {project_name}")
except Exception as e:
    logger.error(f"An unexpected error occurred: {e}")


pickle_file = "src/docs2.pkl"
with open(pickle_file, "rb") as file:
    docs = pickle.load(file)

dataset = pd.read_csv("src/testset.csv")

# Cache the model setup to prevent reinitialization
@st.cache_resource
def setup_rag_model():
    model = RAGModel(
        docs,
        dataset,
        k=5,
        llm_type="Groq",  # 'HuggingFace' or 'Groq'
        vector_store_type="FAISS",
        reranking=True,
        method="ensemble",
        refine_query=True,
        embedding_model="hf_embeddings",
        model_name="llama-3.1-70b-Versatile",
        groq_api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
        seed=42,
    )
    
    model.setup_embeddings()
    model.setup_vector_store()
    model.setup_prompt_template()
    model.setup_llm()
    model.setup_retriever()
    model.setup_retrieverQA()
    
    return model

# Load or cache the model
if "model" not in st.session_state:
    st.session_state.model = setup_rag_model()

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
