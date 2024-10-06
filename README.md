## Project Overview

This project presents the development and evaluation of an AI Assistant aimed at enhancing interactions between operators and the Human-Machine Interface (HMI) of the Laser Genius+ machine by Prima Power. The assistant is designed to interpret and respond to natural language queries, providing relevant information from machine manuals. Two primary AI approaches were explored: **Retrieval-Augmented Generation (RAG)** and **Few-Shot Learning**, to improve operational efficiency within industrial settings.

The project involved:

1. **Text and Image Preprocessing**: Extracting and processing content from the machine manuals.
2. **Dataset Creation**: Generating question-answer pairs to train the AI assistant.
3. **Model Development**: Experimenting with RAG, few-shot learning, and a combined approach to generate accurate, context-aware responses.
4. **Evaluation**: Testing models using metrics like Cosine Similarity, ROUGE scores, Recall, and Mean Reciprocal Rank (MRR).

The AI assistant uses retrieval mechanisms integrated with large language models (LLMs) for precise and contextually relevant answers, setting a foundation for future AI-assisted interfaces in industrial automation.

## Experimental Approaches

### Retrieval-Augmented Generation (RAG)

RAG leverages information retrieval to inform LLM outputs, grounding responses in accurate and context-specific details from domain documents. The approach utilizes different retrievers and embedding models for optimized retrieval and response accuracy.

### Few-Shot Learning

Few-shot learning enables the assistant to generalize from a limited set of examples, making it suitable for contexts with minimal training data. This technique aids in adapting to new queries and domain-specific language through carefully designed prompts.

### Combined RAG and Few-Shot Learning

The combined model leverages RAG's retrieval capabilities along with the adaptability of few-shot learning, enhancing the model's response quality by incorporating both approaches.

### Streamlit Deployment

A user-friendly Streamlit interface was developed for real-time interaction, allowing operators to efficiently use the AI assistant within industrial settings.

## Results Summary

The experiments showed that the RAG model, particularly when using ensemble retriever strategies, significantly improved retrieval quality, thus enhancing response relevance and accuracy. The combined approach of RAG and few-shot learning added further guidance to improve results.

## How to Run the Code

**Environment Setup**

1. Clone the repository:
   ```bash
   git clone https://github.com/hadi-ibra/SICK_Summarization
   ```
2. Create an environment using conda (or virtualenv):
   ```bash
   conda create -n example_env
   conda activate example_env
   pip install -r requirements.txt
   ```

To run experiments, use:

```bash
python3 run.py <PARAMS>
```

**Optional Wandb Logging**: To use Weights & Biases (wandb) for logging, log in with:

```bash
wandb login <TOKEN>
```

If you prefer not to use wandb, include `--not_use_wandb` in your arguments.

To run variations of the presented experiment categories, change the `--framework` argument:

- `"rag"`
- `"few_shot_learning"`
- `"combined_rag_fewshot"`

### Prerequisites

- A valid `hugging_face_token` to access models from Hugging Face.
- A valid `groq_api_key` for using the Groq Cloud API.
- Datasets in `.csv` format for training and testing.

---

## Running Experiments

### 1. RAG Experiments

**Configuration Arguments**

- **hugging_face_token**: Token to access Hugging Face.
- **docs**: Path to preprocessed documents (`.pkl` format).
- **project**: Project name for logging in wandb.
- **framework**: Set to `"rag"` for RAG experiments.
- **exp_name**: Unique experiment name for logging.
- **seed**: Seed value (e.g., `516` for reproducibility).
- **phase**: Experiment phase (training, testing).
- **model_name**: Model name (e.g., `"llama-3.1-70b-Versatile"`).
- **temperature**: Controls randomness in model output.
- **k_rag**: Number of documents retrieved for context.
- **test_dataset**: Path to test dataset (`.csv`).
- **train_dataset**: Path to training dataset (`.csv`).
- **vector_store_type**: Vector storage type (`FAISS`).
- **embedding_model**: Model for generating embeddings.
- **reranking**: Whether to rerank retrieved documents.
- **refine_query**: Whether to refine queries before retrieval.
- **method**: Information extraction method (e.g., `llm_chain_extractor`).
- **groq_api_key**: API key for Groq Cloud resources.

Example command:

```bash
python run.py --hugging_face_token <token> \
              --docs /path/to/docs.pkl \
              --project rag \
              --framework rag \
              --exp_name rag_exp \
              --seed 516 \
              --phase all \
              --model_name llama-3.1-70b-Versatile \
              --temperature 0 \
              --k_rag 5 \
              --test_dataset /path/to/testset.csv \
              --train_dataset /path/to/trainset_manual.csv \
              --vector_store_type FAISS \
              --embedding_model hf_embeddings \
              --reranking False \
              --refine_query False \
              --method llm_chain_extractor \
              --groq_api_key <token>
```

### 2. Few-Shot Learning Experiments

**Configuration Arguments**

- **hugging_face_token**: Token for Hugging Face access.
- **project**: Project name for wandb logging.
- **framework**: Set to `"few_shot_learning"`.
- **exp_name**: Experiment name for logging.
- **seed**: Seed for reproducibility (e.g., `516`).
- **phase**: Experiment phase.
- **temperature**: Model temperature value.
- **k**: Number of examples in the prompt.
- **model_name**: Model name.
- **test_dataset**: Path to test dataset.
- **train_dataset**: Path to training dataset.

Example command:

```bash
python run.py --hugging_face_token <token> \
              --project "few_shot" \
              --framework "few_shot_learning" \
              --exp_name "few_shot_t0.4_k2" \
              --seed 516 \
              --phase "all" \
              --model_name "llama-3.1-70b-Versatile" \
              --temperature 0.4 \
              --k_few_shot 2 \
              --test_dataset /path/to/final_selection.csv \
              --train_dataset /path/to/trainset_manual.csv \
              --groq_api_key <token>
```

### 3. Combined RAG and Few-Shot Experiments

**Configuration Arguments**

- **hugging_face_token**: Token for Hugging Face.
- **docs**: Path to preprocessed documents.
- **project**: Project name for wandb.
- **framework**: Set to `"combined_rag_fewshot"`.
- **exp_name**: Unique experiment name for logging.
- **seed**: Seed for reproducibility.
- **phase**: Experiment phase.
- **model_name**: Model to be used.
- **temperature**: Model output temperature.
- **k_rag**: Number of documents retrieved.
- **k_few_shot**: Few-shot examples added to prompt.
- **test_dataset**: Path to test dataset.
- **train_dataset**: Path to training dataset.
- **vector_store_type**: Type of vector store.
- **embedding_model**: Model type for embeddings.
- **reranking**: Whether to rerank retrieved documents.
- **refine_query**: Option to refine query before retrieval.
- **method**: Retrieval and extraction method (e.g., `ensemble`).
- **groq_api_key**: API key for Groq Cloud.

Example command:

```bash
python run.py --hugging_face_token <token> \
              --docs /path/to/docs.pkl \
              --project combined_rag_fewshot \
              --framework combined_rag_fewshot \
              --exp_name combined_rag_fewshot_best \
              --seed 516 \
              --phase all \
              --model_name llama-3.1-70b-Versatile \
              --temperature 0 \
              --k_rag 5 \
              --k_few_shot 4 \
              --test_dataset /path/to/final_selection.csv \
              --train_dataset /path/to/trainset_manual.csv \
              --vector_store_type FAISS \
              --embedding_model hf_embeddings \
              --reranking True \
              --refine_query False \
              --method ensemble \
              --groq_api_key <token>
```

### 4. Running the Streamlit Application

To launch the Streamlit interface for the RAG application:

```bash
streamlit run src/rag_app.py
```

This will start a local Streamlit server for interacting with the AI assistant.

---

## Dataset and Models

- **Training Data**: Custom datasets from the Laser Genius+ machine manuals containing QA pairs.
- **Model**: `llama-3.1-70b-Versatile` hosted on Groq Cloud was used for both RAG and few-shot learning experiments.

## Evaluation Metrics

- **Cosine Similarity**: Measures semantic similarity between generated and reference answers.
- **ROUGE Scores**: Measures n-gram overlap between generated and reference

answers.

- **Recall & MRR**: Evaluate document retrieval quality and ranking.

## Future Work

This proof of concept lays the foundation for AI-assisted HMI interactions. Future enhancements may include:

- Fine-tuning LLaMA models on diverse datasets.
- Incorporating reinforcement learning from human feedback (RLHF).

## Contact

For questions or feedback, please contact:

- Hadi Ibrahim (Politecnico di Torino) - [s313385@studenti.polito.it](mailto:s313385@studenti.polito.it)
- Naser Ibrahim (University of Vaasa) - [x7823417@student.uwasa.fi](mailto:x7823417@studenti.uwasa.fi)

---

Feel free to adjust any details or add additional context as necessary!
