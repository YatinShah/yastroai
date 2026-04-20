# yAtroAi

A local RAG pipeline using Streamlit, LangChain, Vertex AI, and Qdrant. The project ingests documents, builds embeddings, stores them in a local Qdrant vector database, and answers questions from the ingested content.

## Architecture Overview

This project is built as a simple local retrieval-augmented generation (RAG) system.

1. Document ingestion
   - `31multimodal_ingest.py` reads PDFs from `./data`, extracts text, and describes images using Gemini.
   - Output is stored as semantic documents in a Qdrant vector database.

2. Embedding generation
   - The app uses Vertex AI embeddings via `langchain_google_vertexai.VertexAIEmbeddings`.
   - Embeddings are stored in Qdrant collections for fast similarity search.

3. Retrieval
   - `4ask_rag.py` loads the Qdrant vector store and retrieves top relevant chunks for a user query.
   - It builds a prompt with retrieved context and sends it to Gemini.

4. UI / serving
   - `5app.py` provides a Streamlit chat interface.
   - The app loads `4ask_rag.py` dynamically and displays conversation history.

5. Evaluation
   - `6evaluate_rag.py` performs basic evaluation by comparing the pipeline response to expected answers.

### Architecture diagram

```text
[PDF files / data folder] --> [31multimodal_ingest.py] --> [Qdrant vector store]
                                 |                         ^
                                 v                         |
                       [Vertex AI embeddings]              |
                                 |                         |
                                 v                         |
                          [Stored vectors] -------------->|
                                 |                         |
[User query] --> [4ask_rag.py / 5app.py] --> [Retrieve top vectors] --> [Gemini answer]
```

### Data flow

- Source documents are stored under `./data`
- Qdrant’s storage is persisted under `./qdrant_storage`
- Google service account credentials are read from `./credentials/vertex-ai-key.json`
- `GEMINI_API_KEY` is loaded from `.env`

## Project Files

- `1test_vertextconn.py`: Check Vertex AI/Gemini connection.
- `2ingest.py`: Create sample PDF and ingest content with embeddings.
- `31multimodal_ingest.py`: Bulk ingest PDFs, extract text and images, embed into Qdrant.
- `3local_vector_store.py`: Build a local Qdrant index from a PDF.
- `4ask_rag.py`: Query the Qdrant collection and answer user questions.
- `5app.py`: Streamlit interface for interactive question answering.
- `6evaluate_rag.py`: Evaluate model responses.
- `Dockerfile`: Docker image definition for the Streamlit app.
- `docker-compose.yml`: Compose config for Qdrant and the app.
- `requirements.txt`: Python dependencies.

## Prerequisites

- Linux machine
- Python 3.10+
- Docker & Docker Compose
- Google Cloud SDK (`gcloud`)
- Google Cloud project with Vertex AI enabled
- `./credentials/vertex-ai-key.json` service account credentials
- `.env` containing `GEMINI_API_KEY`

## Local Setup

1. Clone the repository:

```bash
git clone https://github.com/YatinShah/yastroai.git
cd yAtroAi
```

2. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create `.env` in the repository root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

5. Place your Google service account JSON at:

```text
./credentials/vertex-ai-key.json
```

## Google Cloud / Vertex AI Initialization

Install `gcloud` and configure it:

```bash
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg curl
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install google-cloud-cli
```

Then initialize:

```bash
gcloud init
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
```

Verify setup:

```bash
gcloud config list
gcloud ai models list --region=us-central1
```

## Running the Project

### Start Qdrant

```bash
docker compose up qdrant -d
```

### Ingest documents

1. Add your PDF files to `./data`.
2. Run:

```bash
python 31multimodal_ingest.py
```

This reads document text and images, creates embeddings, and stores vectors in Qdrant.

### Query the RAG pipeline

```bash
python 4ask_rag.py
```

### To ingest and run the commandline app 

```bash
python 7atro_ingest.py
```

### Start the Streamlit app

```bash
streamlit run 5app.py
```

Open `http://localhost:8501` in your browser.

## Docker Commands

### Start all services

```bash
docker compose up --build -d
```

### Stop everything

```bash
docker compose down
```

### Run only the Streamlit app

```bash
docker compose up --build app -d
```

## Notes

- convert the data/*.txt files to pdf files in same folder.
- create a .env file and add `GEMINI_API_KEY` from gcloud console
- `qdrant` runs on `localhost:6333`
- The app is available on `localhost:8501`
- Re-run ingestion after changing source documents
- The current pipeline uses 768-dimensional embeddings

## References

- The documents (data.*.txt/pdf) to train the model were downloaded from project gutenburge, related to astrology
- [Google AI model availability](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#united-states)
- [VertexAI Locations](https://docs.cloud.google.com/vertex-ai/docs/general/locations)
- Gemini embedding models: https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/
- Embedding model comparison: https://milvus.io/blog/choose-embedding-model-rag-2026.md ^^VGood^^

## TODO
- Use newer libraries, and remove deprecated libraries/objects from use (e.g. use GenAI instead of Vertex AI)
- Use dockerized embedding model, e.g. Qwen3-VL-2B
- Use 3072-dimensional embeddings instead of 768 currently in use
