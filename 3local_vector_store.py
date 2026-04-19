import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

# Configuration
PROJECTID = "atroai"
REGION = "us-central1"
PDF_PATH = "data/readHoroscope.pdf"
GEMINI_EMBED_MODEL = "gemini-embedding-001"
COLLECTION_NAME = "atroai_pdf_chunks"

def build_local_database():
    api_key = os.getenv("GEMINI_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model=f"models/{GEMINI_EMBED_MODEL}", google_api_key=api_key)

    # 2. Connect to local Qdrant Docker container
    print("Connecting to local Qdrant container...")
    client = QdrantClient(url="http://localhost:6333")
    
    # Check if collection exists, if not, create it
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )

    # 3. Initialize the LangChain Qdrant wrapper
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # 4. Load and chunk the PDF
    print("Parsing and chunking PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # 5. Add documents to Local Qdrant
    print(f"Embedding and uploading {len(chunks)} chunks to Qdrant...")
    vector_store.add_documents(chunks)
    
    print("✅ Successfully upserted chunks into local Qdrant DB!")
    
    # 6. Test a quick similarity search
    query = "What is the main topic of this document?"
    results = vector_store.similarity_search(query, k=3)
    
    print("\n--- Test Search Results ---")
    for i, doc in enumerate(results):
        print(f"Result {i+1}:\n{doc.page_content}\n")
        print(results[i].metadata)  # Show metadata if needed
        print("-" * 50)
        print(results[i])  # Show result value

if __name__ == "__main__":
    build_local_database()