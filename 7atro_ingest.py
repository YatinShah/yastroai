import os
import glob
import fitz  # PyMuPDF
import langchain
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Ensure GEMINI_API_KEY is available and set GOOGLE_API_KEY for LangChain
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    sys.exit(1)
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY # LangChain often expects GOOGLE_API_KEY
 
# --- Application Configuration Constants ---
# Project and Region (Note: These are defined but not actively used in this script's logic)
PROJECTID = "atroai"
REGION = "us-central1"

# Qdrant Collection Name
COLLECTION_NAME = "pdf_rag_collection"

# Gemini Models
GEMINI_EMBED_MODEL = "gemini-embedding-001"         # Used for generating embeddings
IMAGE_DESCRIPTION_MODEL = "gemini-2.5-flash"        # Used for describing images during ingestion
GEMINI_LLM_MODEL = "gemini-2.5-pro"                 # Used for answering questions

# Ingestion Parameters
DOCUMENT_DIR="./data"  # Directory containing PDFs for batch processing
TEXT_CHUNK_SIZE = 500
TEXT_CHUNK_OVERLAP = 100
QDRANT_BATCH_SIZE = 12

# RAG (Retrieval Augmented Generation) Parameters
GEMINI_LLM_TEMPERATURE = 0.5 # Controls creativity/randomness of LLM responses (0.0 for deterministic)
SIMILARITY_SEARCH_K = 5      # Number of top relevant chunks to retrieve from Qdrant

# Qdrant Connection Details (can be overridden by environment variables)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"


def process_bulk_pdfs():
    """
    Processes all PDF documents in the DOCUMENT_DIR, extracts text and images,
    generates descriptions for images using Gemini, chunks the content,
    and ingests it into a Qdrant vector store.
    """
    print("\n--- Starting Bulk PDF Ingestion ---")

    # 1. Initialize Models & Database Clients
    client_genai = genai.Client(api_key=GEMINI_API_KEY)
    embeddings = GoogleGenerativeAIEmbeddings(model=f"models/{GEMINI_EMBED_MODEL}", google_api_key=GEMINI_API_KEY)
    
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)

    # Create Qdrant collection if it doesn't exist
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating Qdrant collection: '{COLLECTION_NAME}'")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE), # 3072 is the dimension for gemini-embedding-001
        )
    else:
        print(f"Qdrant collection '{COLLECTION_NAME}' already exists.")
        
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    
    # Initialize text splitter for breaking down documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE, chunk_overlap=TEXT_CHUNK_OVERLAP
    )

    # 2. Find all PDFs in the specified directory
    if not os.path.exists(DOCUMENT_DIR):
        print(f"Directory '{DOCUMENT_DIR}' not found. Please create it and add PDFs.")
        return

    pdf_files = glob.glob(os.path.join(DOCUMENT_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in {DOCUMENT_DIR}.")
        return

    print(f"Found {len(pdf_files)} PDFs. Starting ingestion...\n")

    # 3. Iterate through each PDF file for content extraction and description
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"--- Processing: {filename} ---")
        
        doc = fitz.open(pdf_path)
        raw_documents = [] # To store Document objects (extracted text and image descriptions) for the current PDF

        # Iterate through pages of the current document
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract Text from the page
            text = page.get_text()
            if text.strip():
                raw_documents.append(Document(
                    page_content=f"[File: {filename} | Page {page_num + 1} Text]\n{text}",
                    metadata={"source": filename, "page": page_num + 1, "type": "text"}
                ))

            # Extract & Describe Images from the page
            images = page.get_images(full=True)
            for img_index, img_info in enumerate(images):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]
                    
                    image_part = types.Part.from_bytes(data=image_bytes, mime_type=f"image/{ext}")
                    # Prompt for Gemini to describe the image
                    prompt = (
                        "Describe this image in high detail. "
                        "If it is a chart or graph, extract the specific data points and trends. "
                        "If it is a diagram, explain the flow and components."
                    )
                    
                    # Generate image description using Gemini's multimodal capabilities
                    response = client_genai.models.generate_content(
                        model=IMAGE_DESCRIPTION_MODEL,
                        contents=[image_part, prompt]
                    )
                    
                    raw_documents.append(Document(
                        page_content=f"[File: {filename} | Page {page_num + 1} Image]\n{response.text}",
                        metadata={"source": filename, "page": page_num + 1, "type": "image_summary"}
                    ))
                except Exception as e:
                    print(f"  ⚠️ Failed to process an image on page {page_num + 1} in {filename}: {e}")

        # 4. Chunk the collected raw documents for the current PDF and upload to Qdrant
        if raw_documents:
            chunks = text_splitter.split_documents(raw_documents)
            vector_store.add_documents(chunks, batch_size=QDRANT_BATCH_SIZE)
            print(f"✅ Upserted {len(chunks)} chunks to Qdrant for {filename}\n")
        else:
            print(f"⚠️ No readable content (text or images) found in {filename}\n")

    print("🎉 Bulk multimodal ingestion complete! All documents are in Qdrant.")


def ask_question(user_question):
    """
    Retrieves relevant documents from Qdrant based on the user's question,
    constructs a prompt with the retrieved context, and generates an answer
    using a Gemini LLM.
    """
    print(f"\n--- Answering Question: '{user_question}' ---")

    # 1. Initialize Embeddings and Qdrant Client
    embeddings = GoogleGenerativeAIEmbeddings(model=f"models/{GEMINI_EMBED_MODEL}", google_api_key=GEMINI_API_KEY)
    client = QdrantClient(url=QDRANT_URL)
    
    # Ensure the collection exists before trying to query it
    if not client.collection_exists(COLLECTION_NAME):
        print(f"❌ Qdrant collection '{COLLECTION_NAME}' does not exist. Please run ingestion first.")
        return "I cannot answer this question as the document collection is not set up."

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # 2. Initialize the Gemini LLM for generating responses
    llm = ChatGoogleGenerativeAI(model=GEMINI_LLM_MODEL, temperature=GEMINI_LLM_TEMPERATURE, google_api_key=GEMINI_API_KEY)

    # 3. Create the RAG Prompt Template
    # This prompt strictly instructs the LLM to only use the provided context.
    # Few-shot prompting examples guide the model on desired output formatting and how to handle unknown answers.
    system_prompt = (
        "You are an expert AI astrologer. Use the following pieces of retrieved context "
        "to answer the user's question. If the answer is not in the context, say 'I cannot answer this.'\n\n"
        "### FORMATTING INSTRUCTIONS & EXAMPLES ###\n"
        "Always format your response exactly like the examples below.\n\n"
        "Example 1:\n"
        "User: What is my birth planet, if my birthdate is 01/01/1972?\n"
        "Output: **Birth date:** 01/01/1972 | **Planet:** Mercury | **Source:** You were born on 01/01/1972, based on birthchart, you were born in planet Mercury.\n\n"
        "Example 2:\n"
        "User: List birthstone based on my birthdate.\n"
        "Output: **Key birthstone:**\n- Emerald\n- Shepphier\n\n"
        "### RETRIEVED CONTEXT ###\n"
        "{context}"
    )
    prompt = PromptTemplate.from_template(system_prompt + "\n\nQuestion: {input}")

    # 4. Retrieve the top relevant chunks from Qdrant based on the user's question
    matches = vector_store.similarity_search_with_relevance_scores(user_question, k=SIMILARITY_SEARCH_K)
    if not matches:
        print("No matching documents found in Qdrant.")
        return "I cannot answer this."

    # Sort retrieved documents by similarity score (highest first) and prepare context string
    matches.sort(key=lambda item: item[1], reverse=True)
    
    context = "\n\n".join(
        f"Chunk {i+1} (score: {score:.3f}):\n{doc.page_content}"
        for i, (doc, score) in enumerate(matches)
    )

    # 5. Generate an answer from Gemini using the constructed prompt and context
    print(f"Preparing prompt for Gemini with retrieved context...")
    prompt_text = prompt.format(input=user_question, context=context)

    response = llm.invoke(prompt_text)

    # 6. Extract and return the answer from Gemini's response
    answer = response.content if hasattr(response, "content") else str(response)
    
    # --- DEBUGGING AND SOURCE INFORMATION ---
    print("\n==================================================")
    print("[DEBUG] 🔍 VECTOR DATABASE RETRIEVAL RESULTS")
    print(f"Retrieved {len(matches)} candidate chunks from Qdrant.")
    print(f"Best match score: {matches[0][1]:.3f}")
    print("==================================================")

    print("\n📚 --- Source Documents Used ---")
    for i, (doc, score) in enumerate(matches):
        metadata = doc.metadata
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page', 'N/A')
        doc_type = metadata.get('type', 'text')
        print(f"Source {i+1}:")
        print(f"  Document: {source}")
        print(f"  Page: {page}")
        print(f"  Type: {doc_type}")
        print(f"  Similarity: {score:.3f}")
        print(f"  Content: {doc.page_content[:400]}...")
        print()

    print("\n==================================================")
    print("[DEBUG] 🧠 GEMINI MODEL EVALUATION & ANSWER")
    print("==================================================")
    print("Gemini read the user question AND the raw text chunks above.")
    print("Here is the final generated answer based ONLY on those chunks:\n")
    print(answer)
    print("\n==================================================\n")

    return answer


if __name__ == "__main__":
    print("Welcome to the Combined Ingestion and RAG Script!")
    print("This script allows you to ingest documents or ask questions.")

    while True:
        print("\n--- Main Menu ---")
        print("1. Ingest documents (process_bulk_pdfs)")
        print("2. Ask a question (ask_question)")
        print("3. Exit")
        
        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == '1':
            process_bulk_pdfs()
        elif choice == '2':
            question = input("Enter your question: ")
            if question.strip():
                ask_question(question)
            else:
                print("Question cannot be empty.")
        elif choice == '3':
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
