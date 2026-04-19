import os
import getpass
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas

# Load environment variables from .env file
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Google-Gemini API key: ")
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")

# Configuration
PROJECTID = "atroai"
REGION = "us-central1"
PDF_PATH = "data/sample.pdf"
GEMINI_EMBED_MODEL = "gemini-embedding-001"

def create_sample_pdf():
    """Create a sample PDF file for testing."""
    os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
    c = pdf_canvas.Canvas(PDF_PATH, pagesize=letter)
    c.drawString(100, 750, "Sample Document for Testing")
    c.drawString(100, 730, "This is a sample PDF file created for testing the ingestion pipeline.")
    c.drawString(100, 710, "It contains some basic text content.")
    c.save()
    print(f"✅ Sample PDF created at {PDF_PATH}")

def process_pdf():
    # Check if PDF file exists and create if needed
    if not os.path.exists(PDF_PATH):
        print(f"📁 PDF file not found at {PDF_PATH}. Creating sample...")
        create_sample_pdf()
    
    # Load the PDF document
    try:
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} pages from PDF")
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return

    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Generate embeddings for each chunk using Vertex AI
    try:
        # Use GoogleGenerativeAIEmbeddings with the API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY not found in environment variables.")
            print("📝 Please set GEMINI_API_KEY in the .env file or as an environment variable.")
            return
        
        embedding_model = GoogleGenerativeAIEmbeddings(model=f"models/{GEMINI_EMBED_MODEL}", google_api_key=api_key)

        # Test embedding generation on the first chunk
        print("Generating sample embedding...")
        sample_vector = embedding_model.embed_query(chunks[0].page_content)
        print(f"✅ Generated embedding vector with {len(sample_vector)} dimensions!")
        print(f"🔍 Sample of the embedding vector (first 5 elements): {sample_vector[:5]} ...")
        print(f"✅ Successfully processed {len(chunks)} chunks")
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")

if __name__ == "__main__":
    create_sample_pdf()
    process_pdf()