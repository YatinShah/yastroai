import os
import glob
import fitz  # PyMuPDF
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Configuration
PROJECTID = "atroai"
REGION = "us-central1"
PDF_PATH = "data/sample.pdf"
COLLECTION_NAME = "pdf_rag_collection"
GEMINI_EMBED_MODEL = "text-embedding-004"
DOCUMENT_DIR="./data"  # Directory containing PDFs for batch processing


def process_bulk_pdfs():
    # 1. Initialize Models & Database
    vertexai.init(project=PROJECTID, location=REGION)
    vision_model = GenerativeModel("gemini-2.5-pro")
    embeddings = VertexAIEmbeddings(model_name=GEMINI_EMBED_MODEL)
    
    client = QdrantClient(url="http://localhost:6333")
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        
    vector_store = QdrantVectorStore(
        client=client, collection_name=COLLECTION_NAME, embedding=embeddings
    )
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) #reduce the chunk size in case of input tokens are more than what Gemini can handle, especially after adding image descriptions. You can experiment with this size based on your documents and needs.

    # 2. Find all PDFs in the directory
    if not os.path.exists(DOCUMENT_DIR):
        print(f"Directory '{DOCUMENT_DIR}' not found. Please create it and add PDFs.")
        return

    pdf_files = glob.glob(os.path.join(DOCUMENT_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in {DOCUMENT_DIR}.")
        return

    print(f"Found {len(pdf_files)} PDFs. Starting bulk ingestion...\n")

    # 3. Iterate through each file
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"--- Processing: {filename} ---")
        
        doc = fitz.open(pdf_path)
        raw_documents = []

        # Iterate through pages of the current document
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract Text
            text = page.get_text()
            if text.strip():
                raw_documents.append(Document(
                    page_content=f"[File: {filename} | Page {page_num + 1} Text]\n{text}",
                    metadata={"source": filename, "page": page_num + 1, "type": "text"}
                ))

            # Extract & Describe Images
            images = page.get_images(full=True)
            for img_index, img_info in enumerate(images):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]
                    
                    image_part = Part.from_data(data=image_bytes, mime_type=f"image/{ext}")
                    prompt = (
                        "Describe this image in high detail. "
                        "If it is a chart or graph, extract the specific data points and trends. "
                        "If it is a diagram, explain the flow and components."
                    )
                    
                    response = vision_model.generate_content([image_part, prompt])
                    
                    raw_documents.append(Document(
                        page_content=f"[File: {filename} | Page {page_num + 1} Image]\n{response.text}",
                        metadata={"source": filename, "page": page_num + 1, "type": "image_summary"}
                    ))
                except Exception as e:
                    print(f"  ⚠️ Failed to process an image on page {page_num + 1}: {e}")

        # 4. Chunk and upload THIS specific document before moving to the next
        if raw_documents:
            chunks = text_splitter.split_documents(raw_documents)
            vector_store.add_documents(chunks, batch_size=12)
            print(f"✅ Upserted {len(chunks)} chunks to Qdrant for {filename}\n")
        else:
            print(f"⚠️ No readable content found in {filename}\n")

    print("🎉 Bulk multimodal ingestion complete! All documents are in Qdrant.")

if __name__ == "__main__":
    process_bulk_pdfs()