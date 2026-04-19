import vertexai
import fitz  # PyMuPDF
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

def process_multimodal_pdf():
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

    # 2. Open the PDF
    print(f"Opening {PDF_PATH} for multimodal extraction...")
    doc = fitz.open(PDF_PATH)
    raw_documents = []

    # 3. Iterate through every page
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # --- A. Extract Standard Text ---
        text = page.get_text()
        if text.strip():
            raw_documents.append(Document(
                page_content=f"[Page {page_num + 1} Text]\n{text}",
                metadata={"source": PDF_PATH, "page": page_num + 1, "type": "text"}
            ))

        # --- B. Extract & Describe Images ---
        images = page.get_images(full=True)
        for img_index, img_info in enumerate(images):
            # Extract the raw image bytes
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            
            print(f"Found image on page {page_num + 1}. Asking Gemini to describe it...")
            
            # Format the image for Gemini
            image_part = Part.from_data(data=image_bytes, mime_type=f"image/{ext}")
            
            # Prompt Gemini to be highly descriptive
            prompt = (
                "Describe this image in high detail. "
                "If it is a chart or graph, extract the specific data points and trends. "
                "If it is a diagram, explain the flow and components."
            )
            
            try:
                response = vision_model.generate_content([image_part, prompt])
                description = response.text
                
                # Append the AI's description as a searchable document
                raw_documents.append(Document(
                    page_content=f"[Page {page_num + 1} Image Description]\n{description}",
                    metadata={"source": PDF_PATH, "page": page_num + 1, "type": "image_summary"}
                ))
            except Exception as e:
                print(f"Failed to process image on page {page_num + 1}: {e}")

    # 4. Chunk the combined text and image descriptions
    print(f"\nExtracted {len(raw_documents)} total elements. Chunking...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(raw_documents)

    # 5. Push everything to local Qdrant
    print(f"Uploading {len(chunks)} multimodal chunks to Qdrant...")
    vector_store.add_documents(chunks)
    
    print("✅ Multimodal ingestion complete!")

if __name__ == "__main__":
    process_multimodal_pdf()