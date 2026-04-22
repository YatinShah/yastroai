# file combines 31* and 4* python scripts into one.
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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

class AstroConfig:
    """Holds configuration constants and environment variables for the application."""
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Ensure GEMINI_API_KEY is available and set GOOGLE_API_KEY for LangChain
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            print("❌ GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
            sys.exit(1)
        os.environ["GOOGLE_API_KEY"] = self.gemini_api_key
        
        # --- Application Configuration Constants ---
        self.project_id = "atroai"
        self.region = "us-central1"
        self.collection_name = "pdf_rag_collection"

        # Gemini Models
        # gemini-embedding-001: Max input limit is 2,048 tokens per chunk
        self.gemini_embed_model = "gemini-embedding-001"
        # gemini-2.5-flash: Max input is 1,000,000 tokens. Max output is 8,192 tokens.
        self.image_description_model = "gemini-2.5-flash"
        # gemini-2.5-pro: Max input is 2,000,000 tokens. Max output is 8,192 tokens.
        self.gemini_llm_model = "gemini-2.5-pro"

        # Ollama Models
        self.ollama_model = "llama3.2:1b"
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://192.168.68.10:9090")
        self.ollama_embed_model = "nomic-embed-text:v1.5"
        self.embed_provider = os.getenv("EMBED_PROVIDER", "fastembed")
        
        # Isolate collections by provider to prevent dimension mismatches
        # Default to pdf_rag_collection for fastembed to preserve existing data
        if self.embed_provider.lower() == "fastembed":
            self.collection_name = "pdf_rag_collection"
        else:
            self.collection_name = f"pdf_rag_collection_{self.embed_provider.lower()}"

        # Ingestion Parameters
        self.document_dir = "./data"
        self.text_chunk_size = 1500
        self.text_chunk_overlap = 150
        self.qdrant_batch_size = 12
        
        env_dim = os.getenv("VECTOR_DIMENSION")
        if env_dim:
            self.vector_dimension = int(env_dim)
        else:
            if self.embed_provider.lower() == "google":
                self.vector_dimension = 3072
            elif self.embed_provider.lower() == "ollama":
                if "minilm" in self.ollama_embed_model.lower():
                    self.vector_dimension = 384
                elif "mxbai" in self.ollama_embed_model.lower():
                    self.vector_dimension = 1024
                else:
                    self.vector_dimension = 768 # default for nomic-embed-text
            else:
                self.vector_dimension = 384 # fastembed default

        # RAG (Retrieval Augmented Generation) Parameters
        self.gemini_llm_temperature = 0.5
        self.similarity_search_k = 5
        self.gemini_llm_max_output_tokens = 8192
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"

        # Qdrant Connection Details
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        self.qdrant_url = f"http://{self.qdrant_host}:{self.qdrant_port}"
        self.qdrant_path = os.getenv("QDRANT_PATH", "./qdrant_storage")
        self.use_qdrant_server = os.getenv("USE_QDRANT_SERVER", "true").lower() == "true"
        self._qdrant_client = None

    def get_qdrant_client(self):
        """Returns a QdrantClient configured for either a server or local disk."""
        if self._qdrant_client is None:
            if self.use_qdrant_server:
                self._qdrant_client = QdrantClient(url=self.qdrant_url)
            else:
                self._qdrant_client = QdrantClient(path=self.qdrant_path)
        return self._qdrant_client

    def get_embeddings(self, provider: str = None):
        """Factory method to get the specified Embeddings."""
        provider = provider or self.embed_provider
        if provider.lower() == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model=self.gemini_embed_model,
                google_api_key=self.gemini_api_key
            )
        elif provider.lower() == "ollama":
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(
                model=self.ollama_embed_model,
                base_url=self.ollama_base_url
            )
        elif provider.lower() == "fastembed":
            from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
            return FastEmbedEmbeddings()
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

class DocumentIngestor:
    """Handles the ingestion of PDF documents into the vector store."""
    def __init__(self, config: AstroConfig):
        self.config = config
        self.client_genai = genai.Client(api_key=self.config.gemini_api_key)
        self.embeddings = self.config.get_embeddings()
        self.qdrant_client = self.config.get_qdrant_client()

    def setup_collection(self):
        """Deletes the Qdrant collection if it exists, then creates a new one."""
        if self.qdrant_client.collection_exists(self.config.collection_name):
            print(f"Deleting existing Qdrant collection: '{self.config.collection_name}'")
            self.qdrant_client.delete_collection(self.config.collection_name)
            
        print(f"Creating new Qdrant collection: '{self.config.collection_name}'")
        self.qdrant_client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(size=self.config.vector_dimension, distance=Distance.COSINE),
        )

    def process_bulk_pdfs(self):
        """Processes all PDF documents in the DOCUMENT_DIR."""
        print("\n--- Starting Bulk PDF Ingestion ---")
        location = self.config.qdrant_url if self.config.use_qdrant_server else self.config.qdrant_path
        print(f"Connecting to Qdrant at {location}...")
        
        self.setup_collection()
        
        vector_store = QdrantVectorStore(
            client=self.qdrant_client, 
            collection_name=self.config.collection_name, 
            embedding=self.embeddings
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.text_chunk_size, 
            chunk_overlap=self.config.text_chunk_overlap
        )

        if not os.path.exists(self.config.document_dir):
            print(f"Directory '{self.config.document_dir}' not found. Please create it and add PDFs.")
            return

        pdf_files = glob.glob(os.path.join(self.config.document_dir, "*.pdf"))
        
        if not pdf_files:
            print(f"No PDFs found in {self.config.document_dir}.")
            return

        print(f"Found {len(pdf_files)} PDFs. Starting ingestion...\n")

        for pdf_path in pdf_files:
            self._process_single_pdf(pdf_path, text_splitter, vector_store)

        print("🎉 Bulk multimodal ingestion complete! All documents are in Qdrant.")

    def _process_single_pdf(self, pdf_path: str, text_splitter: RecursiveCharacterTextSplitter, vector_store: QdrantVectorStore):
        """Extracts text and images from a single PDF and uploads to Qdrant."""
        filename = os.path.basename(pdf_path)
        print(f"--- Processing: {filename} ---")
        print(f"[DEBUG] Opening PDF: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        raw_documents = []
        print(f"[DEBUG] Total pages in {filename}: {len(doc)}")

        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"  [DEBUG] Processing page {page_num + 1}/{len(doc)}")
            
            # Extract Text
            text = page.get_text()
            if text.strip():
                print(f"    [DEBUG] Extracted {len(text)} characters of text from page {page_num + 1}")
                raw_documents.append(Document(
                    page_content=f"[File: {filename} | Page {page_num + 1} Text]\n{text}",
                    metadata={"source": filename, "page": page_num + 1, "type": "text"}
                ))
            else:
                print(f"    [DEBUG] No text found on page {page_num + 1}")

            # Extract & Describe Images
            images = page.get_images(full=True)
            print(f"    [DEBUG] Found {len(images)} images on page {page_num + 1}")
            for img_index, img_info in enumerate(images):
                print(f"    [DEBUG] Processing image {img_index + 1}/{len(images)} on page {page_num + 1}")
                self._process_image(doc, img_info, filename, page_num, raw_documents, img_index)

        if raw_documents:
            print(f"[DEBUG] Splitting {len(raw_documents)} raw documents into chunks...")
            chunks = text_splitter.split_documents(raw_documents)
            print(f"[DEBUG] Created {len(chunks)} chunks. Uploading to Qdrant in batches of {self.config.qdrant_batch_size}...")
            vector_store.add_documents(chunks, batch_size=self.config.qdrant_batch_size)
            print(f"✅ Upserted {len(chunks)} chunks to Qdrant for {filename}\n")
        else:
            print(f"⚠️ No readable content (text or images) found in {filename}\n")

    def _process_image(self, doc, img_info, filename: str, page_num: int, raw_documents: list, img_index: int):
        """Extracts and describes a single image."""
        try:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            print(f"      [DEBUG] Extracted image with xref {xref}, format {ext}, size {len(image_bytes)} bytes")
            
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=f"image/{ext}")
            prompt = (
                "Describe this image in high detail. "
                "If it is a chart or graph, extract the specific data points and trends. "
                "If it is a diagram, explain the flow and components."
            )
            
            print(f"      [DEBUG] Sending image to Gemini {self.config.image_description_model} for description...")
            response = self.client_genai.models.generate_content(
                model=self.config.image_description_model,
                contents=[image_part, prompt]
            )
            print(f"      [DEBUG] Received description from Gemini : {response.text}")
            
            # Save the image to disk
            image_filename = f"{filename}_page{page_num+1}_img{img_index+1}.{ext}"
            image_filepath = os.path.join("data", "extracted_images", image_filename)
            with open(image_filepath, "wb") as img_file:
                img_file.write(image_bytes)
            print(f"      [DEBUG] Saved image to {image_filepath}")
            
            raw_documents.append(Document(
                page_content=f"[File: {filename} | Page {page_num + 1} Image {img_index + 1}]\n{response.text}",
                metadata={"source": filename, "page": page_num + 1, "type": "image_summary", "image_path": image_filepath}
            ))
        except Exception as e:
            print(f"      [DEBUG] ⚠️ Failed to process an image on page {page_num + 1} in {filename}: {e}")

class RAGQueryEngine:
    """Handles retrieving context and answering questions."""
    def __init__(self, config: AstroConfig):
        self.config = config
        self.embeddings = self.config.get_embeddings()
        self.qdrant_client = self.config.get_qdrant_client()

        self.system_prompt = (
            "You are an expert AI astrologer. Use the following pieces of retrieved context "
            "to answer the user's question. If the answer is not in the context, say 'I cannot answer this.'\n\n"
            "### FORMATTING INSTRUCTIONS & EXAMPLES ###\n"
            "Always format your response exactly like the examples below.\n\n"
            "Example 1:\n"
            "User: What is my birth planet, if my birthdate is 01/01/1906?\n"
            "Output: **Birth date:** 01/01/1906 | **Planet:** Venus | **Planet Symbol:** ♉ | **Sign:** Pisces | **Zodiac Symbol:** ♓ | **Chakra:** Throat | **Element:** Water | **Color:** Light Blue / Aqua | **Gemstone:** Moonstone | **Season:** Spring | **Motto:** I dream | **Element Modality:** Mutable | **Planet Modality:** Venus | **Element Triplicity:** Water | **Planet Triplicity:** Venus | **Planet Quadruplicity:** Water | **Planet Quadruplicity:** Venus | **Planet Quadruplicity:** Water | **Planet Quadruplicity:** Venus\n\n"
            "Example 2:\n"
            "User: List birthstone based on my birthdate.\n"
            "Output: **Key birthstone:**\n- Emerald\n- Shepphier\n\n"
            "### RETRIEVED CONTEXT ###\n"
            "{context}"
        )

    def _get_llm(self, provider: str):
        """Factory method to get the specified LLM."""
        if provider.lower() == "google":
            return ChatGoogleGenerativeAI(
                model=self.config.gemini_llm_model, 
                temperature=self.config.gemini_llm_temperature, 
                max_output_tokens=self.config.gemini_llm_max_output_tokens,
                google_api_key=self.config.gemini_api_key
            )
        elif provider.lower() == "ollama":
            return ChatOllama(
                model=self.config.ollama_model,
                base_url=self.config.ollama_base_url,
                temperature=self.config.gemini_llm_temperature
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def ask_question(self, user_question: str, provider: str = "google") -> tuple[str, list]:
        """Retrieves documents from Qdrant and generates an answer."""
        provider_name = "Gemini" if provider.lower() == "google" else "Ollama"
        if(self.config.debug_mode):
            print(f"\n--- Answering Question using {provider_name}: '{user_question}' ---")

        if not self.qdrant_client.collection_exists(self.config.collection_name):
            print(f"❌ Qdrant collection '{self.config.collection_name}' does not exist. Please run ingestion first.")
            return "I cannot answer this question as the document collection is not set up.", []

        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.config.collection_name,
            embedding=self.embeddings,
        )

        prompt = PromptTemplate.from_template(self.system_prompt + "\n\nQuestion: {input}")

        matches = vector_store.similarity_search_with_relevance_scores(user_question, k=self.config.similarity_search_k)
        if not matches:
            print("No matching documents found in Qdrant.")
            return "I cannot answer this.", []

        matches.sort(key=lambda item: item[1], reverse=True)
        
        context = "\n\n".join(
            f"Chunk {i+1} (score: {score:.3f}):\n{doc.page_content}"
            for i, (doc, score) in enumerate(matches)
        )

        image_paths = []
        for doc, score in matches:
            if doc.metadata.get("type") == "image_summary" and "image_path" in doc.metadata:
                image_paths.append(doc.metadata["image_path"])
                
        # Deduplicate while preserving order
        image_paths = list(dict.fromkeys(image_paths))

        if self.config.debug_mode:
            print(f"Preparing prompt for {provider_name} with retrieved context...")
        prompt_text = prompt.format(input=user_question, context=context)

        llm = self._get_llm(provider)
        response = llm.invoke(prompt_text)
        answer = response.content if hasattr(response, "content") else str(response)
        
        self._print_debug_info(user_question, matches, answer, provider_name)

        return answer, image_paths

    def _print_debug_info(self, user_question, matches, answer, provider_name: str):
        """Prints debugging and source information."""
        if not self.config.debug_mode:
            print(f"\n--- Question: {user_question}\n")
            print(f"--- Answer: {answer}\n")
            return

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
        print(f"[DEBUG] 🧠 {provider_name.upper()} MODEL EVALUATION & ANSWER")
        print("==================================================")
        print(f"{provider_name} read the user question AND the raw text chunks above.")
        print("Here is the final generated answer based ONLY on those chunks:\n")
        print(f"Question: {user_question}\n")
        print(f"Answer: {answer}\n")
        print("\n==================================================\n")

class AstroRAGApplication:
    """Main application orchestrator."""
    def __init__(self):
        self.config = AstroConfig()
        self.ingestor = DocumentIngestor(self.config)
        self.query_engine = RAGQueryEngine(self.config)

    def _read_input(self, input_stream):
        line = input_stream.readline()
        if not line:
            return None
        stripped_line = line.strip()
        lines = []
        while stripped_line.endswith('|'):
            lines.append(stripped_line[:-1].strip())
            next_line = input_stream.readline()
            if not next_line:
                break
            stripped_line = next_line.strip()
        lines.append(stripped_line)
        return "\n".join(lines).strip()

    def run(self, input_stream=sys.stdin, output_stream=sys.stdout):
        is_stream = not input_stream.isatty()
        original_stdout = sys.stdout
        sys.stdout = output_stream
        
        try:
            if not is_stream:
                print("Welcome to the Combined Ingestion and RAG Script!")
                print("This script allows you to ingest documents or ask questions.")

            while True:
                if not is_stream:
                    print("\n--- Main Menu ---")
                    print("1. Ingest documents (process_bulk_pdfs)")
                    print("2. Ask a question (Gemini)")
                    print("3. Ask a question (Local Ollama - llama3.2:1b)")
                    print("4. Exit")
                    print("Enter your choice (1, 2, 3, or 4) or type a question to default to Option 3: ", end="")
                    sys.stdout.flush()
                
                choice = self._read_input(input_stream)
                if choice is None:
                    break
                    
                if not choice:
                    continue
                
                # Check if the input is a direct question (doesn't start with a choice number)
                question = None
                if choice not in ['1', '2', '3', '4']:
                    question = choice
                    choice = '3'
                    if not is_stream:
                        print(f"\n--- [Stream Input] Defaulting to Ollama for question: '{question[:50]}...' ---")

                if choice == '1':
                    print("Ingestion temporarily disabled.")
                    # self.ingestor.process_bulk_pdfs()
                elif choice == '2':
                    if not question:
                        print("Enter your question: ", end="")
                        sys.stdout.flush()
                        question = self._read_input(input_stream)
                    if question:
                        answer, image_paths = self.query_engine.ask_question(question, provider="google")
                        if image_paths:
                            print(f"++Also retrieved {len(image_paths)} related images: {image_paths}")
                    else:
                        print("Question cannot be empty.")
                elif choice == '3':
                    if not question:
                        print("Enter your question: ", end="")
                        sys.stdout.flush()
                        question = self._read_input(input_stream)
                    if question:
                        answer, image_paths = self.query_engine.ask_question(question, provider="ollama")
                        if image_paths:
                            print(f"Also retrieved {len(image_paths)} related images: {image_paths}")
                    else:
                        print("Question cannot be empty.")
                elif choice == '4':
                    print("Exiting. Goodbye!")
                    if self.config._qdrant_client:
                        self.config._qdrant_client.close()
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
        finally:
            sys.stdout = original_stdout

if __name__ == "__main__":
    app = AstroRAGApplication()
    app.run()
