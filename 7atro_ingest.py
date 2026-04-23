import os
import glob
import fitz  # PyMuPDF
import langchain
import sys
from abc import ABC, abstractmethod
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

class AppConstants:
    """Centralized constants for the application."""
    PROJECT_ID = "atroai"
    REGION = "us-central1"
    
    # Gemini Models
    GEMINI_EMBED_MODEL = "gemini-embedding-001"
    IMAGE_DESCRIPTION_MODEL = "gemini-2.5-flash"
    GEMINI_LLM_MODEL = "gemini-2.5-pro"
    
    # Ollama Models
    OLLAMA_MODEL = "llama3.2:1b"
    OLLAMA_EMBED_MODEL = "nomic-embed-text:v1.5"
    
    # RAG Parameters
    DEFAULT_CHUNK_SIZE = 1500
    DEFAULT_CHUNK_OVERLAP = 150
    DEFAULT_SIMILARITY_K = 5
    DEFAULT_LLM_TEMP = 0.5
    DEFAULT_MAX_TOKENS = 8192
    
    # Qdrant
    DEFAULT_COLLECTION_NAME = "pdf_rag_collection"
    BATCH_SIZE = 12

    SYSTEM_PROMPT = (
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

class AstroConfig:
    """Holds configuration constants and environment variables for the application."""
    def __init__(self):
        load_dotenv()
        self._init_api_keys()
        self._init_models()
        self._init_rag_params()
        self._init_qdrant_params()
        self._qdrant_client = None

    def _init_api_keys(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            print("❌ GEMINI_API_KEY not found. Please set it in your .env file.")
            sys.exit(1)
        os.environ["GOOGLE_API_KEY"] = self.gemini_api_key

    def _init_models(self):
        self.embed_provider = os.getenv("EMBED_PROVIDER", "fastembed")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://192.168.68.10:9090")
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        
        # Collection isolation
        if self.embed_provider.lower() == "fastembed":
            self.collection_name = AppConstants.DEFAULT_COLLECTION_NAME
        else:
            self.collection_name = f"{AppConstants.DEFAULT_COLLECTION_NAME}_{self.embed_provider.lower()}"

    def _init_rag_params(self):
        self.chunk_size = AppConstants.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = AppConstants.DEFAULT_CHUNK_OVERLAP
        self.similarity_k = AppConstants.DEFAULT_SIMILARITY_K
        self.llm_temp = AppConstants.DEFAULT_LLM_TEMP
        self.max_tokens = AppConstants.DEFAULT_MAX_TOKENS
        self.vector_dimension = self._infer_dimension()

    def _init_qdrant_params(self):
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        self.qdrant_url = f"http://{self.qdrant_host}:{self.qdrant_port}"
        self.qdrant_path = os.getenv("QDRANT_PATH", "./qdrant_storage")
        self.use_qdrant_server = os.getenv("USE_QDRANT_SERVER", "true").lower() == "true"

    def _infer_dimension(self) -> int:
        env_dim = os.getenv("VECTOR_DIMENSION")
        if env_dim: return int(env_dim)
        
        provider = self.embed_provider.lower()
        if provider == "google": return 3072
        if provider == "ollama":
            model = AppConstants.OLLAMA_EMBED_MODEL.lower()
            if "minilm" in model: return 384
            if "mxbai" in model: return 1024
            return 768
        return 384 # fastembed default

    def get_qdrant_client(self):
        if self._qdrant_client is None:
            if self.use_qdrant_server:
                self._qdrant_client = QdrantClient(url=self.qdrant_url)
            else:
                self._qdrant_client = QdrantClient(path=self.qdrant_path)
        return self._qdrant_client

    def get_embeddings(self, provider: str = "ollama"):
        provider = provider or self.embed_provider
        p_lower = provider.lower()
        if p_lower == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(model=AppConstants.GEMINI_EMBED_MODEL)
        elif p_lower == "ollama":
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(model=AppConstants.OLLAMA_EMBED_MODEL, base_url=self.ollama_base_url)
        elif p_lower == "fastembed":
            return FastEmbedEmbeddings()
        raise ValueError(f"Unsupported provider: {provider}")

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

    def process_bulk_pdfs(self, document_dir: str = "./data"):
        """Processes all PDF documents in the specified directory."""
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
            chunk_size=self.config.chunk_size, 
            chunk_overlap=self.config.chunk_overlap
        )

        if not os.path.exists(document_dir):
            print(f"Directory '{document_dir}' not found.")
            return

        pdf_files = glob.glob(os.path.join(document_dir, "*.pdf"))
        if not pdf_files:
            print(f"No PDFs found in {document_dir}.")
            return

        print(f"Found {len(pdf_files)} PDFs. Starting ingestion...\n")
        for pdf_path in pdf_files:
            self._process_single_pdf(pdf_path, text_splitter, vector_store)

    def _process_single_pdf(self, pdf_path: str, text_splitter: RecursiveCharacterTextSplitter, vector_store: QdrantVectorStore):
        """Extracts text and images from a single PDF and uploads to Qdrant."""
        filename = os.path.basename(pdf_path)
        print(f"--- Processing: {filename} ---")
        
        doc = fitz.open(pdf_path)
        raw_documents = []
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
                self._process_image(doc, img_info, filename, page_num, raw_documents, img_index)

        if raw_documents:
            chunks = text_splitter.split_documents(raw_documents)
            vector_store.add_documents(chunks, batch_size=AppConstants.BATCH_SIZE)
            print(f"✅ Upserted {len(chunks)} chunks to Qdrant for {filename}\n")

    def _process_image(self, doc, img_info, filename: str, page_num: int, raw_documents: list, img_index: int):
        """Extracts and describes a single image."""
        try:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=f"image/{ext}")
            prompt = (
                "Describe this image in high detail. "
                "If it is a chart or graph, extract the specific data points and trends. "
                "If it is a diagram, explain the flow and components."
            )
            
            response = self.client_genai.models.generate_content(
                model=AppConstants.IMAGE_DESCRIPTION_MODEL,
                contents=[image_part, prompt]
            )
            
            # Save the image to disk
            image_filename = f"{filename}_page{page_num+1}_img{img_index+1}.{ext}"
            image_filepath = os.path.join("data", "extracted_images", image_filename)
            os.makedirs(os.path.dirname(image_filepath), exist_ok=True)
            with open(image_filepath, "wb") as img_file:
                img_file.write(image_bytes)
            
            raw_documents.append(Document(
                page_content=f"[File: {filename} | Page {page_num + 1} Image {img_index + 1}]\n{response.text}",
                metadata={"source": filename, "page": page_num + 1, "type": "image_summary", "image_path": image_filepath}
            ))
        except Exception as e:
            if self.config.debug_mode:
                print(f"⚠️ Failed to process image on page {page_num + 1}: {e}")

class RAGEngineInterface(ABC):
    """Abstract base class for RAG implementations."""
    @abstractmethod
    def ask_question(self, user_question: str, provider: str = "ollama") -> tuple[str, list]:
        """Retrieves context and generates an answer."""
        pass

class AstroRAGEngine(RAGEngineInterface):
    """Handles retrieving context and answering questions."""
    def __init__(self, config: AstroConfig):
        self.config = config
        self.embeddings = self.config.get_embeddings()
        self.qdrant_client = self.config.get_qdrant_client()

    def _get_llm(self, provider: str="ollama"):
        """Factory method to get the specified LLM."""
        if provider.lower() == "google":
            return ChatGoogleGenerativeAI(
                model=AppConstants.GEMINI_LLM_MODEL, 
                temperature=self.config.llm_temp, 
                max_output_tokens=self.config.max_tokens
            )
        elif provider.lower() == "ollama":
            return ChatOllama(
                model=AppConstants.OLLAMA_MODEL,
                base_url=self.config.ollama_base_url,
                temperature=self.config.llm_temp
            )
        raise ValueError(f"Unsupported provider: {provider}")

    def ask_question(self, user_question: str, provider: str = "ollama") -> tuple[str, list]:
        """Retrieves documents from Qdrant and generates an answer."""
        provider_name = "Gemini" if provider.lower() == "google" else "Ollama"
        if self.config.debug_mode:
            print(f"\n--- Answering Question using {provider_name}: '{user_question}' ---")

        if not self.qdrant_client.collection_exists(self.config.collection_name):
            return "I cannot answer this question as the document collection is not set up.", []

        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.config.collection_name,
            embedding=self.embeddings,
        )

        prompt = PromptTemplate.from_template(AppConstants.SYSTEM_PROMPT + "\n\nQuestion: {input}")
        matches = vector_store.similarity_search_with_relevance_scores(user_question, k=self.config.similarity_k)
        
        if not matches:
            return "I cannot answer this.", []

        matches.sort(key=lambda item: item[1], reverse=True)
        context = "\n\n".join(f"Chunk {i+1} (score: {score:.3f}):\n{doc.page_content}" for i, (doc, score) in enumerate(matches))

        image_paths = list(dict.fromkeys([
            doc.metadata["image_path"] for doc, score in matches 
            if doc.metadata.get("type") == "image_summary" and "image_path" in doc.metadata
        ]))

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
            m = doc.metadata
            print(f"Source {i+1}: {m.get('source')} (Page {m.get('page')}) | Type: {m.get('type')} | Score: {score:.3f}")
            print(f"  Content: {doc.page_content[:200]}...\n")

        print("\n==================================================")
        print(f"[DEBUG] 🧠 {provider_name.upper()} MODEL EVALUATION & ANSWER")
        print("==================================================")
        print(f"Question: {user_question}\n")
        print(f"Answer: {answer}\n")
        print("==================================================\n")

class AstroRAGApplication:
    """Main application orchestrator."""
    def __init__(self):
        self.config = AstroConfig()
        self.ingestor = DocumentIngestor(self.config)
        self.query_engine = AstroRAGEngine(self.config)

    def _read_input(self, input_stream):
        line = input_stream.readline()
        if not line: return None
        stripped_line = line.strip()
        lines = []
        while stripped_line.endswith('|'):
            lines.append(stripped_line[:-1].strip())
            next_line = input_stream.readline()
            if not next_line: break
            stripped_line = next_line.strip()
        lines.append(stripped_line)
        return "\n".join(lines).strip()

    def run(self, input_stream=sys.stdin, output_stream=sys.stdout):
        """High-level application loop."""
        is_stream = not input_stream.isatty()
        original_stdout = sys.stdout
        sys.stdout = output_stream
        
        try:
            if not is_stream:
                print("Welcome to the AstroAI RAG Application!")
            
            while True:
                choice = self._get_user_choice(input_stream, is_stream)
                if choice is None or choice == '4':
                    if not is_stream and choice == '4':
                        print("Exiting. Goodbye!")
                    break
                self._dispatch_choice(choice, input_stream, is_stream)
        finally:
            if self.config._qdrant_client:
                self.config._qdrant_client.close()
            sys.stdout = original_stdout

    def _get_user_choice(self, input_stream, is_stream):
        """Displays menu and gets user input."""
        if not is_stream:
            print("\n--- Main Menu ---")
            print("1. Ingest documents")
            print("2. Ask a question (Gemini)")
            print("3. Ask a question (Local Ollama)")
            print("4. Exit")
            print("Enter choice (1-4) or type a question: ", end="")
            sys.stdout.flush()
        return self._read_input(input_stream)

    def _dispatch_choice(self, choice, input_stream, is_stream):
        """Routes the user choice to the appropriate handler."""
        question = None
        if choice not in ['1', '2', '3', '4']:
            question = choice
            choice = '3'
            if not is_stream:
                print(f"\n--- [Stream Input] Defaulting to Ollama: '{question[:50]}...' ---")

        if choice == '1':
            self._handle_ingestion()
        elif choice in ['2', '3']:
            provider = "google" if choice == '2' else "ollama"
            self._handle_query(input_stream, provider, question)
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

    def _handle_ingestion(self):
        """Triggers document ingestion."""
        print("Ingestion temporarily disabled in refactored handler.")
        # self.ingestor.process_bulk_pdfs()

    def _handle_query(self, input_stream, provider, question=None):
        """Handles question/answer flow for a specific provider."""
        if not question:
            print(f"Enter your question: ", end="")
            sys.stdout.flush()
            question = self._read_input(input_stream)
            
        if question:
            answer, image_paths = self.query_engine.ask_question(question, provider=provider)
            if image_paths:
                prefix = "++" if provider == "google" else ""
                print(f"{prefix}Also retrieved {len(image_paths)} related images: {image_paths}")
        else:
            print("Question cannot be empty.")

if __name__ == "__main__":
    app = AstroRAGApplication()
    app.run()
