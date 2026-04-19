import os
import langchain
from dotenv import load_dotenv

load_dotenv()
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate

#option 1 to debug - may be noisey with all the logs from langchain and qdrant client, but can be helpful to see the flow of data and identify where things might be going wrong.
#langchain.debug = True


# Configuration
PROJECTID = "atroai"
REGION = "us-central1"
GEMINI_EMBED_MODEL = "gemini-embedding-001"
# COLLECTION_NAME = "atroai_pdf_chunks" #use this for text based embeddings.
COLLECTION_NAME = "pdf_rag_collection" #use pdf_rag_collection to use multimodal embeddings.
GEMINI_LLM_MODEL = "gemini-2.5-pro"
GEMINI_LLM_TEMPERATURE = 0.5
SIMILARITY_SEARCH_K = 5

# Get Qdrant connection details from environment or use defaults
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"


def ask_question(user_question):
    # 2. Re-connect to Local Qdrant
    embeddings = GoogleGenerativeAIEmbeddings(model=f"models/{GEMINI_EMBED_MODEL}")
    client = QdrantClient(url=QDRANT_URL)
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # 3. Initialize the Gemini LLM
    # We use ChatGoogleGenerativeAI for conversation generation, setting a temperature for factual accuracy
    llm = ChatGoogleGenerativeAI(model=GEMINI_LLM_MODEL, temperature=GEMINI_LLM_TEMPERATURE)

    # 4. Create the RAG Prompt Template (kept as a string for readability)
    # This strictly instructs the LLM to only use our PDF data
    # **Few shot prompting** examples are provided to guide the model on how to format the answer and what to do if the answer is not found in the retrieved context. 
        # You can modify these examples based on your specific use case and document content.
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

    # 5. Retrieve the top relevant chunks from Qdrant and choose the best match
    matches = vector_store.similarity_search_with_relevance_scores(user_question, k=SIMILARITY_SEARCH_K)
    if not matches:
        print("No matching documents found in Qdrant.")
        return "I cannot answer this."

    # Sort by similarity score descending and use the best match first
    matches.sort(key=lambda item: item[1], reverse=True)
    docs = [doc for doc, _ in matches]
    best_doc, best_score = matches[0]

    context = "\n\n".join(
        f"Chunk {i+1} (score: {score:.3f}):\n{doc.page_content}"
        for i, (doc, score) in enumerate(matches)
    )

    # 6. Generate an answer from Gemini
    print(f"Thinking about: '{user_question}'...\n")
    prompt_text = prompt.format(input=user_question, context=context)

    print(f"\n[DEBUG 1] USER QUESTION RECEIVED: '{user_question}'")
    print(f"[DEBUG 1] Converting question to a vector using '{GEMINI_EMBED_MODEL}'...")
    response = llm.invoke(prompt_text)

    # 7. Output the results
    print("🤖 --- Final Answer ---")
    if hasattr(response, "content"):
        answer = response.content
        print(answer)
    else:
        answer = str(response)
        print(answer)

    # --- DEBUGGING THE RETRIEVER ---
    print("\n==================================================")
    print("[DEBUG 2] 🔍 VECTOR DATABASE RETRIEVAL RESULTS")
    print(f"Retrieved {len(docs)} candidate chunks from Qdrant.")
    print(f"Best match score: {best_score:.3f}")
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

    # --- DEBUGGING THE GENERATION ---
    print("\n==================================================")
    print("[DEBUG 3] 🧠 GEMINI MODEL EVALUATION & ANSWER")
    print("==================================================")
    print("Gemini read the user question AND the raw text chunks above.")
    print("Here is the final generated answer based ONLY on those chunks:\n")
    print(answer)
    print("\n==================================================\n")

    return answer

if __name__ == "__main__":
    print("Welcome to the Astrology RAG Chatbot!")
    print("Ask questions about astrology based on the loaded document.\n")
    
    while True:
        question = input("Your question (or type 'quit' to exit): ")
        if question.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        ask_question(question)
