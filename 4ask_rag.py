import vertexai
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate

# Configuration
PROJECTID = "atroai"
REGION = "us-central1"
PDF_PATH = "data/readHoroscope.pdf"
GEMINI_EMBED_MODEL = "text-embedding-004"
COLLECTION_NAME = "atroai_pdf_chunks"


def ask_question(user_question):
    # 1. Initialize Vertex AI
    vertexai.init(project=PROJECTID, location=REGION)
    
    # 2. Re-connect to Local Qdrant
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    client = QdrantClient(url="http://localhost:6333")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    
    # Create a retriever that fetches the top 3 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 3. Initialize the Gemini LLM
    # We use ChatVertexAI for conversation generation, setting a low temperature for factual accuracy
    llm = ChatVertexAI(model_name="gemini-2.5-pro", temperature=0.1)

    # 4. Create the RAG Prompt Template
    # This strictly instructs the LLM to only use our PDF data
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

    # 5. Retrieve the top-k relevant chunks from Qdrant
    docs = retriever.invoke(user_question)
    context = "\n\n".join(
        f"Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)
    )

    # 6. Generate an answer from Gemini
    print(f"Thinking about: '{user_question}'...\n")
    prompt_text = prompt.format(input=user_question, context=context)
    response = llm.invoke(prompt_text)

    # 7. Output the results
    print("🤖 --- Final Answer ---")
    if hasattr(response, "content"):
        answer = response.content
        print(answer)
    else:
        answer = str(response)
        print(answer)

    print("\n📚 --- Source Documents Used ---")
    for i, doc in enumerate(docs):
        print(f"Source {i+1}: {doc.page_content[:400]}...")

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
