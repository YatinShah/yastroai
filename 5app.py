import streamlit as st
import importlib.util

# Load the module since filename starts with a number
spec = importlib.util.spec_from_file_location("atro_ingest", "7atro_ingest.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
AstroConfig = module.AstroConfig
RAGQueryEngine = module.RAGQueryEngine

# Initialize the query engine once in session state
if "query_engine" not in st.session_state:
    config = AstroConfig()
    st.session_state.query_engine = RAGQueryEngine(config)

# Set up the Streamlit page
st.set_page_config(page_title="Local RAG Assistant", page_icon="📚")
st.title("📚 Local Document RAG Pipeline")
st.markdown("Powered by Qdrant, LangChain, and Gemini 2.5 Pro")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            for img_path in message["images"]:
                st.image(img_path, caption="Matched Image")

# Wait for user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display the user's prompt
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save user prompt to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display a loading spinner while Gemini thinks
    with st.chat_message("assistant"):
        with st.spinner("Searching local Qdrant database..."):
            # Call the new OO RAG Query Engine
            response_text, image_paths = st.session_state.query_engine.ask_question(prompt)
            
            st.markdown(response_text)
            for img_path in image_paths:
                st.image(img_path, caption="Matched Image")
            
    # Save AI response to history
    st.session_state.messages.append({"role": "assistant", "content": response_text, "images": image_paths})