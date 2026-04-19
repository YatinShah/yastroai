import streamlit as st
import importlib.util

# Load the module since filename starts with a number
spec = importlib.util.spec_from_file_location("ask_rag", "4ask_rag.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ask_question = module.ask_question 

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
            # Call your LangChain RAG function
            # Make sure ask_question() returns the final text instead of just printing it
            response_text = ask_question(prompt) 
            
            st.markdown(response_text)
            
    # Save AI response to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})