##LLM as judge exmaple.

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import streamlit as st
import importlib.util

# Load the module since filename starts with a number
spec = importlib.util.spec_from_file_location("ask_rag", "4ask_rag.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ask_question = module.ask_question 


PROJECTID = "atroai"
REGION = "us-central1"

def evaluate_pipeline():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    # Define your test case based on your sample outputs
    test_question = "What is the primary objective of the documents you ingested?"  # Adjust this question based on your PDF content
    expected_answer = "The documents ingested describe astrology, and how to prepares ones horoscope."
    
    print(f"Testing Question: {test_question}")
    
    # 1. Get the actual answer from your local Qdrant/LangChain pipeline
    pipeline_response = ask_question(test_question)
    
    # 2. Build the Evaluation Prompt
    eval_prompt = f"""
    You are an impartial judge evaluating an AI's response.
    Compare the AI's Actual Answer to the Expected Ground Truth Answer.
    
    Question: {test_question}
    Expected Answer: {expected_answer}
    Actual Answer: {pipeline_response}
    
    Score the Actual Answer from 1 to 5 based on factual accuracy and completeness. 
    Provide a brief 1-sentence justification, then output the score like this: 'SCORE: X'.
    """
    
    # 3. Grade the answer
    result = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=eval_prompt,
        config=types.GenerateContentConfig(temperature=0.0)
    )
    print("\n📊 --- Evaluation Results ---")
    print(result.text)

if __name__ == "__main__":
    evaluate_pipeline()