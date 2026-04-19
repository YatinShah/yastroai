##LLM as judge exmaple.

import vertexai
from vertexai.generative_models import GenerativeModel
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
    vertexai.init(project=PROJECTID, location=REGION)
    
    # We use an evaluator model with zero creativity (temperature=0)
    evaluator = GenerativeModel("gemini-2.5-pro", generation_config={"temperature": 0.0})
    
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
    result = evaluator.generate_content(eval_prompt)
    print("\n📊 --- Evaluation Results ---")
    print(result.text)

if __name__ == "__main__":
    evaluate_pipeline()