# models/llm_model.py

from langchain_openai import ChatOpenAI

def get_llm_model(model_name="gpt-3.5-turbo", temperature=0.0):
    return ChatOpenAI(model=model_name, temperature=temperature)
