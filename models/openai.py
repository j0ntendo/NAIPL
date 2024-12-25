# openai.py
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def query_openai(question, context, model="gpt-4o-mini"):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=200,
        stop=["\n"]
    )
    return response.choices[0].text.strip()