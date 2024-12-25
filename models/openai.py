import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def query_openai(prompt, model="gpt-4o-mini"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()