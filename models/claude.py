# claude.py
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=anthropic_api_key)

def query_claude(question, context, model="claude-3-haiku-20240307"):  
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300 
    )
    return response.content[0].text.strip()