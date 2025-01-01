import anthropic
import torch
import os
from dotenv import load_dotenv

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=anthropic_api_key)

def query_claude(prompt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        print(f"Error querying Claude: {e}")
        return "CLAUDE ERROR"