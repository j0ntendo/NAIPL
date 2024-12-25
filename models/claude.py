import anthropic
import os
from dotenv import load_dotenv

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(api_key=anthropic_api_key)

def format_with_choices(question, choices):
    choices_str = '\n'.join([f"{chr(65 + i)}) {choice}" for i, choice in enumerate(choices)])
    return f"{question}\nChoices: {choices_str}"

def query_claude(question, context, choices, model="claude-3-haiku-20240307"):
    formatted_question = format_with_choices(question, choices)
    prompt = f"Context: {context}\n{formatted_question}\nAnswer:"

    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.content[0].text.strip()