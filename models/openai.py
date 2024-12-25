import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def format_with_choices(question, choices):
    choices_str = '\n'.join([f"{chr(65 + i)}) {choice}" for i, choice in enumerate(choices)])
    return f"{question}\nChoices: {choices_str}"

def query_openai(question, context, choices, model="gpt-4o-mini"):
    formatted_question = format_with_choices(question, choices)
    prompt = f"Context: {context}\n{formatted_question}\nAnswer:"

    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=200,
        stop=["\n"] 
    )
    return response.choices[0].text.strip()