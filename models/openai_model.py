#openai_model.py
import os
import math
from openai import OpenAI
from dotenv import load_dotenv
import torch

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def query_openai(prompt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
            logprobs=True  
        )
        choice = completion.choices[0]
        model_answer = choice.message.content
        logprob_conf = math.exp(choice.logprobs.content[0].logprob)
        
        return model_answer, logprob_conf  
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return "OPEN AI ERROR", None