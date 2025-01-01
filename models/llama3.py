import ollama
from ollama import generate
ollama.pull('llama3.3')

def query_llama(query):
    try:
        model = "llama3.3"
        response = generate(model = model, prompt = query)
        return response['response'] 
    except Exception as e:
        print(f"An error occurred while querying the model: {e}")
        return "Error in generating response from llama."

