import torch
from transformers import AutoTokenizer, AutoDistributedModelForCausalLM
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

def initialize_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_bos_token=False, use_auth_token=hf_token)
    model = AutoDistributedModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
    model = model.cuda() 
    return model, tokenizer

def query_model(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")["input_ids"].cuda()
    outputs = model.generate(inputs, max_new_tokens=1024)
    return tokenizer.decode(outputs[0])


llama_70b_model, llama_70b_tokenizer = initialize_model("meta-llama/Llama-3.3-70B-Instruct")
#medllama3_model, medllama3_tokenizer = initialize_model("ProbeMedicalYonseiMAILab/medllama3-v20")
openbio_llm_model, openbio_llm_tokenizer = initialize_model("aaditya/Llama3-OpenBioLLM-8Bl")

#example
if __name__ == "__main__":
    prompt = "Explain the theory of relativity."
    print(query_model(llama_70b_model, llama_70b_tokenizer, prompt))