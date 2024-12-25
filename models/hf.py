from transformers import pipeline


llama_70b = pipeline("text-generation", model="meta-llama/Llama-3.3-70B-Instruct")
medllama3 = pipeline("text-generation", model="ProbeMedicalYonseiMAILab/medllama3-v20")
openbio_llm = pipeline("text-generation", model="aaditya/Llama3-OpenBioLLM-8Bl")

def query_all_hf_models(prompt):
    return {
        "llama_70b": llama_70b(prompt, max_length=200, truncation=True)[0]["generated_text"],
        "medllama3": medllama3(prompt, max_length=200, truncation=True)[0]["generated_text"],
        "openbio_llm": openbio_llm(prompt, max_length=200, truncation=True)[0]["generated_text"]
    }