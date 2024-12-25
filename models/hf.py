# hf.py
from transformers import pipeline

llama_70b_qa = pipeline("text-generation", model="meta-llama/Llama-3.3-70B-Instruct")
medllama3_qa = pipeline("text-generation", model="ProbeMedicalYonseiMAILab/medllama3-v20")
openbio_llm_qa = pipeline("text-generation", model="aaditya/Llama3-OpenBioLLM-8Bl")

def query_all_hf_models(question, context):
    return {
        "llama_70b": llama_70b_qa(question=question, context=context)['answer'],
        "medllama3": medllama3_qa(question=question, context=context)['answer'],
        "openbio_llm": openbio_llm_qa(question=question, context=context)['answer']
    }