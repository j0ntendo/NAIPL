from transformers import pipeline

llama_70b = pipeline("question-answering", model="meta-llama/Llama-3.3-70B-Instruct")
medllama3 = pipeline("question-answering", model="ProbeMedicalYonseiMAILab/medllama3-v20")
openbio_llm = pipeline("question-answering", model="aaditya/Llama3-OpenBioLLM-8Bl")

def format_with_choices(question, choices):
    choices_str = '\n'.join([f"{chr(65 + i)}) {choice}" for i, choice in enumerate(choices)])
    return f"{question}\nChoices: {choices_str}"

def query_models(question, choices, context=None):
    formatted_question = format_with_choices(question, choices)
    full_context = f"{context}\n{formatted_question}" if context else formatted_question
    results = {
        "llama_70b": llama_70b(question=formatted_question, context=context)['answer'] if context else llama_70b(question=formatted_question)['answer'],
        "medllama3": medllama3(question=formatted_question, context=context)['answer'] if context else medllama3(question=formatted_question)['answer'],
        "openbio_llm": openbio_llm(question=formatted_question, context=context)['answer'] if context else openbio_llm(question=formatted_question)['answer']
    }
    return results