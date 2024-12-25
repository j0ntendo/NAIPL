import os
import wandb
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.openai import query_openai
from models.hf import query_all_hf_models
from models.claude import query_claude
from utils.load_data import load_medqa
from evaluation.scorer import evaluate_bleu
from evaluation.benchllama import run_benchllama
from dotenv import load_dotenv

load_dotenv()
wandb.init(project="NAIPL", name="MedQA-Full-Pipeline")

HF_MODELS = {
    "llama_70b": "meta-llama/Llama-3.3-70B-Instruct",
    "medllama3": "ProbeMedicalYonseiMAILab/medllama3-v20",
    "openbio_llm": "aaditya/Llama3-OpenBioLLM-8Bl"
}

def evaluate_question(question, context, choices, reference_answer):
    gpt_answer = query_openai(question, context, choices)
    claude_answer = query_claude(question, context, choices)
    hf_answers = query_all_hf_models(question, context, choices)

    # BLEU 
    gpt_score = evaluate_bleu(gpt_answer, reference_answer)
    claude_score = evaluate_bleu(claude_answer, reference_answer)
    hf_scores = {model: evaluate_bleu(hf_answers[model], reference_answer) for model in HF_MODELS}

    # BenchLLaMA
    benchllama_scores = {model_key: run_benchllama(HF_MODELS[model_key]) for model_key in HF_MODELS}


    wandb.log({
        "question": question,
        "context": context,
        "gpt_score": gpt_score,
        "claude_score": claude_score,
        **{f"{model}_bleu": hf_scores[model] for model in HF_MODELS},
        **{f"{model}_benchllama": benchllama_scores[model]['score'] for model in HF_MODELS}
    })

    return {
        "question": question,
        "context": context,
        "choices": choices,
        "gpt_answer": gpt_answer,
        "claude_answer": claude_answer,
        "hf_answers": hf_answers,
        "gpt_score": gpt_score,
        "claude_score": claude_score,
        "hf_scores": hf_scores,
        "benchllama_scores": benchllama_scores
    }

def run_evaluation():
    dataset = load_medqa()
    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for example in dataset:
            question = example['question']
            context = example['context']
            choices = example['choices']
            reference_answer = example['answer']
            future = executor.submit(evaluate_question, question, context, choices, reference_answer)
            futures.append(future)

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Eval complete for question: {result['question'][:50]}...")

    wandb.finish()
    print("Evaluation completed.")

if __name__ == "__main__":
    run_evaluation()