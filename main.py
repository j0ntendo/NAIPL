import torch
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['TRANSFORMERS_CACHE'] = '/home/work/naipl-framework/huggingface_cache'

import wandb
from huggingface_hub import login
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.openai_model import query_openai
from models.claude import query_claude
#from models.openbio_llm import OpenBioLLM
#from models.llama3 import  query_llama
#from models.llama_70b import Llama70B
from models.llama_8b import Llama8B
from datasets import load_dataset
import re
import json
from contextlib import redirect_stdout
import io

openai_model = query_openai  
claude_model = query_claude  
llama_8b_model = Llama8B()  
#llama_70b_model = Llama70B()  
#openbio_llm_model = OpenBioLLM()
#llama3 = query_llama

wandb.init(project="NAIPL-FRAMEWORK", name="MedQA-Full-Pipeline")

def load_medqa(sample_size=5):
    dataset = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa")['train']
    sample_size = min(sample_size, len(dataset))
    sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))
    processed_data = sampled_dataset.map(lambda x: {
        'id': x['id'], 
        'question_id': x['question_id'],
        'document_id': x['document_id'],
        'question': x['question'],
        'context': x['context'] if x['context'] else "", 
        'choices': x['choices'],
        'answer': x['answer'][0] if isinstance(x['answer'], list) else x['answer']  
    })

    return processed_data

def create_prompt(question, context, choices):
    user_confidence_query = (
        "Before answering, rate your confidence in understanding and interpreting this medical question on a scale from 1 to 10 (UConf.). "
        "Immediately provide the correct answer in the format: Answer: (A/B/C/D/E). "
        "After giving the answer, explain your reasoning and rate how confident you are in your answer (MConf.). "
        "Ensure to conclude your response with: UConf: [1-10] and MConf: [1-10]. "
    )
    choices_text = '\n'.join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
    return f"{user_confidence_query}\n\nContext: {context}\n\nQuestion: {question}\n\nOptions:\n{choices_text}\n"

explainability_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output. ",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o",
    verbose_mode=True,
    #threshold=0.75
)

def extract_user_confidence(input):
    match = re.search(r"UConf:\s*(\d+)", input_text)
    if match:
        return int(match.group(1))
    return None

def extract_model_confidence(output):
    match = re.search(r"MConf:\s*(\d+)", output_text)
    if match:
        return int(match.group(1))
    return None

def extract_answer(output_text):
    match = re.search(r'Answer:\s*\(?([A-E])\)?', output_text, re.IGNORECASE)
    if match:
        return match.group(1).strip().upper()
    return None

def is_correct(model_output, reference_answer):
    extracted_answer = extract_answer(model_output)
    return extracted_answer == reference_answer

def geval(model_query_function, prompt, reference_answer, index, model_name):
    try:
        
        if model_name == "GPT-4o":
            model_answer, logprob_conf = model_query_function(prompt)
        else:
            model_answer = model_query_function(prompt)
            logprob_conf = None  

        user_confidence = extract_user_confidence(model_answer)
        model_confidence = extract_model_confidence(model_answer)
        predicted_answer = extract_answer(model_answer)
        is_correct, extracted_answer = (predicted_answer == reference_answer, predicted_answer)

        test_case = LLMTestCase(
            input=prompt,
            actual_output=model_answer,
            expected_output=reference_answer
        )
        explainability_metric.measure(test_case)

        
        wandb.log({
            "index": index,
            "model_name": model_name,
            "prompt": prompt,
            "user_confidence": user_confidence,
            "model_answer": model_answer,
            "predicted_answer": extracted_answer,
            "correct_answer": reference_answer,
            "is_correct": is_correct,
            "model_confidence": model_confidence,
            "logprob_perplexity": logprob_conf,
            "explainability_score": explainability_metric.score,
            "explanation": explainability_metric.reason
        })

        
        return {
            "index": index,
            "model_name": model_name,
            "prompt": prompt,
            "user_confidence": user_confidence,
            "model_answer": model_answer,
            "predicted_answer": extracted_answer,
            "correct_answer": reference_answer,
            "is_correct": is_correct,
            "model_confidence": model_confidence,
            "logprob_perplexity": logprob_conf,
            "explainability_score": explainability_metric.score,
            "explanation": explainability_metric.reason
        }

    except Exception as e:
        print(f"An error occurred during model evaluation: {str(e)}")
        return {
            "index": index,
            "model_name": model_name,
            "prompt": prompt,
            "user_confidence": None,
            "model_answer": None,
            "predicted_answer": None,
            "correct_answer": reference_answer,
            "is_correct": False,
            "model_confidence": None,
            "logprob_confidence": None,
            "explainability_score": 0,
            "error": str(e)
        }

def run_evaluation():
    dataset = load_medqa(2) 
    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        for index, example in enumerate(dataset):
            prompt = create_prompt(example['question'], example['context'], example['choices'])
            reference_answer = example['answer']

            futures.append(executor.submit(geval, query_openai, prompt, reference_answer, index, "GPT-4o"))
            futures.append(executor.submit(geval, query_claude, prompt, reference_answer, index, "Claude-3"))
            futures.append(executor.submit(geval, llama_8b_model.query, prompt, reference_answer, index, "LLaMA-8B"))

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    results.sort(key=lambda x: (x['index'], x['model_name']))

    json_file = "testgpt.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {json_file}")
    wandb.finish()
    print("Evaluation completed.")

if __name__ == "__main__":
    run_evaluation()