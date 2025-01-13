import os
from dotenv import load_dotenv

load_dotenv()
os.environ["TRANSFORMERS_CACHE"] = "/home/work/naipl-framework/huggingface_cache"


# from models.openai_model import query_openai
# from models.claude import query_claude
from models.hf import LLMModel
from utils.evaluation import gen_prompt
import json
from tqdm import tqdm

# wandb.init(project="NAIPL-FRAMEWORK", name="MedQA-Full-Pipeline")

# load dataset
DATA_DIR = "./filtered_data"
RESULTS_DIR = "./evaluation_results"


def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


llama_model = LLMModel("meta-llama/Llama-3.3-70B-Instruct", gpu=0)


def run_evaluation():
    results = []

    # Iterate through all jsonl files in the data directory
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(DATA_DIR, filename)
            dataset = load_jsonl(file_path)  # Load jsonl data

            print(f"Evaluating {filename}...")

            # Sequential processing for LLaMA (no ThreadPoolExecutor)
            for index, example in enumerate(tqdm(dataset)):
                prompts = gen_prompt(example)
                reference_answer = example["answer"]

                # Process each prompt variation
                for prompt_obj in prompts:
                    prompt = prompt_obj["prompt"]
                    metadata = prompt_obj["metadata"]

                    # Query the existing LLaMA model instance
                    response = llama_model.query_hf(prompt)

                    # Save result
                    results.append(
                        {
                            "index": index,
                            "model_name": "LLaMA-3.3-70B",
                            "prompt": prompt,
                            "response": response,
                            "real_answer": reference_answer,
                            "is_correct": response.strip() == reference_answer,
                            "metadata": metadata,
                        }
                    )

            # Save results per file
            os.makedirs(RESULTS_DIR, exist_ok=True)
            output_file = os.path.join(
                RESULTS_DIR, f"results_{filename.replace('.jsonl', '')}.json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)

            print(f"Results saved to {output_file}")


# def run_evaluation():
#     dataset = load_medqa(5)
#     results = []

#     with ThreadPoolExecutor(max_workers=1) as executor:
#         futures = []

#         for index, example in enumerate(dataset):
#             prompt = create_prompt(example['question'], example['context'], example['choices'])
#             reference_answer = example['answer']

#             # Uncomment to evaluate OpenAI and Claude models
#             # futures.append(executor.submit(start_evaluating, query_openai, prompt, reference_answer, example['choices'], index, "GPT-4o"))
#             # futures.append(executor.submit(start_evaluating, query_claude, prompt, reference_answer, example['choices'], index, "Claude-3"))

#             # Submit HF models to executor
#             for model_name, model_instance in hf_models.items():
#                 futures.append(
#                     executor.submit(
#                         start_evaluating,
#                         lambda p, instance=model_instance: instance.query_hf(p),
#                         prompt,
#                         reference_answer,
#                         example['choices'],
#                         index,
#                         model_name
#                     )
#                 )

#         for future in as_completed(futures):
#             results.append(future.result())

#     results.sort(key=lambda x: (x['index'], x['model_name']))

#     output_file = "evaluation_results.json"
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=4)

#     print(f"Results saved to {output_file}")
#     wandb.finish()
#     print("Evaluation completed.")

if __name__ == "__main__":
    run_evaluation()
