import json
import openai
import os
from tqdm import tqdm


class GPT4MiniEvaluator:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def evaluate_answer(self, response, correct_answer):
        prompt = (
            "Evaluate the chatbot's accuracy by comparing the RESPONSE with the CORRECT ANSWER below. "
            "Return 'True' if the RESPONSE clearly matches the CORRECT ANSWER. "
            "Return 'False' if the RESPONSE contradicts the CORRECT ANSWER. "
            "Return 'NR' if the RESPONSE is unclear, partially related, or ambiguous.\n\n"
            f"RESPONSE: {response}\n"
            f"CORRECT ANSWER: {correct_answer}\n"
            "\nYour evaluation (True, False, NR):"
        )

        response = openai.Completion.create(
            model=self.model_name, prompt=prompt, max_tokens=500
        )
        return self._extract_evaluation(response["choices"][0]["text"])

    def _extract_evaluation(self, text):
        match = re.search(r"(True|False|NR)", text, re.IGNORECASE)
        return match.group(0) if match else "NR"


def process_results(json_entry, evaluator):
    response = json_entry["response"]
    correct_answer = json_entry["real_answer"]
    evaluation = evaluator.evaluate_answer(response, correct_answer)
    json_entry["is_correct"] = evaluation  # Update result to JSON entry
    print(f"Evaluation for index {json_entry['index']}: {evaluation}")
    return json_entry


def evaluate_from_file(input_file, output_file):
    evaluator = GPT4MiniEvaluator()

    with open(input_file, "r") as f:
        data = json.load(f)

    evaluated_results = []
    for entry in tqdm(data, desc="Evaluating Responses"):
        evaluated_results.append(process_results(entry, evaluator))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluated_results, f, indent=4)

    print(f"Evaluation completed. Results saved to {output_file}")


if __name__ == "__main__":
    input_file = "results/results_short.json"
    output_file = "results/evaluated_results_short.json"
    evaluate_from_file(input_file, output_file)
