# answer_extractor.py
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LlamaEvaluator:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", gpu_index=-1):
        self.device = (
            torch.device("cpu")
            if gpu_index == -1
            else torch.device(f"cuda:{gpu_index}")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def evaluate_answer(self, input_text, max_tokens=200):
        prompt = (
            "Evaluate the chatbot's accuracy by comparing the RESPONSE with the CORRECT ANSWER below. "
            "Return 'True' if the RESPONSE clearly matches the CORRECT ANSWER. "
            "Return 'False' if the RESPONSE contradicts the CORRECT ANSWER. "
            "Return 'NR' if the RESPONSE is unclear, partially related, or ambiguous.\n\n"
            f"{input_text}\n"
            "\nYour evaluation (True, False, NR):"
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        evaluation = self._extract_evaluation(generated_text)
        return evaluation

    def _extract_evaluation(self, text):
        match = re.search(r"(True|False|NR)", text, re.IGNORECASE)
        return match.group(0) if match else "NR"
