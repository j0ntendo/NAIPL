import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

llm_instances = {}


class LLMModel:
    def __init__(self, model_name, gpu=0):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device(
            f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    def query_hf(self, prompt, max_new_tokens=400):
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def evaluate_response(self, input_text, max_tokens=200):
        # Format the evaluation prompt
        prompt = (
            "Evaluate the chatbot's accuracy by comparing the RESPONSE with the CORRECT ANSWER below. "
            "Return 'True' if the RESPONSE clearly matches the CORRECT ANSWER. "
            "Return 'False' if the RESPONSE contradicts the CORRECT ANSWER. "
            "Return 'NR' if the RESPONSE is unclear, partially related, or ambiguous.\n\n"
            f"{input_text}\n"
            "\nYour evaluation (True, False, NR):"
        )

        # Tokenize and generate
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


def query_hf(prompt, model_name, gpu=0):
    global llm_instances
    if model_name not in llm_instances:
        llm_instances[model_name] = LLMModel(model_name, gpu)

    return llm_instances[model_name].query_hf(prompt)
