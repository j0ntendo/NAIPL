# openbio_llm.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class OpenBioLLM:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("aaditya/Llama3-OpenBioLLM-8B")
        self.model = AutoModelForCausalLM.from_pretrained("aaditya/Llama3-OpenBioLLM-8B")
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def query(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        outputs = self.model.generate(inputs, max_length=1024)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
