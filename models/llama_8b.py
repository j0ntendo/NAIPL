import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Llama8B:
    def __init__(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

        
        if torch.cuda.device_count() > 1:
            self.model = self._parallelize_model()
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)

        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()  

    def _parallelize_model(self):
        print("Distributing model across GPUs...")

        
        return self.model.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16
        )

    def query(self, prompt, max_new_tokens=100):
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        
        inputs['attention_mask'] = inputs['input_ids'].ne(self.tokenizer.pad_token_id)

        try:
            
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

            
            generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

            
            logprobs = self._calculate_logprobs(outputs, inputs['input_ids'])

            
            avg_logprob = sum(logprobs) / len(logprobs) if logprobs else None

            return generated_text, avg_logprob

        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory. Switching to CPU...")
            self.model = self.model.to("cpu")  
            return self.query(prompt)

        except Exception as e:
            print(f"Error during inference: {e}")
            return None, None

    def _calculate_logprobs(self, outputs, input_ids):
        
        logits = torch.cat([score.unsqueeze(1) for score in outputs.scores], dim=1)
        
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.log(probs)

        sequences = outputs.sequences[0]

        
        logprobs = []
        for i, token_id in enumerate(sequences[1:]):
            if i < log_probs.shape[1]:
                logprob = log_probs[0, i, token_id].item()
                logprobs.append(logprob)

        return logprobs