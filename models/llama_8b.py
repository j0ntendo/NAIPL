import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


class Llama8B:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct"
        )

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
        return AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
        )

    def query(self, prompt, max_new_tokens=100):
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)
        inputs["attention_mask"] = inputs["input_ids"].ne(self.tokenizer.pad_token_id)

        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            generated_text = self.tokenizer.decode(
                outputs.sequences[0], skip_special_tokens=True
            )
            avg_logprob, perplexity = self._calculate_logprobs(outputs, inputs)

            return generated_text, perplexity

        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory. Switching to CPU...")
            self.model = self.model.to("cpu")
            return self.query(prompt)

        except Exception as e:
            print(f"Error during inference: {e}")
            return None, None

    def _calculate_logprobs(self, outputs, inputs):
        # Compute transition scores (token logprobs)
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]

        logprobs = []
        for tok, score in zip(generated_tokens[0], transition_scores[0]):
            logprobs.append(score.detach().cpu().item())

        avg_logprob = np.mean(logprobs) if logprobs else float("-inf")
        perplexity = np.exp(-avg_logprob)  # Perplexity is exp(-avg log prob)

        # Log token probabilities for visualization
        print("\n| Token | Token String | Log Prob | Probability |")
        print("-" * 50)
        for tok, score in zip(generated_tokens[0], transition_scores[0]):
            print(
                f"| {tok.item():5d} | {self.tokenizer.decode(tok):8s} | {score.detach().cpu().numpy():.4f} | {np.exp(score.detach().cpu().numpy()):.2%}"
            )

        return avg_logprob, perplexity


# # Testing the model
# llama8b = Llama8B()
# prompt = "What are the symptoms of a heart attack?"
# response, perplexity = llama8b.query(prompt)

# print(f"\nGenerated Response:\n{response}")
# print(f"Perplexity Score: {perplexity:.4f}")
