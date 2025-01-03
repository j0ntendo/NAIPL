import os
import math
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import torch

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4",
    max_tokens=500,
    temperature=0,
    stop=None,
    seed=123,
    tools=None,
    logprobs=True,
    top_logprobs=2,
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion

def query_openai(prompt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_used = "gpt-4o"  
    
    try:
        response = get_completion(
            [{"role": "user", "content": prompt}],
            model=model_used,
            logprobs=True
        )
        
        if response and response.choices:
            choice = response.choices[0]
            logprobs = [token.logprob for token in choice.logprobs.content]
            perplexity_score = np.exp(-np.mean(logprobs))  
            logprob_conf = perplexity_score

            return choice.message.content, logprob_conf

        else:
            return "No response from API", None

    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return "OPEN AI ERROR", None


if __name__ == "__main__":
    test_prompt = "Before answering, rate your confidence in understanding and interpreting this medical question on a scale from 1 to 10 (UConf.). Immediately provide the correct answer in the format: Answer: (A/B/C/D/E). After giving the answer, explain your reasoning and rate how confident you are in your answer (MConf.). Ensure to conclude your response with: UConf: [1-10] and MConf: [1-10]. Question: A 36-year-old man is brought to the emergency department by his wife 20 minutes after having a seizure. Over the past 3 days, he has had a fever and worsening headaches. This morning, his wife noticed that he was irritable and demonstrated strange behavior; he put the back of his fork, the salt shaker, and the lid of the coffee can into his mouth. He has no history of serious illness and takes no medications. His temperature is 39°C (102.2°F), pulse is 88/min, and blood pressure is 118/76 mm Hg. Neurologic examination shows diffuse hyperreflexia and an extensor response to the plantar reflex on the right. A T2-weighted MRI of the brain shows edema and areas of hemorrhage in the left temporal lobe. Which of the following is most likely the primary mechanism of the development of edema in this patient?\n\nOptions:\n(A) Release of vascular endothelial growth factor\n(B) Cellular retention of sodium\n(C) Breakdown of endothelial tight junctions\n(D) Degranulation of eosinophils\n(E) Increased hydrostatic pressure\n"
    response, perplexity = query_openai(test_prompt)

    print(f"Response: {response}")
    print(f"Perplexity Score: {perplexity}")