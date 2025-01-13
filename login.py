import os
from huggingface_hub import login
from dotenv import load_dotenv
import wandb


def setup_environment():
    load_dotenv()

    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=hf_token)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)


if __name__ == "__main__":
    setup_environment()
