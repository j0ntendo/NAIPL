import os
from huggingface_hub import login
from dotenv import load_dotenv
import wandb

def setup_environment():
    
    load_dotenv()

    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✅ Successfully logged into Hugging Face!")
    else:
        print("❌ Hugging Face token not found. Please check your .env file.")

    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        print("✅ OpenAI API key set.")
    else:
        print("❌ OpenAI API key not found. Please check your .env file.")

    
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        print("✅ Anthropic API key set.")
    else:
        print("❌ Anthropic API key not found. Please check your .env file.")

    
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        print("✅ Successfully logged into Weights & Biases (wandb)!")
    else:
        print("❌ Weights & Biases API key not found. Please check your .env file.")

    print("\nEnvironment setup complete!")

if __name__ == "__main__":
    setup_environment()