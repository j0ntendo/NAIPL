from datasets import load_dataset
import pandas as pd  # Useful for viewing dataset structure in a tabular format

def load_medqa(sample_size=10):
    # Load the dataset from the Hugging Face datasets library
    dataset = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa")['train']
    
    # Limit the dataset to a manageable size for testing
    sample_size = min(sample_size, len(dataset))
    sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))

    # Process the data to format it correctly
    processed_data = sampled_dataset.map(lambda x: {
        'id': x['id'],
        'question_id': x['question_id'],
        'document_id': x['document_id'],
        'question': x['question'],
        'context': x['context'] if x['context'] else "",
        'choices': ', '.join(x['choices']),  # Join choices into a single string separated by commas
        'answer': x['answer'][0] if isinstance(x['answer'], list) else x['answer']
    })

    # Convert to DataFrame for easy viewing and manipulation
    df = pd.DataFrame(processed_data)
    print(df.head())  # Print the first few rows of the dataframe
    return processed_data

def create_prompt(question, context, choices):
    # Format choices as a numbered list
    choices_text = '\n'.join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices.split(', '))])
    # Explicitly instruct the model to select the best option
    prompt_instruction = "\nChoose the best option from the above choices and explain why it is the most appropriate next step in management:"
    return f"{context}\nQuestion: {question}\nOptions:\n{choices_text}{prompt_instruction}"



# Example usage:
data = load_medqa(5)  # Load 5 examples for testing
for item in data:
    prompt = create_prompt(item['question'], item['context'], item['choices'])
    print(prompt)  # View how the prompt looks