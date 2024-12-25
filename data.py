from datasets import load_dataset

# Load the English QA subset (correct config)
dataset = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa")

# Print the dataset to see the structure
print(dataset)

# Access the 'train' split and print the first element
first_sample = dataset['train'][0]
print(first_sample)