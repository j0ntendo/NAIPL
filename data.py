from datasets import load_dataset

# Load the English QA subset (correct config)
dataset = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa")
print(dataset)