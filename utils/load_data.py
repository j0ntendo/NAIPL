from datasets import load_dataset

def load_medqa(sample_size=100):
    dataset = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa")['train']
    return dataset.select(range(min(sample_size, len(dataset))))