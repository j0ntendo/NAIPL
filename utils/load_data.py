from datasets import load_dataset

def load_medqa(sample_size=100):
        dataset = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa")['train']
        
        sample_size = min(sample_size, len(dataset))
        sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))

        processed_data = sampled_dataset.map(lambda x: {'context': x['context'], 'question': x['question'], 'answer': x['answer']})

        return processed_data
