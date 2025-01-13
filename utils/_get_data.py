from datasets import load_dataset


def load_medqa(sample_size=5):
    dataset = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa")["train"]
    sample_size = min(sample_size, len(dataset))
    sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))
    processed_data = sampled_dataset.map(
        lambda x: {
            "id": x["id"],
            "question_id": x["question_id"],
            "document_id": x["document_id"],
            "question": x["question"],
            "context": x["context"] if x["context"] else "",
            "choices": x["choices"],
            "answer": x["answer"][0] if isinstance(x["answer"], list) else x["answer"],
        }
    )
    return processed_data
