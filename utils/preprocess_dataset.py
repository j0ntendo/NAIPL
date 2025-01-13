import os
import json

# Define input and output directories
DATA_DIR = "./4_options"
OUTPUT_DIR = "./filtered_data"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

KEYWORD = "which of the following is the most likely diagnosis?"


def filter_by_keyword(file_path, output_path, keyword):
    filtered_data = []

    # Read and filter the jsonl file
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if keyword.lower() in data["question"].lower():
                filtered_data.append(data)

    # Save filtered data to a new jsonl file
    with open(output_path, "w") as f:
        for item in filtered_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(filtered_data)} filtered questions to {output_path}")


def process_all_files():
    # Process each jsonl file in the dataset directory
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".jsonl"):
            input_path = os.path.join(DATA_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, f"filtered_{filename}")

            print(f"Processing {filename}...")
            filter_by_keyword(input_path, output_path, KEYWORD)


if __name__ == "__main__":
    process_all_files()
