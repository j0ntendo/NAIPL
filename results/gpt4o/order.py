import json


def update_correct_answers(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    for item in data:
        response_text = item.get("response", "")
        real_answer = item.get("real_answer", "")
        first_line = response_text.split("\n", 1)[0].strip()

        parted = first_line.split(")", 1)
        if len(parted) == 2:
            extracted_answer = parted[1].strip()
        else:
            extracted_answer = first_line

        is_correct_now = extracted_answer.lower() == real_answer.lower()

        item["is_correct"] = is_correct_now

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=4)

    print(f"Updated data saved to {output_path}")


if __name__ == "__main__":
    input_json = "results/gpt4o/gpt4o_dev.json"
    output_json = "results/gpt4o/clean_gpt4o_dev.json"

    update_correct_answers(input_json, output_json)
