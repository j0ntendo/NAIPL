import orjson


def transform_json(old_file_path, new_file_path):
    with open(old_file_path, "rb") as infile:
        old_data = orjson.loads(infile.read())

    new_data = []

    for item in old_data:
        old_label = item.get("label", "")
        old_model_name = item.get("model_name", "")
        old_prompt = item.get("prompt", "")
        old_response = item.get("response", "")
        correct_ans = item.get("correct_answer", "")
        metadata = item.get("metadata", {})

        first_line = old_response.split("\n")[0].strip()
        splitted = first_line.split(")")
        if len(splitted) > 1:
            extracted_answer_raw = splitted[1].strip()
        else:
            extracted_answer_raw = first_line

        extracted_answer = extracted_answer_raw.split(".")[0].strip()
        is_correct = extracted_answer.lower() == correct_ans.lower()

        new_item = {
            "label": old_label,
            "model_name": old_model_name,
            "prompt": old_prompt,
            "response": old_response,
            "correct_answer": correct_ans,
            "is_correct": is_correct,
            "metadata": metadata,
        }

        new_data.append(new_item)

    with open(new_file_path, "wb") as outfile:
        outfile.write(orjson.dumps(new_data, option=orjson.OPT_INDENT_2))


transform_json("./results/claude/claude3_test.json", "clean_claude3_test.json")
transform_json("./results/claude/claude3_dev.json", "clean_claude3_dev.json")
# import os
# import orjson

# def transform_json(json_path):
#     """
#     Reads the JSON array from json_path using orjson,
#     updates each item to:
#       - build a label from metadata (ai_role, physician_description, tone),
#       - set model_name = "claude-3-5-haiku",
#       - re-check is_correct (optional).
#     Then writes the updated array back to json_path.
#     """
#     # 1) Read JSON with orjson (binary mode)
#     with open(json_path, "rb") as infile:
#         data = orjson.loads(infile.read())  # Expecting a list of dicts

#     for item in data:
#         # 2) Construct label from metadata
#         metadata = item.get("metadata", {})
#         ai_role = metadata.get("ai_role", "expert")
#         physician_desc = metadata.get("physician_description", "expert")
#         tone = metadata.get("tone", "assertive")

#         label = (
#             f"{ai_role.capitalize()} AI / "
#             f"{physician_desc.capitalize()} Physician / "
#             f"{tone.capitalize()} Tone"
#         )
#         item["label"] = label

#         # 3) Set model_name
#         item["model_name"] = "claude-3-5-haiku"

#         # 4) Re-check correctness if needed
#         #    For instance, parse the user's chosen answer from the 'response'
#         #    Compare to 'real_answer' (or 'correct_answer', as your data may have).
#         real_answer = item.get("real_answer") or item.get("correct_answer")
#         if real_answer:
#             response_text = item.get("response", "")
#             # e.g., if your logic is: the first line might be "(B) Arthritis ...",
#             # then parse out the piece after the parenthesis
#             first_line = response_text.strip().split("\n", 1)[0]

#             # Try to remove parentheses if present
#             # e.g. (B) Arthritis --> parted[0] = '(B', parted[1] = 'Arthritis'
#             parted = first_line.split(")", 1)
#             if len(parted) == 2:
#                 chosen_answer = parted[1].strip()
#             else:
#                 chosen_answer = first_line.strip()

#             # Compare, ignoring case
#             item["is_correct"] = chosen_answer.lower() == real_answer.lower()

#         # If there's no real_answer key, we won't modify is_correct

#     # 5) Write updated data back (in binary mode) with orjson
#     with open(json_path, "wb") as outfile:
#         outfile.write(
#             orjson.dumps(data, option=orjson.OPT_INDENT_2)
#         )

#     print(f"Updated file: {json_path}")

# def main():
#     # Update both dev/test files in place
#     files_to_update = [
#         "results/claude/claude3_dev.json",
#         "results/claude/claude3_test.json"
#     ]

#     for fpath in files_to_update:
#         if os.path.exists(fpath):
#             transform_json(fpath)
#         else:
#             print(f"File not found: {fpath}")

# if __name__ == "__main__":
#     main()
