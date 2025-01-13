import orjson


def merge_json_files(file1_path, file2_path, output_file_path):
    """
    Merges two JSON files into one and writes the result to a new file.

    :param file1_path: Path to the first JSON file.
    :param file2_path: Path to the second JSON file.
    :param output_file_path: Path to the output JSON file.
    """
    try:
        # Read the first JSON file
        with open(file1_path, "rb") as f1:
            data1 = orjson.loads(f1.read())

        # Read the second JSON file
        with open(file2_path, "rb") as f2:
            data2 = orjson.loads(f2.read())

        # Ensure the data is a list and merge
        if not isinstance(data1, list) or not isinstance(data2, list):
            raise ValueError("Both JSON files should contain a list of objects.")

        merged_data = data1 + data2

        # Write the merged data to the output file
        with open(output_file_path, "wb") as out:
            out.write(orjson.dumps(merged_data, option=orjson.OPT_INDENT_2))

        print(f"Successfully merged files into {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
file1 = "/home/work/naipl-framework/results/gpt4o/clean_gpt4o_dev.json"
file2 = "/home/work/naipl-framework/results/gpt4o/clean_gpt4o_test.json"
output_file = "gpt4o(dev_test).json"
merge_json_files(file1, file2, output_file)
