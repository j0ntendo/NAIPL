import orjson

# Define the file path
file_path = "/home/work/naipl-framework/gpt4o(dev_test).json"

# Read the JSON file
with open(file_path, "r") as file:
    # Load the JSON data
    data = orjson.loads(file.read())

# Iterate over each item in the data list and remove the 'index' key
for item in data:
    if "index" in item:
        del item["index"]

# Write the modified data back to the file
with open(file_path, "w") as file:
    # Convert the Python object back to JSON formatted string and write to file
    file.write(orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8"))
