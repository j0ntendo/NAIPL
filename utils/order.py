import json

# Read the JSON file
with open('evaluation_results.json', 'r') as file:
    data = json.load(file)

# Sort the data by index
sorted_data = sorted(data, key=lambda x: x['index'])

# Write the sorted data to a new JSON file
with open('sorted_evaluation_results.json', 'w') as file:
    json.dump(sorted_data, file, indent=4)

print("Data sorted by index and saved to 'sorted_evaluation_results.json'")