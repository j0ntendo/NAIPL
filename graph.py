import os
import orjson
import matplotlib.pyplot as plt
from collections import defaultdict


# Load the JSON data
def load_json(file_path):
    with open(file_path, "rb") as file:
        data = orjson.loads(file.read())
    return data


# Modified functions to include numbers on bars and save graphs in a folder
def plot_bar_charts_by_label(data, output_folder):
    grouped_data = defaultdict(list)
    for item in data:
        grouped_data[item["label"]].append(item)

    for label, items in grouped_data.items():
        correct = sum(1 for item in items if item["is_correct"])
        incorrect = sum(1 for item in items if not item["is_correct"])
        model_name = items[0]["model_name"] if items else "Unknown Model"

        # Create bar chart for each label
        plt.figure(figsize=(6, 4))
        bars = plt.bar(
            ["Correct", "Incorrect"],
            [correct, incorrect],
            color=["#4CAF50", "#F44336"],
        )
        plt.title(f"Label: {label}\nModel: {model_name}", fontsize=10)
        plt.xlabel("Outcome")
        plt.ylabel("Count")

        # Add text annotations for the counts
        for bar, count in zip(bars, [correct, incorrect]):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(count),
                ha="center",
                fontsize=10,
            )

        plt.tight_layout()

        # Save the chart in the output folder
        filename = f"bar_chart_{label.replace(' ', '_').replace('/', '_')}.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved bar chart for {label} (Model: {model_name}) as {filepath}")


def plot_line_graph_comparison(data, output_folder):
    grouped_data = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    for item in data:
        label = item["label"]
        if item["is_correct"]:
            grouped_data[label]["correct"] += 1
        else:
            grouped_data[label]["incorrect"] += 1

    labels = list(grouped_data.keys())
    correct_counts = [grouped_data[label]["correct"] for label in labels]
    incorrect_counts = [grouped_data[label]["incorrect"] for label in labels]

    # Create line graph
    plt.figure(figsize=(10, 6))
    plt.plot(
        labels,
        correct_counts,
        marker="o",
        label="Correct",
        linestyle="--",
        color="#77c298",
    )
    plt.plot(
        labels,
        incorrect_counts,
        marker="x",
        label="Incorrect",
        linestyle="--",
        color="#e84d60",
    )
    plt.title(
        "Comparison of Correct and Incorrect Responses Across Labels", fontsize=12
    )
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    # Save the chart in the output folder
    filepath = os.path.join(output_folder, "line_graph_comparison.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved line graph as {filepath}")


# Main function to generate all graphs
def generate_graphs(json_file_path):
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    output_folder = f"{base_name}_graphs"
    os.makedirs(output_folder, exist_ok=True)

    data = load_json(json_file_path)
    plot_bar_charts_by_label(data, output_folder)
    plot_line_graph_comparison(data, output_folder)

    print(f"All graphs saved in folder: {output_folder}")


# Example usage
generate_graphs(
    "/home/work/naipl-framework/gpt4o(dev_test).json"
)  # Replace with your JSON file path
