from datasets import load_dataset

# Step 1: Load a Hugging Face dataset (AG News for example)
dataset = load_dataset("ag_news", split="train")

# Step 2: Convert text to lowercase (simple data preparation example)
dataset = dataset.map(lambda x: {"text": x["text"].lower(), "label": x["label"]})

# Step 3: Print first example to verify
print(dataset[0])