# Import required libraries
from openai import OpenAI
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Setup OpenAI client
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
client = OpenAI()

# Load processed MATH dataset
with open("processed_math_data.jsonl", "r") as f:
    processed_data = [json.loads(line) for line in f.readlines()[:50]]  # Use 50 samples for balance

# Enhanced verification logic with regex patterns
import re
class Verifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
        self.invalid_pattern = re.compile(r"contradicts|wrong|incorrect|error", re.IGNORECASE)
        
    def validate_step(self, current_step, previous_steps):
        # Check for explicit invalidation keywords
        if re.search(self.invalid_pattern, current_step):
            return False
        # Check numerical consistency using simple pattern matching
        prev_numbers = re.findall(r"\d+\.?\d*", " ".join(previous_steps))
        current_numbers = re.findall(r"\d+\.?\d*", current_step)
        if current_numbers and not any(num in prev_numbers for num in current_numbers):
            return False
        return True

# Modified Reward Model with improved input handling
class RewardModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=2)
        
    def predict(self, solution):
        inputs = self.tokenizer(solution, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        return torch.sigmoid(outputs.logits[0][1]).item()  # Use second class as validity score

# Updated Reasoner with answer extraction
class Reasoner:
    def __init__(self):
        self.verifier = Verifier()
        self.reward_model = RewardModel()
        
    def generate_cot(self, problem):
        messages = [{
            "role": "system",
            "content": "Solve the problem step-by-step and put your final answer within \\boxed{}."
        }, {
            "role": "user",
            "content": f"Solve: {problem}"
        }]
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.3,
            max_tokens=512
        )
        return response.choices[0].message.content

# Experiment 1: Enhanced CoT with answer extraction
print("Experiment 1: Enhanced CoT with system prompt...")
vanilla_predictions = []
for example in processed_data[:20]:  # Increased sample size
    cot = Reasoner().generate_cot(example["problem"])
    # Check for boxed answer with number
    has_boxed = bool(re.search(r"\\boxed\{.*?\d+.*?\}", cot))
    vanilla_predictions.append(1 if has_boxed else 0)

true_labels = [int(ex["valid_solution"]) for ex in processed_data[:20]]
vanilla_acc = accuracy_score(true_labels, vanilla_predictions)
print(f"Enhanced CoT Accuracy: {vanilla_acc:.2f}")

# Experiment 2: PARC with numerical consistency checks
print("\nExperiment 2: PARC with numerical consistency...")
verified_predictions = []
verifier = Verifier()

for example in processed_data[:20]:
    steps = example["steps"]
    valid_steps = []
    for i, step in enumerate(steps):
        if verifier.validate_step(step, steps[:i]):
            valid_steps.append(step)
    # Consider solution valid if >80% steps verified
    verified_predictions.append(1 if len(valid_steps)/len(steps) > 0.8 else 0)

parc_acc = accuracy_score(true_labels, verified_predictions)
print(f"PARC-Verified Accuracy: {parc_acc:.2f}")

# Generate updated visualizations
plt.figure(figsize=(10,6))
bars = plt.bar(['Enhanced CoT', 'PARC Verified'], [vanilla_acc, parc_acc], 
               color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
plt.title('Step-Improved Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Accuracy', fontstyle='italic', fontsize=12)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2%}', ha='center', va='bottom')
plt.savefig('Figure_1.png')

plt.figure(figsize=(10,6))
check_counts = [sum(vanilla_predictions), sum(verified_predictions)]
explode = (0.1, 0)
plt.pie(check_counts, labels=['CoT Solutions', 'Verified Solutions'], 
        colors=['#FFE66D', '#88D8B0'], autopct='%1.1f%%',
        startangle=90, shadow=True)
plt.title('Solution Verification Distribution', fontsize=14, fontweight='bold', pad=20)
plt.savefig('Figure_2.png')