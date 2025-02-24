# Minimal Open Reasoner

A minimal framework for iterative code generation and arithmetic reasoning using a common pipeline of policy initialization, reward design, search, and learning.

## Overview

This repository contains an experimental “minimal open reasoner” showcasing how a single system can tackle:
1. **Code Generation** (with outcome-based rewards).
2. **Mathematical Reasoning** (with partial-step rewards).

By integrating small tasks (like FizzBuzz or two-digit arithmetic) and iterating over policy, search, and learning steps, the framework demonstrates how newly discovered correct solutions (or partially correct steps) are collected and used to refine the model over time.

### Key Components

1. **Policy Initialization**  
   A “mini” policy model (referred to as “gpt-4o-mini”) serves as the initial checkpoint. It can be any transform-based language model that supports text generation.

2. **Reward Design**  
   - **Outcome-Based** (for code): A binary reward (1 for success, 0 for failure) based on passing all test cases.  
   - **Partial-Step** (for math): Fractional rewards for each correct sub-step, plus a bonus if the final solution is correct.

3. **Search**  
   - **Best-of-N Sampling** (code): Generate multiple solutions for each prompt and pick the one with the highest reward (pass/fail test results).  
   - **Sequential Revision** (math): Generate an initial chain-of-thought; if incorrect, request a one-time “revision” to improve intermediate steps.

4. **Learning**  
   - **Behavior Cloning (BC)**: Gather high-reward (correct) solutions and fine-tune the policy to replicate them.  
   - **Reinforcement Learning** (optional): Use partial-step trajectories for policy-gradient updates (e.g., PPO, DPO).

5. **Aggregator**  
   An aggregator (or “reducer”) collects newly generated solutions—retaining correct or partially correct outputs, removing duplicates, and structuring data for subsequent fine-tuning. Over multiple iterations, it builds a growing dataset reflecting incremental improvements.

## Repository Structure

```
.
├── README.md               <- This file
├── code_tasks/             <- Example code prompts and tests
│   ├── fizzbuzz.json
│   ├── reverse_string.json
│   └── ...
├── math_tasks/             <- Example arithmetic tasks (with step-by-step references)
│   ├── single_digit.json
│   ├── double_digit.json
│   └── ...
├── aggregator/             <- Scripts for merging newly generated data into the dataset
│   └── aggregator.py
├── policy/                 <- Checkpoints and fine-tuning scripts for "gpt-4o-mini"
│   ├── policy_init.bin
│   ├── policy_finetune.py
│   └── ...
├── search/                 <- Scripts for best-of-N or revision-based generation
│   ├── code_search.py
│   ├── math_search.py
│   └── ...
├── rewards/                <- Reward functions for code and math
│   ├── code_evaluator.py
│   └── math_evaluator.py
├── examples/               <- Sample outputs and logs
└── ...
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/minimal-open-reasoner.git
cd minimal-open-reasoner
```

### 2. Install Dependencies

A typical environment might include Python 3.8+, PyTorch, Transformers, and any libraries needed to evaluate code snippets (e.g., a Python execution sandbox). Use:

```bash
pip install -r requirements.txt
```

*(Note: Requirements file not shown here, tailor it to your environment.)*

### 3. Run Code Generation Experiments

1. Place your code prompts or tasks in `code_tasks/`.  
2. Configure hyperparameters or search parameters in `search/code_search.py`.  
3. Execute:

   ```bash
   python search/code_search.py --model-checkpoint policy/policy_init.bin
   ```

4. The script will:
   - Sample multiple solutions per prompt (best-of-N).  
   - Evaluate them using the binary pass/fail tests in `rewards/code_evaluator.py`.  
   - Merge successful solutions into the aggregator for subsequent fine-tuning.

5. Fine-tune the policy:

   ```bash
   python policy/policy_finetune.py --aggregated-data aggregator/output_code.json
   ```

### 4. Run Math Reasoning Experiments

1. Place arithmetic problems (with step-level references) in `math_tasks/`.  
2. Configure partial-step reward logic in `rewards/math_evaluator.py`.  
3. Run:

   ```bash
   python search/math_search.py --model-checkpoint policy/policy_init.bin
   ```

4. The script will:
   - Generate chain-of-thought solutions.  
   - Optionally trigger one revision pass if the final answer is incorrect.  
   - Score partial correctness for intermediate steps.  
   - Update the aggregator with correct or near-correct solutions.

5. Fine-tune the policy (via behavior cloning or RL-like updates):

   ```bash
   python policy/policy_finetune.py --aggregated-data aggregator/output_math.json
   ```

## Experiments and Results

We illustrate two small sets of tasks:
- **Code**: (1) FizzBuzz, (2) String Reversal, (3) Summation of array elements.  
- **Math**: (1) Single-digit addition/subtraction, (2) Double-digit word problems.

In a minimal test, the pipeline often finds correct solutions quickly (sometimes on the first try), so improvement metrics (like pass rates over iterations) can appear flat. However, the pipeline’s core logic—searching for solutions, assigning rewards, storing new data, and fine-tuning—functions identically for more complex tasks, where iterative improvements become more evident.

## Future Plans

- **Scaling Task Complexity**: Move beyond toy FizzBuzz or single-digit arithmetic to more complex code (multi-module, concurrency) and mathematics (multi-step proofs).  
- **Advanced Search**: Incorporate Monte Carlo Tree Search (MCTS) or deeper iterative revision.  
- **Reinforcement Learning**: Add policy-gradient methods (PPO, DPO) for more nuanced optimization.  
- **Enhanced Aggregator**: Integrate active learning, preference scoring, or domain-specific constraints to guide solution selection.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have ideas for:
- New tasks or datasets
- Improved reward functions
- Advanced search algorithms
- Refined aggregator logic

## License

This project is licensed under the [MIT License](LICENSE). Please see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration inquiries, create a GitHub issue or reach out to the maintainers directly via email.

---

**Citation**  
If you use or reference this minimal open reasoner for your work, please cite our accompanying report in your publications:

```
@misc{agentlab_minimal_open_reasoner,
  title   = {Research Report: A Minimal Open Reasoner Implementation for Code and Math Tasks},
  author  = {Agent Laboratory},
  year    = {2023},
  note    = {Demonstrates code and math integration using policy, reward, search, and learning.}
}
```

Enjoy exploring the minimal open reasoner framework!