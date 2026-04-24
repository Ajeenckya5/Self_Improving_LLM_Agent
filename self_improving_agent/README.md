# Self-Improving LLM Agent for Long-Horizon Tasks

This project implements an experimental pipeline that builds and evaluates a **Self-Improving LLM Agent** designed for long-horizon sequential tasks. The agent learns from its own failures without any model retraining: it analyzes execution traces, generates corrective strategies, stores them in a persistent memory, and retrieves relevant strategies to guide future task execution. The system is inspired by the SELF-REFINE framework (Madaan et al., 2023), which demonstrates that iterative self-feedback and refinement can improve LLM task performance by ~20% on average across diverse benchmarks.

The key hypothesis is that **failure memory compounds**: as the agent encounters and analyzes more failures, the quality of retrieved strategies improves, leading to monotonically increasing cumulative success rates over the task sequence — a property that flat-line ReAct and Plan-and-Act baselines cannot exhibit. We evaluate this hypothesis on two benchmark families (AgentBench OS tasks and WebArena-style web navigation tasks) across four horizon lengths (H ∈ {5, 10, 15, 20} steps), and we validate each component's contribution via an ablation study.

---

## Project Structure

```
self_improving_agent/
├── agent/             # ReAct, Plan-and-Act, and Strategy-guided agents
├── analysis/          # Failure analyzer and strategy generator
├── memory/            # SQLite strategy store and semantic retriever
├── environments/      # OS task simulator and web task stub
├── evaluation/        # Metrics, plotting, and the main experiment loop
├── experiments/       # Runnable experiment scripts
├── prompts/           # Prompt templates
├── tests/             # pytest test suite (no API keys required)
├── results/           # Auto-generated CSVs and plots (gitignored)
├── config.yaml        # All hyperparameters
└── requirements.txt
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd SelfImprovingLLMAgent

# 2. Create a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r self_improving_agent/requirements.txt
```

### API Key Configuration

Set at least one of the following environment variables:

```bash
# Option A: xAI Grok-4 (default project profile)
export XAI_API_KEY="xai-..."

# Option B: OpenAI
export OPENAI_API_KEY="sk-..."

# Option C: Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# Option D: Local Ollama (zero cost)
export OLLAMA_BASE_URL="http://localhost:11434"
# Then pull a model: ollama pull llama3

# Option E: Mock LLM (for CI / testing only — no real task solving)
export MOCK_LLM=1
```

---

## Running Experiments

### AgentBench OS Experiment (Recommended starting point)

```bash
# Full run (100 tasks × 3 conditions = ~300 LLM-task executions)
python -m self_improving_agent.experiments.run_agentbench

# Quick sanity check (5 tasks per condition)
python -m self_improving_agent.experiments.run_agentbench --dry-run

# Custom horizons and task count
python -m self_improving_agent.experiments.run_agentbench --horizons 5 10 --n-tasks 10
```

### WebArena Experiment

```bash
python -m self_improving_agent.experiments.run_webarena
python -m self_improving_agent.experiments.run_webarena --dry-run
```

### Ablation Study

```bash
# Compare: Full System vs. No Memory vs. No Analysis vs. Plain ReAct
python -m self_improving_agent.experiments.ablation

# Custom horizon and task count
python -m self_improving_agent.experiments.ablation --horizon 15 --n-tasks 20 --dry-run
```

### Member 2 Failure/Strategy Prompt Evaluation

```bash
# Uses Grok-4.20 reasoning when XAI_API_KEY is set
python -m self_improving_agent.experiments.member2_eval --profile xai

# CI-safe smoke test with deterministic mock responses
python -m self_improving_agent.experiments.member2_eval --mock

# Prompt ablation study across grounded, weakened, generic, and heuristic variants
python -m self_improving_agent.experiments.member2_ablation --profile xai
python -m self_improving_agent.experiments.member2_ablation --mock
```

Member 2 outputs are written to `results/member2_eval_results.csv` and
`results/member2_ablation/member2_ablation_summary.csv`.

---

## Running Tests

Tests run with the mock LLM — no API keys needed:

```bash
cd SelfImprovingLLMAgent
python -m pytest self_improving_agent/tests/ -v
```

---

## Interpreting Results

All results are saved under `results/`:

| File | Description |
|------|-------------|
| `agentbench_results.csv` | Per-task results for all 3 conditions |
| `webarena_results.csv` | Per-task results for WebArena |
| `ablation_results.csv` | Per-task results for ablation |
| `summary_table.csv` | Mean ± std success rate per condition |
| `success_vs_horizon.png` | Line chart: success rate vs. horizon length |
| `failure_mode_dist.png` | Grouped bar: failure type breakdown per method |
| `cumulative_success.png` | Learning curve: cumulative success over time |
| `ablation.png` | Bar chart: component contribution |
| `llm_calls.jsonl` | Full log of every LLM prompt + response |

### What to look for

1. **`success_vs_horizon.png`** — The self-improving method's advantage should widen at longer horizons (H=20). Baselines should degrade more steeply.
2. **`cumulative_success.png`** — Only the self-improving curve should trend upward monotonically; baselines should be flat or declining.
3. **`failure_mode_dist.png`** — The self-improving method should have fewer `repeated_action` and `circular_loop` failures than baselines, since strategies specifically target these.
4. **`ablation.png`** — All four bars should differ, confirming each component contributes.

---

## Component Descriptions

### `agent/base_agent.py` — ReAct Agent
Implements the Thought → Action → Observation loop. Records every step as a structured trace entry and returns `(success, AgentTrace)`.

### `agent/plan_act_agent.py` — Plan-and-Act Agent
Generates a numbered plan before execution, then executes each step with a ReAct sub-loop. Used as the second baseline.

### `agent/strategy_agent.py` — Strategy-Guided Agent (our method)
Extends the ReAct agent. Before starting, retrieves relevant past strategies from memory and injects them into the system prompt.

### `analysis/failure_analyzer.py` — Failure Pattern Detection
Analyzes execution traces to detect: `repeated_action`, `circular_loop`, `context_truncation`, `incorrect_reasoning`, `tool_misuse`. No LLM call required — pure heuristics.

### `analysis/llm_failure_analyzer.py` — LLM Failure Analysis
Uses the prompt in `prompts/failure_analysis.txt` to classify root-cause failure type, failed steps, and grounded summary. The default `xai` profile runs this with `grok-4.20-reasoning`; the heuristic analyzer remains available as a fallback and ablation condition.

### `analysis/strategy_generator.py` — Strategy Generation
Makes one LLM call per failure to generate a 2-4 sentence corrective strategy and 3-5 retrieval tags.

### `evaluation/llm_judge.py` — Prompt Evaluation Harness
Scores analyzer and strategy outputs for label correctness, failed-step overlap, trace grounding, actionability, and retrieval tag quality. This is the GPT-4-style judge harness requested for Member 2, configured here to use Grok-4 through the `xai` model profile.

### `memory/strategy_memory.py` — Persistent Storage
SQLite-backed store with `store()`, `retrieve()`, and `update_outcome()` methods. Embeddings stored as numpy arrays.

### `memory/retriever.py` — Semantic Retrieval
Uses `sentence-transformers` for embedding; cosine similarity over all stored strategy embeddings.

### `environments/os_env.py` — OS Environment
Sandboxed subprocess-based simulator for AgentBench-style OS tasks at horizons 5/10/15/20.

### `environments/web_env.py` — Web Environment
Mock HTML page environment for WebArena-style sequential navigation tasks.

### `evaluation/evaluate.py` — Experiment Loop
`run_experiment()` orchestrates: task loop, strategy retrieval, agent execution, failure analysis, memory update, result collection.

### `evaluation/metrics.py` — Metrics and Plots
`task_success_rate`, `failure_mode_distribution`, `success_vs_horizon`, `cumulative_success_curve`, `repeated_failure_rate`, plus matplotlib plot functions.

---

## Configuration

Edit `self_improving_agent/config.yaml` to change models, memory parameters, or evaluation settings:

```yaml
active_profile: "xai"

model:
  primary: "grok-4.20-reasoning"
  analyzer: "grok-4.20-reasoning"
  judge: "grok-4.20-reasoning"
  embedding: "all-MiniLM-L6-v2"
  backend: "xai"

agent:
  max_steps: 25
  temperature: 0.7
  max_tokens: 1000

memory:
  top_k: 3
  similarity_threshold: 0.65

analysis:
  failure_analyzer: "llm"
  failure_prompt: "failure_analysis.txt"
  strategy_prompt: "strategy_gen.txt"
  judge_prompt: "evaluation_judge.txt"
```

---

## Citations

```bibtex
@inproceedings{madaan2023selfrefine,
  title={{SELF-REFINE}: Iterative Refinement with Self-Feedback},
  author={Madaan, Aman and Tandon, Niket and Gupta, Prakhar and Hallinan, Skyler
          and Gao, Luyu and Wiegreffe, Sarah and Alon, Uri and Dziri, Nouha
          and Prabhumoye, Shrimai and Yang, Yiming and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

@article{liu2023agentbench,
  title={{AgentBench}: Evaluating LLMs as Agents},
  author={Liu, Xiao and Yu, Hao and Zhang, Hanchen and Xu, Yifan and Lei, Xuanyu
          and Lai, Hanyu and Gu, Yu and Ding, Hangliang and Men, Kaiwen
          and Yang, Kejuan and others},
  journal={arXiv preprint arXiv:2308.03688},
  year={2023}
}

@inproceedings{zhou2024webarena,
  title={{WebArena}: A Realistic Web Environment for Building Autonomous Agents},
  author={Zhou, Shuyan and Xu, Frank F and Zhu, Hao and Zhou, Xuhui
          and Lo, Robert and Sridhar, Abishek and Cheng, Xianyi and
          Bisk, Yonatan and Fried, Daniel and Alon, Uri and others},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
