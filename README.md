# Agent Evaluation Framework

Long-horizon task evaluation comparing **baseline** vs **strategy-enhanced** LLM agents. The system implements:

- **Long-horizon tasks** (file-system, database) with automatic verifiers
- **Plan → Act → Observe** loop (ReAct-style)
- **Trace logging** of (state, action, observation) trajectories
- **Failure analysis** (rule-based + LLM classification) and corrective strategy generation
- **Strategy memory** indexed by embeddings for retrieval on similar tasks
- **Experiments** with metrics and plots

## Setup

```bash
cd agent-evaluation
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
```

## Run Experiments

```bash
python main.py run
```

Options:
- `--max-steps N` – max steps per task (default 15)
- `--attempts N` – attempts per task per agent (default 2)
- `--output DIR` – where to save plots
- `--no-plot` – skip plot generation

## Architecture

```
tasks/           # Long-horizon tasks + verifiers
  - filesystem   # Multi-step file organize, nested structure, backup
  - database     # Create/alter tables, join/aggregate

environment/     # Controlled sandbox for tool execution

agent/           # Plan-Act-Observe agents
  - baseline     # No strategy memory
  - strategy_enhanced  # Injects retrieved strategies

tracing/         # Execution trace logger

failure_analysis/  # Rule-based + LLM failure classification

strategy_memory/   # Embedding-indexed strategy store (ChromaDB)

experiments/      # Runner + metrics + plots
```

## Flow

1. **Baseline** runs on each task; failures are analyzed.
2. **Failure analysis** applies rules (repeated actions, env errors, etc.) and LLM classification (planning_error, memory_limitation, instruction_misinterpretation, environmental_change, false_assumption).
3. **Corrective strategies** are generated and stored in strategy memory.
4. **Strategy-enhanced** agent retrieves similar strategies and injects them into the prompt.
5. **Metrics**: success rate, max steps solved, failure distribution; plots compare baseline vs strategy-enhanced.
