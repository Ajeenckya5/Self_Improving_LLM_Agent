# Self-Improving LLM Agent for Long-Horizon Tasks

A research implementation of a self-improving LLM agent that learns from failures without model retraining. The agent analyzes execution traces, generates corrective strategies, stores them in memory, and retrieves relevant strategies to guide future task execution.

## What This Project Does

The system tests the hypothesis that **failure memory compounds**: as the agent encounters and analyzes more failures, the quality of retrieved strategies improves, leading to better performance on sequential tasks. This approach is inspired by SELF-REFINE (Madaan et al., 2023).

**Key Features:**
- Three agent strategies: ReAct (baseline), Plan-and-Act (baseline), and Strategy-Guided (our method)
- Failure analysis via heuristics or LLM-based reasoning
- Semantic memory for storing and retrieving corrective strategies
- Evaluation on OS tasks (AgentBench) and web navigation tasks
- Ablation study to measure component contributions

## Quick Start

### 1. Install

```bash
# Clone and enter project directory
cd CS_639_Project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r self_improving_agent/requirements.txt
```

### 2. Configure API Keys

Set one of the following environment variables (or create a `.env` file in the project root):

```bash
# xAI Grok-4 (default)
export XAI_API_KEY="your-key-here"

# Or: Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# Or: OpenAI
export OPENAI_API_KEY="sk-..."

# Or: Local Ollama (free)
export OLLAMA_BASE_URL="http://localhost:11434"

# Or: Mock LLM for testing (no API key needed)
export MOCK_LLM=1
```

### 3. Run Experiments

**Option A: Controlled Filesystem Experiment** (Recommended for quick testing)
```bash
python main.py run --dry-run                    # 5-minute sanity check
python main.py run --attempts 1 -o results      # Full run
```

**Option B: AgentBench OS Tasks**
```bash
python main.py agentbench --dry-run             # Quick test
python main.py agentbench --n-tasks 100         # Full experiment
```

**Option C: Terminal-Bench**
```bash
python main.py terminalbench --n-tasks 10 --n-concurrent 2
```

**Option D: Ablation Study** (Compare components)
```bash
python main.py ablation --dry-run
python main.py ablation --horizon 15 --n-tasks 20
```

## Project Structure

```
CS_639_Project/
├── main.py                      # Unified CLI entry point
├── self_improving_agent/
│   ├── agent/                   # Three agent implementations
│   │   ├── base_agent.py       # ReAct agent
│   │   ├── plan_act_agent.py   # Plan-and-Act agent
│   │   └── strategy_agent.py   # Strategy-Guided agent (our method)
│   ├── analysis/                # Failure analysis & strategy generation
│   │   ├── failure_analyzer.py
│   │   ├── llm_failure_analyzer.py
│   │   └── strategy_generator.py
│   ├── memory/                  # SQLite strategy store
│   │   ├── strategy_memory.py
│   │   └── retriever.py
│   ├── environments/            # Task simulators
│   │   ├── os_env.py           # AgentBench-style OS tasks
│   │   ├── web_env.py          # WebArena-style web tasks
│   │   └── diverse_os_tasks.py
│   ├── evaluation/              # Metrics and experiment loop
│   │   ├── evaluate.py         # Main experiment orchestration
│   │   ├── metrics.py          # Success rate, failure modes, plots
│   │   └── llm_judge.py        # Prompt evaluation harness
│   ├── experiments/             # Runnable experiment scripts
│   │   ├── run_agentbench.py
│   │   ├── run_webarena.py
│   │   ├── run_terminal_bench.py
│   │   ├── ablation.py
│   │   ├── member2_eval.py
│   │   └── member2_ablation.py
│   ├── prompts/                 # LLM prompt templates
│   ├── config.yaml              # Hyperparameters & model profiles
│   ├── requirements.txt
│   ├── tests/                   # pytest suite (mock LLM, no API keys)
│   └── README.md
├── fangkai/                     # Optional: additional task implementations
└── results/                     # Auto-generated results & plots
```

## Key Experiments

### Experiment 1: Controlled Tasks
Tests on filesystem operations and database queries with increasing horizon lengths (8, 12, 16, 20 steps).

```bash
python main.py run --dry-run    # Test run with 3 tasks
python main.py run -o results   # Full run
```

**Output:** CSV results + plots showing success rate, failure modes, and cumulative learning curve.

### Experiment 2: AgentBench OS
Real OS command tasks from the AgentBench benchmark.

```bash
python main.py agentbench --n-tasks 100 --horizons 5 10 15 20
```

### Experiment 3: Ablation Study
Compares: Full System vs. No Memory vs. No Analysis vs. Plain ReAct.

```bash
python main.py ablation --horizon 15 --n-tasks 50
```

### Experiment 4: Prompt Evaluation (Member 2)
Evaluates failure/strategy prompts with an LLM judge.

```bash
python main.py member2-eval --profile xai
python main.py member2-ablation --profile xai
```

## Running Tests

No API keys needed — tests use mock LLM responses:

```bash
python -m pytest self_improving_agent/tests/ -v
```

## Configuration

Edit `self_improving_agent/config.yaml` to:
- Change active model profile: `xai` | `haiku` | `groq` | `ollama`
- Adjust agent hyperparameters (max_steps, temperature, max_tokens)
- Set memory parameters (top_k strategies, similarity threshold)
- Configure evaluation horizons and task counts
- Choose failure analyzer: `llm` or `heuristic`

Example:
```yaml
active_profile: "haiku"

model_profiles:
  haiku:
    primary: "claude-haiku-4-5-20251001"
    backend: "anthropic"
```

## Understanding Results

All results are saved in `results/`:

| File | Description |
|------|-------------|
| `controlled_results.csv` | Per-task results for all conditions |
| `controlled_summary.csv` | Mean ± std success rate |
| `controlled_success_vs_horizon.png` | Success rate vs. horizon length |
| `controlled_failure_dist.png` | Failure type breakdown |
| `controlled_cumulative.png` | Learning curve (key metric) |
| `recurrence_analysis.csv` | Repeated failures per condition |
| `llm_calls.jsonl` | Full log of all LLM calls |

**Key Insights to Look For:**

1. **Learning Curve** (`controlled_cumulative.png`): Only the self-improving method should show monotonically increasing cumulative success. Baselines should be flat or declining.

2. **Horizon Scaling** (`controlled_success_vs_horizon.png`): The self-improving method's advantage should widen at longer horizons (H=20).

3. **Failure Mode Reduction** (`controlled_failure_dist.png`): The self-improving method should have fewer `repeated_action` and `circular_loop` failures than baselines.

4. **Ablation Contribution** (`ablation_results.csv`): All components should meaningfully contribute.

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `XAI_API_KEY` | xAI Grok access | `xai-...` |
| `ANTHROPIC_API_KEY` | Claude access | `sk-ant-...` |
| `OPENAI_API_KEY` | GPT access | `sk-...` |
| `OLLAMA_BASE_URL` | Local LLM | `http://localhost:11434` |
| `MOCK_LLM` | Mock responses (testing) | `1` |

## References

```bibtex
@inproceedings{madaan2023selfrefine,
  title={{SELF-REFINE}: Iterative Refinement with Self-Feedback},
  author={Madaan, Aman and Tandon, Niket and Gupta, Prakhar and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

@article{liu2023agentbench,
  title={{AgentBench}: Evaluating LLMs as Agents},
  author={Liu, Xiao and Yu, Hao and Zhang, Hanchen and others},
  journal={arXiv preprint arXiv:2308.03688},
  year={2023}
}
```

## Troubleshooting

**No API key error?**
- Set `XAI_API_KEY`, `ANTHROPIC_API_KEY`, or another provider key
- Or set `MOCK_LLM=1` for testing

**Import errors?**
- Ensure you're in the virtual environment: `source .venv/bin/activate`
- Reinstall: `pip install -r self_improving_agent/requirements.txt`

**Tests failing?**
- Tests should work without API keys (mock LLM)
- Run: `python -m pytest self_improving_agent/tests/ -v`

**Results not generating?**
- Check `results/` directory exists
- Check `results/llm_calls.jsonl` for LLM errors
- For debugging: Add `--dry-run` to test with fewer tasks

## Next Steps

1. **Start with `python main.py run --dry-run`** — runs 5 tasks to verify setup
2. **Run full controlled experiment** — `python main.py run`
3. **Analyze results** — Check `results/controlled_cumulative.png` for learning curve
4. **Run ablation study** — `python main.py ablation` to measure component contributions
5. **Try other benchmarks** — `python main.py agentbench` or `python main.py terminalbench`
