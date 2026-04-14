# Project

This is a Python research framework for a **Self-Improving LLM Agent** that learns from its own failures on long-horizon tasks — no model retraining, just failure analysis, strategy generation, and persistent memory retrieval. Evaluated on AgentBench OS tasks and WebArena-style web navigation benchmarks.

---

# Development Workflow

**Always use `python` / `pip` inside the `.venv` virtual environment.**

```bash
# 1. Activate the virtual environment
source .venv/bin/activate

# 2. Install / sync dependencies
pip install -r self_improving_agent/requirements.txt

# 3. Run tests (no API key needed — uses mock LLM)
python -m pytest self_improving_agent/tests/ -v

# 4. Run a single test by keyword
python -m pytest self_improving_agent/tests/ -v -k "test_name"

# 5. Quick sanity check (dry-run, 3 tasks only)
python main.py run --dry-run

# 6. Full controlled experiment
python main.py run --attempts 1 --output results/

# 7. AgentBench experiment
python main.py agentbench --dry-run

# 8. Ablation study
python main.py ablation --dry-run
```

Run tests after every change. Fix all failures before moving on.

---

# Code Style

- Python 3.10+ — use `match/case`, `|` union types, and `from __future__ import annotations` where needed
- Use `from pathlib import Path` for all file paths — never raw string concatenation
- Prefer `dataclasses` or `pydantic` models over plain dicts for structured data
- Use `async/await` only when the call site is already async; don't introduce async for a single call
- No bare `except:`; always catch specific exceptions
- All prompt templates live in `self_improving_agent/prompts/` — never inline multi-line prompts in Python files
- Colocate tests next to the module they test inside `self_improving_agent/tests/`

---

# Architecture

```
self_improving_agent/
├── agent/              # ReAct, Plan-and-Act, and Strategy-guided agents
├── analysis/           # Failure pattern detection (heuristic) + strategy generator (LLM)
├── memory/             # SQLite strategy store + sentence-transformers retriever
├── environments/       # OS task simulator (subprocess) and web task mock
├── evaluation/         # Metrics, plotting, and the main experiment loop
├── experiments/        # Runnable experiment entry points
├── prompts/            # All prompt templates (.txt)
├── tasks/              # Task definitions (filesystem, database, base)
├── utils/              # LLM client, logger, trace logger
├── tests/              # pytest suite (conftest + per-module tests)
├── results/            # Auto-generated CSVs, plots, logs (do not commit)
└── config.yaml         # All hyperparameters (models, memory, evaluation)
main.py                 # Unified CLI entry point
```

**Key data flows:**
- `evaluation/evaluate.py:run_experiment()` is the orchestration core — task loop → strategy retrieval → agent execution → failure analysis → memory update
- `memory/strategy_memory.py` is the SQLite-backed store; `memory/retriever.py` does cosine similarity over sentence-transformer embeddings
- `analysis/failure_analyzer.py` is pure heuristics (no LLM); `analysis/strategy_generator.py` makes one LLM call per failure
- `utils/llm_client.py` is the single point of contact for all LLM calls — do not call the OpenAI/Anthropic SDK directly elsewhere

---

# Configuration

All hyperparameters are in `self_improving_agent/config.yaml`. Do not hardcode model names, `top_k`, thresholds, or horizon lists anywhere else — read from config.

Key fields:

| Key | Default | Notes |
|-----|---------|-------|
| `model.primary` | `gpt-4` | Main agent model |
| `model.analyzer` | `gpt-4` | Failure analysis + strategy gen |
| `model.embedding` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `memory.top_k` | `3` | Retrieved strategies per task |
| `memory.similarity_threshold` | `0.65` | Min cosine sim for retrieval |
| `evaluation.horizons` | `[5,10,15,20]` | Task step horizons |

---

# Gotchas

- **Never call OpenAI/Anthropic SDKs directly** — always go through `utils/llm_client.py`
- `analysis/failure_analyzer.py` must stay **LLM-free** (heuristics only) — adding LLM calls here breaks the ablation study's "no analysis" condition
- The `results/` directory is gitignored; re-run experiments to regenerate plots and CSVs
- Tests use `MOCK_LLM=1` (set in `conftest.py`) — never require a real API key in the test suite
- `self_improving_agent/requirements.txt` has a typo: `requests>=2.31.0exit` — the trailing `exit` is harmless but do not propagate it to new lines
- Environment variables are loaded from `.env` at startup in `main.py` — copy `.env.example` to `.env` and fill in keys; do not commit `.env`
- Do not add new dependencies without updating `self_improving_agent/requirements.txt` and noting it in a PR

---

# Permissions (pre-allowed safe commands)

```
python -m pytest self_improving_agent/tests/ -v
python main.py run --dry-run
python main.py agentbench --dry-run
python main.py ablation --dry-run
git status
git diff
git log --oneline -20
pip install -r self_improving_agent/requirements.txt
```

---

# Rules (updated as we go)

- [ ] *(add rules here as the project grows)*
