"""Configuration for the agent evaluation framework."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
ENV_ROOT = PROJECT_ROOT / "sandbox"
TRACES_DIR = PROJECT_ROOT / "traces"
STRATEGY_DB_PATH = PROJECT_ROOT / "strategy_memory.db"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
RESULTS_DIR = PROJECT_ROOT / "results"

# Agent settings
MAX_STEPS = 15
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Embedding model (local, no API key needed)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Experiment settings
NUM_TASK_ATTEMPTS = 3  # Retries before declaring failure
EXPERIMENT_SEED = 42
