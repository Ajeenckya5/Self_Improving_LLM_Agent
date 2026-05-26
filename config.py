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

# ── Distillation settings ──────────────────────────────────────────────────
# Teacher: Grok 4 via xAI API (generates training data from failure traces)
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
GROK_TEACHER_MODEL = "grok-4"
XAI_BASE_URL = "https://api.x.ai/v1"

# Student: QLoRA fine-tuned LLaMA-3.2-1B-Instruct
STUDENT_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
STUDENT_ADAPTER_PATH = PROJECT_ROOT / "models" / "failure_analyzer_lora"
DISTILL_DATA_PATH = PROJECT_ROOT / "data" / "distill_train.jsonl"

# Set USE_STUDENT_ANALYZER=1 to route failure analysis through the local
# QLoRA student instead of the OpenAI/Grok API.
USE_STUDENT_ANALYZER = os.getenv("USE_STUDENT_ANALYZER", "0") == "1"
