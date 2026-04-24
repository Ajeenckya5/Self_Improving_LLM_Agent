"""
Unified LLM client supporting OpenAI, xAI, Anthropic, Ollama, and a mock backend.

Backend selection priority:
  1. MOCK_LLM=1 env var  → deterministic mock (for CI)
  2. OLLAMA_BASE_URL set → local Ollama
  3. ANTHROPIC_API_KEY   → Anthropic Claude
  4. XAI_API_KEY         → xAI Grok
  5. OPENAI_API_KEY      → OpenAI GPT
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logger import get_logger

logger = get_logger(__name__)

_LOG_PATH: Optional[Path] = None


def _init_log_path(config: Dict[str, Any]) -> Path:
    global _LOG_PATH
    if _LOG_PATH is None:
        p = Path(config.get("logging", {}).get("llm_calls_log", "results/llm_calls.jsonl"))
        p.parent.mkdir(parents=True, exist_ok=True)
        _LOG_PATH = p
    return _LOG_PATH


class LLMClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._backend = self._detect_backend()
        logger.info("LLMClient using backend: %s", self._backend)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Send a chat request and return the assistant's text response."""
        start = time.time()
        try:
            if self._backend == "mock":
                response = self._mock_response(messages)
            elif self._backend == "ollama":
                response = self._ollama_chat(messages, model, temperature, max_tokens)
            elif self._backend == "anthropic":
                response = self._anthropic_chat(messages, model, temperature, max_tokens)
            elif self._backend == "groq":
                response = self._groq_chat(messages, model, temperature, max_tokens)
            elif self._backend == "xai":
                response = self._xai_chat(messages, model, temperature, max_tokens)
            else:
                response = self._openai_chat(messages, model, temperature, max_tokens)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            response = f"Error: LLM call failed — {exc}"

        elapsed = time.time() - start
        self._log_call(messages, response, model, elapsed)
        return response

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _openai_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        import openai

        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content or ""

    def _anthropic_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        # Separate system prompt from conversation
        system_prompt = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
            else:
                filtered.append(m)

        # Map model names: if it still says gpt-4, pick a sensible claude model
        claude_model = model
        if "gpt" in model.lower():
            claude_model = "claude-sonnet-4-6"

        for attempt in range(6):
            try:
                resp = client.messages.create(
                    model=claude_model,
                    system=system_prompt or "You are a helpful AI agent.",
                    messages=filtered,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.content[0].text if resp.content else ""
            except anthropic.RateLimitError:
                wait = min(15 * (2 ** attempt), 120)
                logger.warning("Anthropic rate limit (attempt %d/6); retrying in %ds", attempt + 1, wait)
                time.sleep(wait)
        raise RuntimeError("Anthropic rate limit exceeded after 6 retries")

    def _groq_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        import openai

        client = openai.OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        for attempt in range(6):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return completion.choices[0].message.content or ""
            except openai.RateLimitError as exc:
                wait = min(10 * (2 ** attempt), 120)
                logger.warning("Groq rate limit (attempt %d/6); retrying in %ds", attempt + 1, wait)
                time.sleep(wait)
        raise RuntimeError("Groq rate limit exceeded after 6 retries")

    def _xai_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        import openai

        client = openai.OpenAI(
            api_key=os.environ["XAI_API_KEY"],
            base_url=os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1"),
            timeout=float(os.environ.get("XAI_TIMEOUT", "3600")),
        )
        for attempt in range(6):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return completion.choices[0].message.content or ""
            except openai.RateLimitError:
                wait = min(10 * (2 ** attempt), 120)
                logger.warning("xAI rate limit (attempt %d/6); retrying in %ds", attempt + 1, wait)
                time.sleep(wait)
        raise RuntimeError("xAI rate limit exceeded after 6 retries")

    def _ollama_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        import requests

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    def _mock_response(self, messages: List[Dict[str, str]]) -> str:
        """Deterministic mock for CI testing — returns a plausible ReAct response."""
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        # Strategy generation mock
        if (
            "evaluation judge" not in last_user.lower()
            and ("corrective strategy" in last_user.lower() or "generate a corrective" in last_user.lower())
        ):
            return json.dumps({
                "strategy_text": "The agent repeated an action after the observation showed no new state change. For this task type, inspect the latest observation, choose a different tool or next dependency, and verify the target state before calling finish(result).",
                "decision_rule": "If an action has already returned the same observation twice, never repeat it; switch to a state-inspection or dependency-creation step.",
                "tags": ["error_handling", "precondition_check", "alternative_approach"],
            })

        # Failure analysis mock
        if "failure analysis judge" in last_user.lower() or (
            "failure" in last_user.lower() and "analyze" in last_user.lower()
        ):
            return json.dumps({
                "failure_type": "repeated_action",
                "failed_steps": [3, 4, 5],
                "pattern_summary": "The agent repeated the same bash command without checking its output.",
                "confidence": 0.8,
            })

        # Evaluation judge mock
        if "evaluation judge" in last_user.lower() and "overall_score" in last_user.lower():
            return json.dumps({
                "failure_type_correct": True,
                "failed_steps_overlap": 1.0,
                "analysis_grounding_score": 4,
                "strategy_specificity_score": 4,
                "strategy_actionability_score": 4,
                "retrieval_tags_score": 4,
                "overall_score": 4,
                "rationale": "The candidate is grounded in the repeated action pattern and gives a concrete prevention rule.",
            })

        # Planning mock
        if "numbered list" in last_user.lower() or "generate a plan" in last_user.lower():
            return "1. Identify the target\n2. Perform the action\n3. Verify the result\n4. Report completion"

        # Default ReAct mock
        if "observation:" in last_user.lower():
            return (
                "Thought: The previous action completed. I should verify the result and finish.\n"
                "Action: finish(Task completed successfully)"
            )

        return (
            "Thought: I need to analyze the task and take the first step.\n"
            "Action: bash(echo 'Starting task execution')"
        )

    # ------------------------------------------------------------------

    def _detect_backend(self) -> str:
        if os.environ.get("MOCK_LLM") == "1":
            return "mock"
        # Explicit backend from config takes priority over env-var auto-detection
        explicit = self.config.get("model", {}).get("backend", "")
        if explicit in ("anthropic", "groq", "ollama", "openai", "xai"):
            if explicit == "xai" and not os.environ.get("XAI_API_KEY"):
                logger.warning("xAI backend selected but XAI_API_KEY is not set. Using mock LLM.")
                return "mock"
            return explicit
        # Env-var fallback (Ollama last — check it's reachable before selecting)
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("GROQ_API_KEY"):
            return "groq"
        if os.environ.get("XAI_API_KEY"):
            return "xai"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("OLLAMA_BASE_URL") and self._ollama_reachable():
            return "ollama"
        logger.warning("No API key found. Using mock LLM.")
        return "mock"

    def _ollama_reachable(self) -> bool:
        try:
            import requests as _req
            base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            _req.get(f"{base}/api/tags", timeout=2)
            return True
        except Exception:
            return False

    def _log_call(
        self,
        messages: List[Dict[str, str]],
        response: str,
        model: str,
        elapsed: float,
    ) -> None:
        try:
            log_path = _init_log_path(self.config)
            entry = {
                "model": model,
                "elapsed_s": round(elapsed, 3),
                "messages": messages,
                "response": response,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.debug("LLM call logging failed: %s", exc)
