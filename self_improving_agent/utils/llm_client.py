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
import re
import shlex
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

        # Default ReAct mock. For generated OS benchmark tasks, return a
        # deterministic one-shot action so mock runs test the agent/tool loop
        # instead of the old placeholder echo command.
        if "observation:" in last_user.lower():
            return (
                "Thought: The previous action completed. I should verify the result and finish.\n"
                "Action: finish(Task completed successfully)"
            )

        task_description = self._extract_mock_task_description(messages)
        task_action = self._mock_os_task_action(task_description)
        if task_action:
            return task_action

        return (
            "Thought: I need to analyze the task and take the first step.\n"
            "Action: bash(echo 'Starting task execution')"
        )

    def _extract_mock_task_description(self, messages: List[Dict[str, str]]) -> str:
        """Find the original task text in a mock conversation."""
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            task_idx = content.lower().find("task:")
            if task_idx == -1:
                continue
            task_text = content[task_idx + len("task:"):].strip()
            for marker in ("\n\nbegin.", "\n\nstart by", "\navailable tools:"):
                marker_idx = task_text.lower().find(marker)
                if marker_idx != -1:
                    task_text = task_text[:marker_idx].strip()
            return task_text
        return ""

    def _mock_os_task_action(self, task_description: str) -> str:
        """Return task-aware mock actions for the generated OS benchmark."""
        desc = task_description.lower()
        if not desc:
            return ""

        diverse_action = self._mock_diverse_os_task_action(task_description)
        if diverse_action:
            return diverse_action

        if "file named 'run.sh'" in desc or "make it executable" in desc:
            return (
                "Thought: I need to create the script, make it executable, and run it.\n"
                "Action: bash(printf '#!/bin/bash\\necho hello\\n' > run.sh && chmod +x run.sh && ./run.sh)"
            )

        if "count the number of lines" in desc and "result.txt" in desc:
            return (
                "Thought: I can count data.txt lines and write only the integer result.\n"
                "Action: bash(python3 -c \"from pathlib import Path; "
                "Path('result.txt').write_text(str(len(Path('data.txt').read_text().splitlines())))\" && cat result.txt)"
            )

        if "find the file named" in desc and "read its content" in desc:
            return (
                "Thought: I should locate the requested file and print its contents.\n"
                "Action: bash(find . -name 'secret*.txt' -type f -print -exec cat {} \\;)"
            )

        if "directory structure" in desc and "project/src" in desc:
            return (
                "Thought: I need to create all requested project directories and placeholders.\n"
                "Action: bash(mkdir -p project/src project/tests project/docs && "
                "touch project/src/.gitkeep project/tests/.gitkeep project/docs/.gitkeep && "
                "find project -maxdepth 2 -type d | sort)"
            )

        if "move all .txt files" in desc and "archive/manifest.txt" in desc:
            return (
                "Thought: I need to move inbox text files and record the archive manifest.\n"
                "Action: bash(mkdir -p archive && for f in inbox/*.txt; do [ -e \"$f\" ] && mv \"$f\" archive/; done; "
                "for f in archive/*.txt; do basename \"$f\"; done | sort > archive/manifest.txt; cat archive/manifest.txt)"
            )

        if "replace all occurrences of 'debug=false'" in desc:
            return (
                "Thought: I need to replace every DEBUG=false flag in config.ini.\n"
                "Action: bash(python3 -c \"from pathlib import Path; p=Path('config.ini'); "
                "p.write_text(p.read_text().replace('DEBUG=false','DEBUG=true'))\" && cat config.ini)"
            )

        if "update the 'host' field" in desc and "app.conf.bak" in desc:
            return (
                "Thought: I need to back up app.conf and update the host and port fields.\n"
                "Action: bash(cp app.conf app.conf.bak && python3 -c \"from pathlib import Path; p=Path('app.conf'); "
                "s=p.read_text().replace('host=localhost','host=127.0.0.1').replace('port=3000','port=8080'); "
                "p.write_text(s)\" && cat app.conf)"
            )

        if "add the line '# reviewed'" in desc and "src/review.md" in desc:
            return (
                "Thought: I need to prepend the review marker to each module and write the review list.\n"
                "Action: bash(python3 -c \"from pathlib import Path; files=[Path('src/module_a.py'),Path('src/module_b.py'),Path('src/module_c.py')]; "
                "[p.write_text('# reviewed\\n'+p.read_text()) for p in files]; "
                "Path('src/REVIEW.md').write_text('\\n'.join(p.name for p in files)+'\\n')\" && cat src/REVIEW.md)"
            )

        if "compute.py" in desc and "outputs 'ok'" in desc:
            return (
                "Thought: I need to fix compute.py by passing an integer and document the fix.\n"
                "Action: bash(perl -0pi -e 's/compute\\(\"hello\"\\)/compute(5)/' compute.py && "
                "echo 'Changed compute input to integer 5.' > fix.txt && python3 compute.py)"
            )

        if "initialize a git repository" in desc and "feature.py" in desc:
            return (
                "Thought: I need to create two commits, use a feature branch, and merge it to main.\n"
                "Action: bash(printf '# My Project\\n' > README.md && git add README.md && git commit -m 'Initial commit' && "
                "git switch -c feature && printf 'print(1)\\n' > feature.py && git add feature.py && "
                "git commit -m 'Add feature' && git switch main && git merge feature)"
            )

        if "broken test suite" in desc and "test_report.txt" in desc:
            return (
                "Thought: I should run the tests and save the passing report.\n"
                "Action: bash(python3 -m pytest tests/ -q > test_report.txt && cat test_report.txt)"
            )

        if "data processing pipeline" in desc and "pipeline.sh" in desc:
            return (
                "Thought: I need to create the input, processor, output, and runnable pipeline script.\n"
                "Action: bash(mkdir -p data/raw data/processed && printf 'value\\n1\\n2\\n3\\n4\\n5\\n' > data/raw/input.csv && "
                "printf 'from pathlib import Path\\nPath(\"data/processed/output.csv\").write_text(\"value\\\\n2\\\\n4\\\\n6\\\\n8\\\\n10\\\\n\")\\n' > process.py && "
                "python3 process.py && printf '#!/bin/bash\\npython3 process.py\\n' > pipeline.sh && chmod +x pipeline.sh && test -f data/processed/output.csv)"
            )

        return ""

    def _mock_diverse_os_task_action(self, task_description: str) -> str:
        """Return deterministic actions for diverse generated OS tasks."""
        text = task_description

        match = re.search(
            r"Find the file named '([^']+)'.*write the exact content to '([^']+)'",
            text,
        )
        if match:
            filename, output = match.groups()
            cmd = (
                f"mkdir -p {self._mock_parent_dir(output)} && "
                f"found=$(find . -name {shlex.quote(filename)} -type f -print -quit); "
                f"cat \"$found\" > {shlex.quote(output)} && cat {shlex.quote(output)}"
            )
            return (
                "Thought: I need to locate the requested file and copy its content to the answer file.\n"
                f"Action: bash({cmd})"
            )

        match = re.search(
            r"Create an executable script '([^']+)' that prints '([^']+)'.*save stdout to '([^']+)'",
            text,
        )
        if match:
            script, message, output = match.groups()
            cmd = (
                f"mkdir -p {self._mock_parent_dir(script)} {self._mock_parent_dir(output)} && "
                f"printf '#!/bin/bash\\necho %s\\n' {shlex.quote(message)} > {shlex.quote(script)} && "
                f"chmod +x {shlex.quote(script)} && "
                f"./{script} > {shlex.quote(output)} && cat {shlex.quote(output)}"
            )
            return (
                "Thought: I need to create the executable script, run it, and capture stdout.\n"
                f"Action: bash({cmd})"
            )

        match = re.search(
            r"Count the lines in '([^']+)' and write the integer total to '([^']+)'",
            text,
        )
        if match:
            source, output = match.groups()
            cmd = (
                f"mkdir -p {self._mock_parent_dir(output)} && "
                f"wc -l < {shlex.quote(source)} | tr -d ' ' > {shlex.quote(output)} && "
                f"cat {shlex.quote(output)}"
            )
            return (
                "Thought: I need to count source lines and write only the integer result.\n"
                f"Action: bash({cmd})"
            )

        match = re.search(
            r"Create these directories: (.+?)\. In each directory, create a '.gitkeep'",
            text,
        )
        if match:
            dirs = re.findall(r"'([^']+)'", match.group(1))
            mkdirs = " ".join(shlex.quote(d) for d in dirs)
            touches = " ".join(shlex.quote(f"{d}/.gitkeep") for d in dirs)
            cmd = f"mkdir -p {mkdirs} && touch {touches} && find . -maxdepth 3 -type d | sort"
            return (
                "Thought: I need to create every requested directory and placeholder file.\n"
                f"Action: bash({cmd})"
            )

        match = re.search(
            r"Move all \.txt files from '([^']+)' to '([^']+)'.*manifest to '([^']+)'",
            text,
        )
        if match:
            source, archive, manifest = match.groups()
            manifest_name = Path(manifest).name
            cmd = (
                f"mkdir -p {shlex.quote(archive)} {self._mock_parent_dir(manifest)} && "
                f"for f in {shlex.quote(source)}/*.txt; do [ -e \"$f\" ] && mv \"$f\" {shlex.quote(archive)}/; done; "
                f"find {shlex.quote(archive)} -maxdepth 1 -name '*.txt' ! -name {shlex.quote(manifest_name)} "
                f"-exec basename {{}} \\; | sort > {shlex.quote(manifest)} && "
                f"cat {shlex.quote(manifest)}"
            )
            return (
                "Thought: I need to move text files and write the moved-file manifest.\n"
                f"Action: bash({cmd})"
            )

        match = re.search(
            r"In file '([^']+)', replace every '([^']+)' with '([^']+)'",
            text,
        )
        if match:
            config, old, new = match.groups()
            code = (
                "from pathlib import Path\n"
                f"p = Path({config!r})\n"
                f"p.write_text(p.read_text().replace({old!r}, {new!r}))\n"
            )
            cmd = f"{self._mock_python_exec_cmd(code)} && cat {shlex.quote(config)}"
            return (
                "Thought: I need to replace every old token in the config file.\n"
                f"Action: bash({cmd})"
            )

        match = re.search(
            r"Read '([^']+)', update the 'host' field to '([^']+)'.*'port' field to '([^']+)'.*backup at '([^']+)'",
            text,
        )
        if match:
            config, host, port, backup = match.groups()
            code = (
                "from pathlib import Path\n"
                f"p = Path({config!r})\n"
                f"Path({backup!r}).parent.mkdir(parents=True, exist_ok=True)\n"
                f"Path({backup!r}).write_text(p.read_text())\n"
                "lines = []\n"
                "for line in p.read_text().splitlines():\n"
                f"    if line.startswith('host='): lines.append('host={host}')\n"
                f"    elif line.startswith('port='): lines.append('port={port}')\n"
                "    else: lines.append(line)\n"
                "p.write_text('\\n'.join(lines) + '\\n')\n"
            )
            cmd = f"{self._mock_python_exec_cmd(code)} && cat {shlex.quote(config)}"
            return (
                "Thought: I need to back up the config and update host and port.\n"
                f"Action: bash({cmd})"
            )

        match = re.search(
            r"Prepend the line '([^']+)' to each of these files: (.+?)\. Then create '([^']+)'",
            text,
        )
        if match:
            marker, files_text, review = match.groups()
            files = re.findall(r"'([^']+)'", files_text)
            code = (
                "from pathlib import Path\n"
                f"files = {[str(path) for path in files]!r}\n"
                f"marker = {marker!r}\n"
                "for raw in files:\n"
                "    p = Path(raw)\n"
                "    p.write_text(marker + '\\n' + p.read_text())\n"
                f"review = Path({review!r})\n"
                "review.parent.mkdir(parents=True, exist_ok=True)\n"
                "review.write_text('\\n'.join(Path(raw).name for raw in files) + '\\n')\n"
            )
            cmd = f"{self._mock_python_exec_cmd(code)} && cat {shlex.quote(review)}"
            return (
                "Thought: I need to prepend the review marker to all files and write the review list.\n"
                f"Action: bash({cmd})"
            )

        match = re.search(
            r"Fix script '([^']+)' so it prints 'OK', then write a fix summary to '([^']+)'",
            text,
        )
        if match:
            script, fix = match.groups()
            cmd = (
                f"mkdir -p {self._mock_parent_dir(fix)} && "
                f"perl -0pi -e 's/compute\\(\"bad\"\\)/compute(5)/' {shlex.quote(script)} && "
                f"echo 'Changed compute input to integer 5.' > {shlex.quote(fix)} && "
                f"python3 {shlex.quote(script)}"
            )
            return (
                "Thought: I need to fix the bad compute input and document the fix.\n"
                f"Action: bash({cmd})"
            )

        match = re.search(
            r"Initialize a git repository, create file '([^']+)' with content '([^']+)'.*branch '([^']+)'.*file '([^']+)' with 'print\((\d+)\)'",
            text,
        )
        if match:
            readme, title, branch, feature, number = match.groups()
            cmd = (
                f"printf '%s\\n' {shlex.quote(title)} > {shlex.quote(readme)} && "
                f"git add {shlex.quote(readme)} && git commit -m 'Initial commit' && "
                f"git switch -c {shlex.quote(branch)} && "
                f"printf 'print({number})\\n' > {shlex.quote(feature)} && "
                f"git add {shlex.quote(feature)} && git commit -m 'Add feature' && "
                "git switch main && git merge "
                f"{shlex.quote(branch)}"
            )
            return (
                "Thought: I need two commits on a feature branch and then merge back to main.\n"
                f"Action: bash({cmd})"
            )

        match = re.search(
            r"Run 'python3 -m pytest tests/'.*write the pytest summary to '([^']+)'",
            text,
        )
        if match:
            report = match.group(1)
            cmd = (
                f"mkdir -p {self._mock_parent_dir(report)} && "
                f"python3 -m pytest tests/ -q > {shlex.quote(report)} && "
                f"cat {shlex.quote(report)}"
            )
            return (
                "Thought: I need to run the test suite and save the pytest summary.\n"
                f"Action: bash({cmd})"
            )

        match = re.search(
            r"Set up a data pipeline: create '([^']+)'.*write script '([^']+)'.*into '([^']+)'.*create '([^']+)'",
            text,
        )
        if match:
            source, script, output, pipeline = match.groups()
            script_code = (
                "from pathlib import Path\n"
                f"Path({output!r}).parent.mkdir(parents=True, exist_ok=True)\n"
                f"Path({output!r}).write_text('value\\n2\\n4\\n6\\n8\\n10\\n')\n"
            )
            code = (
                "from pathlib import Path\n"
                f"Path({source!r}).parent.mkdir(parents=True, exist_ok=True)\n"
                f"Path({output!r}).parent.mkdir(parents=True, exist_ok=True)\n"
                f"Path({source!r}).write_text('value\\n1\\n2\\n3\\n4\\n5\\n')\n"
                f"Path({script!r}).write_text({script_code!r})\n"
                f"exec(compile({script_code!r}, {script!r}, 'exec'))\n"
                f"Path({pipeline!r}).write_text('#!/bin/bash\\npython3 {script}\\n')\n"
                f"Path({pipeline!r}).chmod(0o755)\n"
            )
            cmd = (
                f"{self._mock_python_exec_cmd(code)} && test -f {shlex.quote(output)}"
            )
            return (
                "Thought: I need to create the input, processor, output, and rerun script.\n"
                f"Action: bash({cmd})"
            )

        return ""

    def _mock_parent_dir(self, path: str) -> str:
        parent = Path(path).parent.as_posix()
        return shlex.quote(parent if parent != "." else ".")

    def _mock_python_exec_cmd(self, code: str) -> str:
        return "python3 -c " + shlex.quote(f"exec({code!r})")

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
