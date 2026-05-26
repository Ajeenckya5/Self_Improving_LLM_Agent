"""
Grok 4 teacher for knowledge distillation.

Uses Grok 4 (xAI API, OpenAI-compatible) to generate high-quality
failure analysis annotations from agent execution traces. Output is
saved as instruction-following JSONL to be used for QLoRA fine-tuning
of a smaller student model.

Usage:
    python -m distillation.grok_teacher --traces-dir traces/ --out data/distill_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

XAI_BASE_URL = "https://api.x.ai/v1"
GROK_MODEL = "grok-4"

SYSTEM_PROMPT = """You are an expert agent-failure analyst. Given an execution trace of a
long-horizon task agent, identify:
1. The root failure category
2. A concrete, reusable corrective strategy for future similar tasks

Always respond with JSON only — no markdown, no commentary."""

TASK_TEMPLATE = """Analyze this failed agent execution.

Task description: {task_description}

Execution trace:
{trace_steps}

Rule-based findings: {rule_findings}

Respond with JSON:
{{
  "category": "<one of: tool_usage_error, reasoning_error, environment_misread, loop_detected, path_error, schema_error, constraint_violation, timeout, unknown>",
  "corrective_strategy": "<One specific, actionable rule. Example: Always list_dir before moving files to confirm filenames exist.>",
  "root_cause": "<Brief explanation of why the task failed>"
}}"""


def _call_grok(prompt: str, api_key: str, temperature: float = 0.1) -> str:
    payload = json.dumps({
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }).encode("utf-8")

    request = Request(
        f"{XAI_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Grok API error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach Grok API: {exc.reason}") from exc


def _format_trace(trace_data: dict[str, Any]) -> str:
    steps = trace_data.get("steps", [])
    lines = []
    for s in steps[:15]:
        act = s.get("action", {})
        tool = act.get("tool") or act.get("action") or str(act)[:60]
        obs = (s.get("observation") or "")[:200]
        reason = (s.get("reasoning") or "")[:120]
        line = f"Step {s.get('step', '?')}: {tool} → {obs}"
        if reason:
            line += f" | thinking: {reason}"
        lines.append(line)
    return "\n".join(lines) if lines else "(no steps)"


def generate_training_data(
    traces_dir: Path,
    output_path: Path,
    api_key: str,
    max_traces: int = 500,
    delay_s: float = 0.5,
) -> int:
    """
    Read JSONL trace files from traces_dir, call Grok 4 for each failed trace,
    and write instruction-following JSONL for QLoRA training.

    Returns the number of training examples written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_files = sorted(
        [*traces_dir.glob("*.json"), *traces_dir.glob("*.jsonl")]
    )[:max_traces]

    count = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for trace_file in trace_files:
            try:
                raw_trace = trace_file.read_text(encoding="utf-8").strip()
                if not raw_trace:
                    continue
                if trace_file.suffix == ".jsonl":
                    # JSONL traces may contain one record per line. Current TraceLogger
                    # writes JSON, but accepting JSONL keeps this script compatible with
                    # exported trace streams.
                    trace_data = json.loads(raw_trace.splitlines()[-1])
                else:
                    trace_data = json.loads(raw_trace)
            except (json.JSONDecodeError, OSError):
                continue

            # Only distill from failures
            if trace_data.get("success", True):
                continue

            task_desc = trace_data.get("task_description", "unknown task")
            trace_steps = _format_trace(trace_data)
            rule_findings = trace_data.get("rule_findings", [])

            user_prompt = TASK_TEMPLATE.format(
                task_description=task_desc,
                trace_steps=trace_steps,
                rule_findings=rule_findings or "None",
            )

            try:
                raw = _call_grok(user_prompt, api_key)
                # Strip markdown if present
                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0].strip()
                analysis = json.loads(raw)
            except (RuntimeError, json.JSONDecodeError) as exc:
                print(f"  [SKIP] {trace_file.name}: {exc}")
                continue

            # Instruction-following format for SFT / QLoRA
            example = {
                "instruction": user_prompt,
                "input": "",
                "output": json.dumps(analysis, ensure_ascii=False),
                "task_id": trace_data.get("task_id", trace_file.stem),
                "teacher_model": GROK_MODEL,
            }
            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
            time.sleep(delay_s)

    print(f"Generated {count} training examples → {output_path}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grok 4 teacher: generate distillation data")
    parser.add_argument("--traces-dir", type=Path, default=Path("traces"),
                        help="Directory of JSONL execution traces")
    parser.add_argument("--out", type=Path, default=Path("data/distill_train.jsonl"),
                        help="Output JSONL path")
    parser.add_argument("--max-traces", type=int, default=500)
    parser.add_argument("--api-key", default=os.getenv("XAI_API_KEY", ""),
                        help="xAI API key (or set XAI_API_KEY env var)")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Set XAI_API_KEY environment variable or pass --api-key")

    n = generate_training_data(
        traces_dir=args.traces_dir,
        output_path=args.out,
        api_key=args.api_key,
        max_traces=args.max_traces,
    )
    print(f"Done. {n} examples written.")
