"""
Student failure analyzer using QLoRA fine-tuned local model.

Replaces the OpenAI/Grok API dependency at inference time.
Loads the LoRA adapter on top of the quantized base model and
provides the same interface as FailureAnalyzer.

Usage:
    from distillation.student_analyzer import StudentFailureAnalyzer
    analyzer = StudentFailureAnalyzer(adapter_path="models/failure_analyzer_lora")
    result = analyzer.analyze(trace, task_description)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tracing.logger import ExecutionTrace


@dataclass
class FailureAnalysis:
    rule_based_findings: list[str]
    failure_category: str | None
    corrective_strategy: str
    raw_response: str = ""
    model_source: str = "student"


class StudentFailureAnalyzer:
    """
    Local failure analyzer using QLoRA fine-tuned LLaMA-3.2-1B-Instruct.

    The model is loaded in 4-bit NF4 quantization with the LoRA adapter
    produced by qlora_trainer.py. At inference, it accepts the same
    execution trace format and produces the same JSON output as
    the Grok 4 teacher.
    """

    def __init__(
        self,
        adapter_path: str | Path = "models/failure_analyzer_lora",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> None:
        self.adapter_path = Path(adapter_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "Install: pip install transformers peft bitsandbytes accelerate"
            ) from exc

        manifest_path = self.adapter_path / "training_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No training manifest at {manifest_path}. "
                "Run distillation/qlora_trainer.py first."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        base_model = manifest["base_model"]

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb_config, device_map="auto"
        )
        self._model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_path))
        self._tokenizer.pad_token = self._tokenizer.eos_token

    def _infer(self, prompt: str) -> str:
        import torch
        self._load()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_ids = ids[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def _format_prompt(self, task_description: str, trace: ExecutionTrace, rule_findings: list[str]) -> str:
        steps_text = []
        for s in trace.steps[:15]:
            act = s.action.get("tool") or s.action.get("action") or str(s.action)[:60]
            obs = (s.observation or "")[:200]
            steps_text.append(f"Step {s.step}: {act} → {obs}")

        user_content = (
            f"Task: {task_description}\n\n"
            f"Rule-based findings: {rule_findings or 'None'}\n\n"
            f"Execution trace:\n" + "\n".join(steps_text) +
            '\n\nRespond with JSON: {"category": "...", "corrective_strategy": "..."}'
        )
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are an expert agent-failure analyst. Respond with JSON only."
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_content}"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

    def analyze(self, trace: ExecutionTrace, task_description: str) -> FailureAnalysis:
        rule_findings = self._rule_based_checks(trace)
        prompt = self._format_prompt(task_description, trace, rule_findings)
        raw = self._infer(prompt)

        try:
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            data = json.loads(raw)
            category = data.get("category")
            strategy = data.get("corrective_strategy", "Retry with careful planning")
        except (json.JSONDecodeError, KeyError):
            category = None
            strategy = "Retry with careful planning; verify environment state first"

        return FailureAnalysis(
            rule_based_findings=rule_findings,
            failure_category=category,
            corrective_strategy=strategy,
            raw_response=raw,
            model_source="student_qlora",
        )

    def _rule_based_checks(self, trace: ExecutionTrace) -> list[str]:
        from collections import Counter
        findings = []
        actions = [s.action.get("tool") or s.action.get("action") for s in trace.steps]
        for action, count in Counter(a for a in actions if a).items():
            if count >= 3:
                findings.append(f"Repeated action '{action}' {count} times")
        for s in trace.steps:
            obs = s.observation or ""
            if "Error:" in obs or "does not exist" in obs:
                findings.append(f"Environment error in step {s.step}: {obs[:80]}")
                break
        return findings
