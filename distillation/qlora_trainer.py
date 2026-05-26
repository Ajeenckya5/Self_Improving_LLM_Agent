"""
QLoRA fine-tuning pipeline for failure analysis student model.

Trains a small local model (LLaMA-3.2-1B-Instruct) on Grok-4-generated
failure analysis data using 4-bit NF4 quantization + LoRA adapters.
The fine-tuned adapter reduces API dependency and inference cost by ~95%.

Requirements:
    pip install transformers peft trl bitsandbytes accelerate datasets

Usage:
    python -m distillation.qlora_trainer \\
        --data data/distill_train.jsonl \\
        --output models/failure_analyzer_lora \\
        --epochs 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

LORA_CONFIG = dict(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

BNB_CONFIG = dict(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)

TRAINING_ARGS = dict(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=50,
    fp16=False,
    bf16=True,
    optim="paged_adamw_8bit",
    max_seq_length=1024,
    report_to="none",
)


def _format_prompt(example: dict) -> str:
    """Convert instruction-following example to chat prompt."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are an expert agent-failure analyst. Respond with JSON only."
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{example['instruction']}"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        f"{example['output']}<|eot_id|>"
    )


def load_dataset_from_jsonl(path: Path):
    """Load JSONL into HuggingFace Dataset."""
    from datasets import Dataset
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    dataset = Dataset.from_list(records)
    dataset = dataset.map(lambda x: {"text": _format_prompt(x)})
    return dataset


def train(
    data_path: Path,
    output_dir: Path,
    base_model: str = BASE_MODEL,
    epochs: int = 3,
    hf_token: str | None = None,
) -> None:
    """
    Fine-tune base_model on failure analysis data with QLoRA.

    Steps:
    1. Load base model in 4-bit NF4 (bitsandbytes)
    2. Apply LoRA adapters via peft
    3. Fine-tune with SFTTrainer (trl)
    4. Save LoRA adapter weights to output_dir
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        import torch
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}\n"
            "Install with: pip install transformers peft trl bitsandbytes accelerate datasets"
        ) from exc

    print(f"Loading dataset from {data_path}...")
    dataset = load_dataset_from_jsonl(data_path)
    print(f"  {len(dataset)} training examples")

    print(f"Loading base model: {base_model} in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(**BNB_CONFIG)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Applying LoRA adapters...")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    print(f"  Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    args_dict = {**TRAINING_ARGS, "num_train_epochs": epochs, "output_dir": str(output_dir)}
    training_args = TrainingArguments(**args_dict)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=TRAINING_ARGS["max_seq_length"],
        args=training_args,
    )

    print("Starting QLoRA fine-tuning...")
    trainer.train()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"LoRA adapter saved to {output_dir}")

    # Write training manifest
    manifest = {
        "base_model": base_model,
        "teacher_model": "grok-4",
        "lora_config": LORA_CONFIG,
        "bnb_config": BNB_CONFIG,
        "training_args": {k: v for k, v in TRAINING_ARGS.items()},
        "epochs": epochs,
        "training_examples": len(dataset),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for failure analysis student model")
    parser.add_argument("--data", type=Path, required=True,
                        help="JSONL training data from grok_teacher.py")
    parser.add_argument("--output", type=Path, default=Path("models/failure_analyzer_lora"),
                        help="Output directory for LoRA adapter")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for gated models")
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        hf_token=args.hf_token,
    )
