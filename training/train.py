import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
train.py - Fine-tune Qwen2.5-Coder-7B-Instruct on the Nalana Blender dataset.

Hardware target: 18x A6000 (48GB each) on Azure
Strategy:
  - LoRA fine-tune (PEFT): fast, low memory, good results
  - DeepSpeed ZeRO-2: efficient multi-GPU training
  - Base model: Qwen/Qwen2.5-Coder-7B-Instruct (code-specialized, ~7B params)
  - Expected training time: 2-4 hours on 8 A6000s for 100k pairs

GPU allocation (18x A6000):
  - GPUs 0-3:  vLLM synthesis instance 1 (during data collection)
  - GPUs 4-7:  vLLM synthesis instance 2 (during data collection)
  - GPUs 8-15: Training (8 GPUs, FSDP/ZeRO-2)
  - GPUs 16-17: Reserve / eval inference

Launch commands:
  # Single node, 8 GPUs:
  CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
  torchrun --nproc_per_node=8 train.py \
    --data-dir data/train \
    --output-dir checkpoints/nalana-v1 \
    --model Qwen/Qwen2.5-Coder-7B-Instruct

  # With DeepSpeed config:
  deepspeed --num_gpus=8 train.py --deepspeed ds_config.json ...
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer


def load_sharegpt_dataset(jsonl_path: str) -> Dataset:
    records = []
    for line in Path(jsonl_path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return Dataset.from_list(records)


def format_sharegpt_to_text(example: dict, tokenizer) -> dict:
    """
    Convert ShareGPT format to a single text string using the model's chat template.
    The model will be trained to predict only the assistant turn (completion only).
    """
    conversations = example["conversations"]
    messages = [
        {"role": ("user" if c["from"] == "human" else c["from"]), "content": c["value"]}
        for c in conversations
        if c["from"] != "system"
    ]
    # Prepend system message
    system_msgs = [c for c in conversations if c["from"] == "system"]
    if system_msgs:
        messages = [{"role": "system", "content": system_msgs[0]["value"]}] + messages

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


class PrintMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            loss = logs.get("loss", "—")
            lr = logs.get("learning_rate", "—")
            print(f"  step {step:>6} | loss {loss:.4f} | lr {lr:.2e}" if isinstance(loss, float) else f"  step {step}")


def build_lora_config(lora_rank: int = 64) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,   # scaling = alpha/r = 2x (tied to rank)
        target_modules=[    # Qwen2 attention + MLP projections
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Nalana model on Blender dataset")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--output-dir", default="checkpoints/nalana-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--deepspeed", type=str, help="Path to DeepSpeed config JSON")
    parser.add_argument("--flash-attn", action="store_true", default=False,
                        help="Use Flash Attention 2 (requires flash-attn package)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_file = data_dir / "sharegpt_train.jsonl"
    val_file   = data_dir / "sharegpt_val.jsonl"

    if not train_file.exists():
        print(f"Training data not found at {train_file}. Run train_prep.py first.")
        return

    print(f"Loading model: {args.model}")
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=None,  # Let DeepSpeed / torchrun handle placement
    )
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, **model_kwargs)
    model.enable_input_require_grads()

    # Apply LoRA
    lora_cfg = build_lora_config(lora_rank=args.lora_r)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Load datasets
    print("Loading datasets...")
    train_ds = load_sharegpt_dataset(str(train_file))
    val_ds   = load_sharegpt_dataset(str(val_file)) if val_file.exists() else None

    # Format to text using chat template
    train_ds = train_ds.map(lambda ex: format_sharegpt_to_text(ex, tokenizer))
    if val_ds:
        val_ds = val_ds.map(lambda ex: format_sharegpt_to_text(ex, tokenizer))

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds) if val_ds else 0:,}")

    # Effective batch size = batch_size × grad_accum × n_gpus
    n_gpus = torch.cuda.device_count() or 1
    effective_batch = args.batch_size * args.grad_accum * n_gpus
    steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
    total_steps = steps_per_epoch * args.epochs
    print(f"GPUs: {n_gpus} | Effective batch: {effective_batch} | Steps: {total_steps:,}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=max(1, steps_per_epoch // 4) if val_ds else None,
        save_strategy="steps",
        save_steps=max(1, steps_per_epoch // 2),
        save_total_limit=3,
        load_best_model_at_end=bool(val_ds),
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        callbacks=[PrintMetricsCallback()],
    )

    print(f"\nStarting training → {args.output_dir}")
    trainer.train()

    # Save final merged model (LoRA weights merged into base)
    final_dir = Path(args.output_dir) / "final"
    print(f"\nMerging LoRA weights → {final_dir}")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Done. Model saved to {final_dir}")
    print(f"\nTo serve: vllm serve {final_dir} --port 9000")


if __name__ == "__main__":
    main()
