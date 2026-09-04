"""LoRA supervised fine-tuning for local strategic policies.

Heavy dependencies are imported only when training is requested, so evaluator
and simulator demos remain CPU-light.
"""
from pathlib import Path
import inspect


def make_example(context: str, canonical_action: str) -> dict[str, str]:
    return {"prompt": context, "completion": canonical_action}


def train_lora_sft(config: dict) -> dict:
    """Train and save a causal-LM LoRA adapter from JSONL `text` records."""
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  DataCollatorForLanguageModeling, Trainer, TrainingArguments)
    except ImportError as exc:
        raise ImportError("install strategic-llm-agents[train]") from exc
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config.get("revision"))
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config.get("revision"),
                                                 torch_dtype=getattr(torch, config.get("dtype", "float32")))
    lora = config.get("lora", {})
    model = get_peft_model(model, LoraConfig(r=int(lora.get("rank", 8)),
                         lora_alpha=int(lora.get("alpha", 16)), lora_dropout=float(lora.get("dropout", .05)),
                         target_modules=lora.get("target_modules"), task_type="CAUSAL_LM"))
    data = load_dataset("json", data_files={"train": config["train_path"], "validation": config["validation_path"]})
    maximum = int(config.get("max_length", 1024))
    tokenized = data.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=maximum),
                         batched=True, remove_columns=data["train"].column_names)
    output = Path(config["output_dir"])
    arguments = dict(output_dir=str(output), seed=seed,
        max_steps=int(config.get("max_steps", 200)), learning_rate=float(config.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(config.get("batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation", 4)),
        eval_steps=int(config.get("eval_steps", 50)), save_steps=int(config.get("save_steps", 100)), report_to=[])
    strategy_name = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    arguments[strategy_name] = "steps"
    args = TrainingArguments(**arguments)
    trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"],
                      eval_dataset=tokenized["validation"],
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trained = trainer.train(); evaluation = trainer.evaluate()
    adapter = output / "adapter"; model.save_pretrained(adapter); tokenizer.save_pretrained(adapter)
    return {"train_examples": len(data["train"]), "validation_examples": len(data["validation"]),
            "training_loss": trained.training_loss, "evaluation_loss": evaluation.get("eval_loss"),
            "adapter": str(adapter)}
