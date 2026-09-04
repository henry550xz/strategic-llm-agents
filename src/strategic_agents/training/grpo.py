"""Strict action support and a local-policy GRPO training entry point."""
import json


def canonical_price_action(price: float) -> str:
    if price < 5 or price > 20 or abs(price*2-round(price*2)) > 1e-9:
        raise ValueError("optimizer action is outside the executable 0.5 grid")
    return json.dumps({"action_type": "set_price", "value": {"price": float(price)}}, separators=(",", ":"))


def parse_price_action(text: str) -> float:
    try: price = float(json.loads(text)["value"]["price"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc: raise ValueError("invalid action JSON") from exc
    canonical_price_action(price)
    return price


def train_grpo(config: dict, reward_function) -> dict:
    """Train a LoRA policy with TRL GRPOTrainer and an executable reward.

    The supplied reward function receives only strict-valid prices. Invalid or
    off-grid completions receive the configured execution-failure reward; they
    are never rounded or clamped.
    """
    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise ImportError("install strategic-llm-agents[train]") from exc
    records = [json.loads(line) for line in open(config["prompt_path"], encoding="utf-8") if line.strip()]
    dataset = Dataset.from_list(records)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config.get("revision"))
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config.get("revision"))
    lora = config.get("lora", {})
    model = get_peft_model(model, LoraConfig(r=int(lora.get("rank", 8)), lora_alpha=int(lora.get("alpha", 16)),
                         lora_dropout=float(lora.get("dropout", .05)), task_type="CAUSAL_LM"))
    invalid_reward = float(config.get("invalid_action_reward", -1.0))
    def strict_reward(completions, **kwargs):
        values = []
        for completion in completions:
            try: values.append(float(reward_function(parse_price_action(completion), **kwargs)))
            except (ValueError, TypeError, KeyError): values.append(invalid_reward)
        return values
    args = GRPOConfig(output_dir=config["output_dir"], max_steps=int(config.get("max_steps", 50)),
                      learning_rate=float(config.get("learning_rate", 1e-5)),
                      num_generations=int(config.get("group_size", 4)), report_to=[])
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=strict_reward,
                          args=args, train_dataset=dataset)
    result = trainer.train(); trainer.save_model(config["output_dir"] + "/adapter")
    return {"training_loss": result.training_loss, "records": len(records),
            "adapter": config["output_dir"] + "/adapter"}
