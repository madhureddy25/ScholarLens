"""Train Quill with GRPO (automatic, verifiable rewards) and DPO (author decisions).

Both produce a LoRA adapter on top of a small open instruct model, so training
fits a 6 GB laptop GPU (Qwen2.5-0.5B/1.5B-Instruct; add ``load_in_4bit`` for 1.5B+)
and the result is a few tens of MB that anyone can download and use with
``-m hf:<base> --opt adapter=<dir>``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from scholarlens.rl.rewards import QuillReward


@dataclass
class RLConfig:
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    out_dir: str = "models/quill-rl"
    max_steps: int = 200
    learning_rate: float = 1e-5
    num_generations: int = 4  # GRPO group size G
    max_completion_length: int = 448
    temperature: float = 0.9
    beta: float = 0.04  # KL to the base model: keeps the author-facing behaviour close to the base
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    load_in_4bit: bool = False
    use_cpu: bool = False
    seed: int = 13
    logging_steps: int = 5

    def peft(self):
        from peft import LoraConfig

        return LoraConfig(r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    def quantization(self):
        if not self.load_in_4bit:
            return None
        import torch
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    def precision(self) -> dict:
        if self.use_cpu:
            return {"use_cpu": True}
        import torch

        return {"bf16": True} if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else {"fp16": True}


def _write_card(out: Path, cfg: RLConfig, kind: str, n: int, extra: dict | None = None) -> None:
    (out / "scholarlens_adapter.json").write_text(json.dumps(
        {"kind": kind, "base_model": cfg.base_model, "n_examples": n, "config": asdict(cfg), **(extra or {})}, indent=2), encoding="utf-8")


def fits(example: dict, cfg: RLConfig) -> bool:
    """Skip paragraphs whose rewrite cannot finish within max_completion_length: a truncated
    completion always fails integrity, every sample in the group scores the same, the
    group-relative advantage is zero and nothing is learned."""
    return len(example["masked"].split()) * 1.6 < cfg.max_completion_length * 0.85


def train_grpo(examples: list[dict], reward: QuillReward, cfg: RLConfig, *, log=print) -> Path:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    def reward_fn(prompts, completions, masked, mode, **_):
        texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
        return [reward.score(m, t, mode=md).total for m, t, md in zip(masked, texts, mode, strict=True)]

    reward_fn.__name__ = "quill_verifiable_reward"
    args = GRPOConfig(output_dir=cfg.out_dir, learning_rate=cfg.learning_rate, max_steps=cfg.max_steps,
                      num_generations=cfg.num_generations, max_completion_length=cfg.max_completion_length,
                      temperature=cfg.temperature, beta=cfg.beta,
                      per_device_train_batch_size=cfg.per_device_train_batch_size,
                      gradient_accumulation_steps=cfg.gradient_accumulation_steps, logging_steps=cfg.logging_steps,
                      save_strategy="no", report_to="none", seed=cfg.seed, gradient_checkpointing=not cfg.use_cpu,
                      **cfg.precision())
    usable = [e for e in examples if fits(e, cfg)]
    log(f"GRPO on {len(usable)} paragraphs ({len(examples) - len(usable)} too long for max_completion_length)")
    if not usable:
        raise ValueError("no paragraphs short enough for max_completion_length")
    ds = Dataset.from_list([{k: e[k] for k in ("prompt", "masked", "mode")} for e in usable])
    trainer = GRPOTrainer(model=cfg.base_model, reward_funcs=reward_fn, args=args, train_dataset=ds,
                          peft_config=cfg.peft(), quantization_config=cfg.quantization())
    trainer.train()
    out = Path(cfg.out_dir)
    trainer.save_model(str(out))
    _write_card(out, cfg, "grpo", len(examples))
    return out


def train_dpo(pairs: list[dict], cfg: RLConfig, *, init_adapter: str | Path | None = None, log=print) -> Path:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    args = DPOConfig(output_dir=cfg.out_dir, learning_rate=cfg.learning_rate / 2, max_steps=cfg.max_steps,
                     beta=0.1, per_device_train_batch_size=max(1, cfg.per_device_train_batch_size // 2),
                     gradient_accumulation_steps=cfg.gradient_accumulation_steps, logging_steps=cfg.logging_steps,
                     save_strategy="no", report_to="none", seed=cfg.seed, max_length=1024, **cfg.precision())
    model = cfg.base_model
    peft_config = cfg.peft()
    if init_adapter is not None:  # continue from the GRPO adapter
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        if cfg.load_in_4bit:
            from peft import prepare_model_for_kbit_training

            base = prepare_model_for_kbit_training(AutoModelForCausalLM.from_pretrained(
                cfg.base_model, quantization_config=cfg.quantization(), device_map="auto"))
        else:
            base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype="auto")
        model = PeftModel.from_pretrained(base, str(init_adapter), is_trainable=True)
        peft_config = None
    trainer = DPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(pairs), peft_config=peft_config,
                         quantization_config=cfg.quantization() if init_adapter is None else None)
    trainer.train()
    out = Path(cfg.out_dir)
    trainer.save_model(str(out))
    _write_card(out, cfg, "dpo" if init_adapter is None else "grpo+dpo", len(pairs))
    return out
