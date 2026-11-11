"""One command to improve Quill: GRPO on public paragraphs + DPO on community decisions,
then a gated comparison with the current adapter on a frozen evaluation set."""

from __future__ import annotations

import json
import random
from pathlib import Path

from scholarlens.bench.schema import read_jsonl
from scholarlens.learn.contribute import load_contributions
from scholarlens.rl.data import dpo_pairs, prompts_from_contributions, prompts_from_records
from scholarlens.rl.evaluate import evaluate_policy, gate
from scholarlens.rl.rewards import QuillReward
from scholarlens.rl.train import RLConfig, train_dpo, train_grpo


def improve_quill(*, bench: str | Path, repo_dir: str | Path, cfg: RLConfig, reward: QuillReward, mode: str = "proofread",
                  current_adapter: str | None = None, n_train: int = 400, n_eval: int = 60, load_model=None,
                  log=print) -> dict:
    if load_model is None:
        from scholarlens.llm import load_model
    records = list(read_jsonl(bench))
    # frozen evaluation: the bench *test* split, never used for training
    eval_set = prompts_from_records(records, mode=mode, splits=("test",), limit=n_eval)
    train = prompts_from_records(records, mode=mode, splits=("train",), limit=n_train)
    contrib, notes = load_contributions(repo_dir)
    for n in notes:
        log(f"  contribution skipped: {n}")
    train += prompts_from_contributions(contrib, mode=mode)
    random.Random(cfg.seed).shuffle(train)
    log(f"train prompts {len(train)} · contributions {len(contrib)} · frozen eval {len(eval_set)}")

    out = Path(cfg.out_dir)
    adapter = train_grpo(train, reward, cfg, log=log)
    pairs = dpo_pairs(contrib)
    if pairs:
        log(f"DPO on {len(pairs)} author-decision pairs")
        adapter = train_dpo(pairs, RLConfig(**{**cfg.__dict__, "out_dir": str(out)}), init_adapter=adapter, log=log)

    opts = {"load_in_4bit": cfg.load_in_4bit, "max_tokens": cfg.max_completion_length}
    current = load_model(f"hf:{cfg.base_model}", adapter=current_adapter, **opts)
    cur_metrics = evaluate_policy(current, eval_set, reward)
    del current
    new_metrics = evaluate_policy(load_model(f"hf:{cfg.base_model}", adapter=str(adapter), **opts), eval_set, reward)
    accepted, reasons = gate(new_metrics, cur_metrics)
    report = {"accepted": accepted, "reasons": reasons, "current": {"adapter": current_adapter, **cur_metrics},
              "candidate": {"adapter": str(adapter), **new_metrics}, "mode": mode, "n_train": len(train),
              "n_contributions": len(contrib), "n_dpo_pairs": len(pairs), "base_model": cfg.base_model}
    (out / "learn_report.json").write_text(json.dumps(report, indent=2))
    return report
