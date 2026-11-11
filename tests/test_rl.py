import ast
from pathlib import Path

import pytest

from scholarlens.llm.testing import ScriptedModel
from scholarlens.masking import Masker
from scholarlens.rl.data import dpo_pairs, prompts_from_contributions
from scholarlens.rl.evaluate import evaluate_policy, gate
from scholarlens.rl.rewards import LLMJudge, QuillReward, mask_from_masked

SRC = Path(__file__).parents[1] / "src" / "scholarlens"
TEXT = "Our results shows that BRCA1 expression rise by 12% in tumours (Smith et al., 2020) across the cohort."
MASKED = Masker().mask(TEXT).masked


class FakeGrammar:
    """Counts a few known errors so the quality reward can be tested without LanguageTool."""

    def count(self, text: str) -> int:
        import re

        return sum(len(re.findall(rf"\b{e}\b", text)) for e in ("results shows", "expression rise"))


def test_rl_package_never_imports_a_detector():
    """Design rule: Quill's rewards contain no detector or detection statistic."""
    for path in (SRC / "rl").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for n in names:
                assert not n.startswith("scholarlens.detect"), f"{path.name} imports {n}"
                assert not n.startswith("scholarlens.features.lm"), f"{path.name} imports {n}"
                assert n != "scholarlens.bench.metrics", f"{path.name} imports {n}"


def test_mask_from_masked_neutral_surfaces():
    m = mask_from_masked("We saw ⟦NUM_1⟧ cases ⟦CITE_1⟧ of ⟦TERM_1⟧.")
    assert m.render() == "We saw 3 cases [1] of X."
    assert m.verify("Cases of ⟦TERM_1⟧ numbered ⟦NUM_1⟧ ⟦CITE_1⟧.").ok


def test_integrity_failure_is_hard_gate():
    r = QuillReward().score(MASKED, MASKED.replace("⟦CITE_1⟧", ""))
    assert not r.integrity_ok and r.total == -1.0


def test_fixing_errors_beats_unchanged_beats_overediting():
    reward = QuillReward(grammar=FakeGrammar())
    fixed = MASKED.replace("results shows", "results show").replace("expression rise", "expression rises")
    rewritten = ("Across the cohort, tumours showed rising ⟦TERM_1⟧ levels of ⟦NUM_1⟧ according to our analysis "
                 "⟦CITE_1⟧, which our results show clearly.")
    s_fixed = reward.score(MASKED, fixed, mode="proofread")
    s_same = reward.score(MASKED, MASKED, mode="proofread")
    s_over = reward.score(MASKED, rewritten, mode="proofread")
    assert s_fixed.errors_before == 2 and s_fixed.errors_after == 0
    assert s_fixed.total > s_same.total > s_over.total
    assert s_over.economy < 0 and s_over.total < s_same.total  # no reward for wholesale rewriting


def test_semantic_drift_penalised():
    reward = QuillReward(similarity=lambda a, b: 0.80)
    assert reward.score(MASKED, MASKED.replace("rise", "fall")).total < 0
    assert QuillReward(similarity=lambda a, b: 0.97).score(MASKED, MASKED).total > 0


def test_format_penalty_for_chat_wrappers():
    reward = QuillReward()
    clean = reward.score(MASKED, MASKED).total
    wrapped = reward.score(MASKED, "Here is the revised paragraph:\n\n" + MASKED).total
    assert wrapped < clean


def test_llm_judge_parsing():
    assert LLMJudge(ScriptedModel(["5"])).score("a", "b") == 1.0
    assert LLMJudge(ScriptedModel(["Score: 1"])).score("a", "b") == -1.0
    assert LLMJudge(ScriptedModel(["no idea"])).score("a", "b") == 0.0


def test_dpo_pairs_from_decisions():
    acc = {"id": "a", "mode": "proofread", "format": "text", "decision": "accept", "masked_input": MASKED,
           "masked_output": MASKED.replace("results shows", "results show")}
    rej = {**acc, "id": "b", "decision": "reject"}
    broken = {**acc, "id": "c", "masked_output": MASKED.replace("⟦CITE_1⟧", "")}  # accepted but unsafe: dropped
    pairs = dpo_pairs([acc, rej, broken])
    assert len(pairs) == 2
    assert pairs[0]["chosen"][0]["content"] == acc["masked_output"] and pairs[0]["rejected"][0]["content"] == MASKED
    assert pairs[1]["chosen"][0]["content"] == MASKED
    assert pairs[0]["prompt"][0]["role"] == "system"
    assert len(prompts_from_contributions([acc, rej])) == 1  # same input de-duplicated


def test_evaluate_and_gate():
    examples = [{"masked": MASKED, "mode": "proofread"}] * 3
    good = evaluate_policy(ScriptedModel(lambda u: MASKED.replace("results shows", "results show")), examples,
                           QuillReward(grammar=FakeGrammar()))
    bad = evaluate_policy(ScriptedModel(lambda u: "Sure! " + MASKED.replace("⟦NUM_1⟧", "")), examples,
                          QuillReward(grammar=FakeGrammar()))
    assert good["integrity_rate"] == 1.0 and bad["integrity_rate"] == 0.0
    ok, _ = gate(good, bad)
    assert ok
    ok, reasons = gate(bad, good)
    assert not ok and any("integrity" in r for r in reasons)


@pytest.mark.slow
def test_grpo_trains_an_adapter(tmp_path):
    pytest.importorskip("trl")
    from scholarlens.rl.train import RLConfig, train_grpo

    ex = [{"prompt": [{"role": "user", "content": "<paragraph>\nWe was here ⟦CITE_1⟧.\n</paragraph>"}],
           "masked": "We was here ⟦CITE_1⟧.", "mode": "proofread"}] * 2
    cfg = RLConfig(out_dir=str(tmp_path / "a"), max_steps=1, num_generations=2, per_device_train_batch_size=2,
                   gradient_accumulation_steps=1, max_completion_length=24, use_cpu=True)
    out = train_grpo(ex, QuillReward(), cfg, log=lambda *_: None)
    assert (out / "adapter_model.safetensors").is_file() and (out / "scholarlens_adapter.json").is_file()
