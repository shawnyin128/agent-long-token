"""Type-level ablation (spec §5 + §7 Gate 2).

For each of the 6 claim types t, remove all claims of that type from
debate history and replay the final round; compute Δ_t = acc(with) −
acc(without) on the ``single_wrong ∧ debate_right`` subset.

A hard ``MAX_NEW_LLM_CALLS=500`` cap (spec §6.1) is enforced against
``LLMClient.call_count`` — cache hits are free.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

from agentdiet.config import Config
from agentdiet.dataset import parse_answer
from agentdiet.types import Dialogue, Message


log = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """Raised when ablation exhausts the MAX_NEW_LLM_CALLS budget."""


MAX_NEW_LLM_CALLS_DEFAULT = 500


def load_dialogue_and_claims(cfg: Config, qid: str) -> tuple[Dialogue, dict]:
    dpath = cfg.dialogues_dir / f"{qid}.json"
    cpath = cfg.claims_dir / f"{qid}.json"
    if not dpath.exists():
        raise FileNotFoundError(f"dialogue artifact missing: {dpath}")
    if not cpath.exists():
        raise FileNotFoundError(f"claims artifact missing: {cpath}")
    d = Dialogue.model_validate_json(dpath.read_text(encoding="utf-8"))
    cd = json.loads(cpath.read_text(encoding="utf-8"))
    return d, cd


def _pilot_single_path(cfg: Config, qid: str) -> Path:
    return cfg.artifacts_dir / "pilot" / "single" / cfg.model_slug / f"{qid}.json"


def _load_single_doc(cfg: Config, qid: str) -> Optional[dict]:
    path = _pilot_single_path(cfg, qid)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_single_wrong_debate_right(
    single_doc: dict, debate_dialogue: Dialogue, gold: str
) -> bool:
    s_final = single_doc.get("final_answer")
    d_final = debate_dialogue.final_answer
    if s_final is None or d_final is None:
        return False
    return str(s_final).strip() != str(gold).strip() and \
        str(d_final).strip() == str(gold).strip()


def select_subset(cfg: Config, target_size: int) -> list[str]:
    dialogue_qids = {p.stem for p in cfg.dialogues_dir.glob("*.json")}
    claim_qids = {p.stem for p in cfg.claims_dir.glob("*.json")}
    both = sorted(dialogue_qids & claim_qids)

    eligible: list[str] = []
    for qid in both:
        single = _load_single_doc(cfg, qid)
        if single is None:
            continue
        try:
            dialogue = Dialogue.model_validate_json(
                (cfg.dialogues_dir / f"{qid}.json").read_text(encoding="utf-8")
            )
        except Exception:  # noqa: BLE001
            continue
        if is_single_wrong_debate_right(single, dialogue, dialogue.gold_answer):
            eligible.append(qid)

    if target_size >= len(eligible):
        return eligible
    rng = random.Random(cfg.seed)
    return sorted(rng.sample(eligible, target_size))
