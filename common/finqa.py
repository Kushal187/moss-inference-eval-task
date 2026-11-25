"""Bridge to the FinQA project (git submodule at external/finqa).

Reuses its prompt builders, answer matching, and judge prompts unchanged, so the only
variable in a Moss-vs-baseline comparison is the model/API.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINQA = ROOT / "external" / "finqa"
os.environ.setdefault("DATA_PROCESSED_DIR", str(FINQA / "data" / "processed"))
if str(FINQA) not in sys.path:
    sys.path.insert(0, str(FINQA))

from src.data.load_data import load_chunks, load_qa  # noqa: E402
from src.generation.answer import answers_match, extract_final_answer, normalize_answer  # noqa: E402
from src.generation.context import get_chunk_contents  # noqa: E402
from src.generation.prompts import build_messages, load_few_shot_examples  # noqa: E402
from src.detection import llm_judge  # noqa: E402

SAVED_RUNS = FINQA / "data" / "runs"
DETECTION_SET = FINQA / "data" / "processed" / "detection_eval.jsonl"

__all__ = [
    "FINQA", "SAVED_RUNS", "DETECTION_SET", "load_chunks", "load_qa", "answers_match",
    "extract_final_answer", "normalize_answer", "get_chunk_contents", "build_messages",
    "load_few_shot_examples", "llm_judge",
]
