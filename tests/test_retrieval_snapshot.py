"""Snapshot regression tests on real KG.

Loads the KG once (module-scoped fixture) and verifies that 5 canonical
questions still retrieve the expected sections above a loose threshold.
These act as canaries: a dead retrieval pipeline or a regression in boost
keywords / tokenization / scoring will cause one or more to fail.

Skip automatically if the parquet KG isn't built yet.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from eval.evaluation import QAPair, citation_f1


# Frozen snapshots: (question, expected_sections, citation_F1_threshold).
# Threshold is loose so day-to-day tuning doesn't break tests; failure means
# a real retrieval/answer regression.
SNAPSHOTS: list[tuple[str, list[str], float]] = [
    ("การลักทรัพย์มีโทษอะไร?",          ["334"],         0.5),
    ("ฉ้อโกงมีองค์ประกอบความผิดอย่างไร?",  ["341"],         0.5),
    ("ละเมิดคืออะไร?",                  ["420"],         0.5),
    ("ยักยอกทรัพย์มีโทษอย่างไร?",          ["352"],         0.5),
    ("อายุความในคดีแพ่งทั่วไปกำหนดไว้กี่ปี?", ["193/30"],     0.4),
]


@pytest.fixture(scope="module")
def kg():
    parquet_dir = Path(config.PARQUET_DIR) if hasattr(config, "PARQUET_DIR") else None
    if parquet_dir is None or not parquet_dir.exists():
        # Fallback search
        candidates = list(Path("output").rglob("entities.parquet"))
        if not candidates:
            pytest.skip("Parquet KG not built; run `python main.py index` first")

    from graphrag.graph_builder import KnowledgeGraph
    g = KnowledgeGraph()
    g.load_parquet()
    g.compute_embeddings()
    return g


@pytest.mark.parametrize("question,sections,threshold", SNAPSHOTS)
def test_retrieval_snapshot(kg, question, sections, threshold):
    """Each canonical question should still retrieve its target sections."""
    from graphrag.query_engine import answer

    res = answer(question, kg, mode="local")
    sr = res["search_result"]

    # Hit rate: at least one expected section appears in retrieved entity names.
    section_ents = [e for e in sr.entities if e.get("type") == "SECTION"]
    retrieved_secs = set()
    for e in section_ents[:20]:
        retrieved_secs.update(re.findall(r"มาตรา\s*(\d+(?:/\d+)?)", e.get("name", "")))
    assert any(s in retrieved_secs for s in sections), (
        f"None of {sections} retrieved for: {question}\n"
        f"  retrieved: {sorted(retrieved_secs)[:10]}"
    )

    # Citation F1 on the generated answer.
    f1 = citation_f1(res["answer"], sections)
    assert f1 >= threshold, (
        f"citation_f1={f1:.2f} < {threshold} for: {question}\n"
        f"  answer head: {res['answer'][:200]!r}"
    )
