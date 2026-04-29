"""Pure-unit regression tests (no KG loading).

Covers helpers whose behaviour we want frozen against accidental changes:
- _extract_cited_sections regex (must require 'มาตรา' prefix)
- citation_f1 edge cases
- k-fold splitter (deterministic, disjoint, covering)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.evaluation import _extract_cited_sections, citation_f1
from scripts.kfold_eval import _kfold_indices


# ── _extract_cited_sections ──────────────────────────────────────────

def test_extract_cited_sections_basic():
    text = "**มาตราที่เกี่ยวข้อง:** มาตรา 334, มาตรา 335"
    assert _extract_cited_sections(text) == {"334", "335"}


def test_extract_cited_sections_ignores_year():
    # Years like 2560 must NOT be picked up.
    text = "พุทธศักราช 2560\n**มาตราที่เกี่ยวข้อง:** มาตรา 5"
    assert _extract_cited_sections(text) == {"5"}


def test_extract_cited_sections_with_slash():
    text = "**มาตราที่เกี่ยวข้อง:** มาตรา 112/1, มาตรา 200"
    assert _extract_cited_sections(text) == {"112/1", "200"}


def test_extract_cited_sections_fallback_when_no_citation_line():
    text = "ตามมาตรา 7 และมาตรา 9 แล้ว ..."
    assert _extract_cited_sections(text) == {"7", "9"}


# ── citation_f1 ──────────────────────────────────────────────────────

def test_citation_f1_perfect():
    text = "**มาตราที่เกี่ยวข้อง:** มาตรา 334"
    assert citation_f1(text, ["334"]) == 1.0


def test_citation_f1_empty_both():
    # No prediction and no ground truth → vacuously correct.
    assert citation_f1("ไม่ระบุมาตรา", []) == 1.0


def test_citation_f1_miss():
    text = "**มาตราที่เกี่ยวข้อง:** มาตรา 100"
    assert citation_f1(text, ["334"]) == 0.0


def test_citation_f1_partial():
    text = "**มาตราที่เกี่ยวข้อง:** มาตรา 334, มาตรา 335, มาตรา 336"
    f1 = citation_f1(text, ["334", "335"])
    # precision=2/3, recall=2/2 → 2*pr*re/(pr+re) = 0.8
    assert abs(f1 - 0.8) < 1e-6


# ── k-fold splitter ─────────────────────────────────────────────────

def test_kfold_disjoint_and_covering():
    folds = _kfold_indices(n=112, k=5, seed=42)
    assert len(folds) == 5
    flat = [i for fold in folds for i in fold]
    assert sorted(flat) == list(range(112))
    # No duplicates across folds
    assert len(set(flat)) == 112


def test_kfold_deterministic():
    a = _kfold_indices(n=50, k=5, seed=7)
    b = _kfold_indices(n=50, k=5, seed=7)
    assert a == b


def test_kfold_balanced():
    folds = _kfold_indices(n=100, k=5, seed=42)
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= 1
