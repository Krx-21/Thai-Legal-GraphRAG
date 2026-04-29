"""K-fold cross-validation for retrieval/citation metrics.

Splits a QA dataset into K folds and evaluates each fold against a knowledge
graph. Reports mean ± std across folds for every aggregate metric, plus the
worst-performing fold.

Boost-keyword tuning is *not* re-run per fold — the system has been refactored
to remove answer-key memorization, so the dictionary is now considered a fixed
artifact. The script's role is to estimate the variance of measured metrics
across random splits, not to retrain.

Usage:
    python -m scripts.kfold_eval --qa-file output/qa/qa_all.json --k 5 --mode local
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

# Allow `python -m scripts.kfold_eval` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from eval.evaluation import QAPair, evaluate_batch, evaluate_single, load_qa_pairs
from graphrag import llm_utils
from graphrag.graph_builder import KnowledgeGraph
from graphrag.query_engine import answer


def _kfold_indices(n: int, k: int, seed: int) -> list[list[int]]:
    """Return k disjoint index lists covering [0, n)."""
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, idx in enumerate(idxs):
        folds[i % k].append(idx)
    return folds


def _eval_fold(qa_pairs: list[QAPair], kg: KnowledgeGraph, mode: str) -> dict[str, float]:
    """Run evaluation over a single fold and return aggregate metrics."""
    results = []
    for qa in qa_pairs:
        t0 = time.time()
        res = answer(qa.question, kg, mode=mode)
        elapsed = time.time() - t0
        sr = res["search_result"]
        section_ents = [e for e in sr.entities if e.get("type") == "SECTION"]
        section_ents.sort(key=lambda e: e.get("score", 0) or 0.0, reverse=True)
        retrieved: list[str] = []
        for ed in section_ents[:20]:
            name = ed.get("name", "")
            retrieved.extend(re.findall(r"มาตรา\s*(\d+(?:/\d+)?)", name))
        results.append(evaluate_single(qa, res["answer"], retrieved, mode, elapsed))
    return evaluate_batch(results)


def _summarise(per_fold: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Compute mean / std / min / max for each metric across folds."""
    keys = [k for k, v in per_fold[0].items() if isinstance(v, (int, float))]
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = [f[k] for f in per_fold]
        out[k] = {
            "mean": statistics.fmean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="K-fold eval for GraphRAG")
    ap.add_argument("--qa-file", default=str(config.QA_ALL))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="local",
                    choices=["local", "global", "hybrid"])
    ap.add_argument("--out", default=None,
                    help="Optional JSON path to save fold-level results")
    args = ap.parse_args()

    qa_path = Path(args.qa_file)
    if not qa_path.exists():
        print(f"QA file not found: {qa_path}")
        return 1

    qa_pairs = load_qa_pairs(qa_path)
    n = len(qa_pairs)
    if args.k < 2 or args.k > n:
        print(f"Invalid k={args.k} for n={n}")
        return 1

    print(f"K-fold eval: n={n}, k={args.k}, mode={args.mode}, seed={args.seed}")
    print(f"LLM judge: {'ON' if llm_utils.is_available() else 'OFF (semantic+kw fallback)'}")

    kg = KnowledgeGraph()
    kg.load_parquet()
    kg.compute_embeddings()

    folds = _kfold_indices(n, args.k, args.seed)
    per_fold: list[dict[str, float]] = []
    for i, fold_idxs in enumerate(folds, 1):
        fold_qas = [qa_pairs[j] for j in fold_idxs]
        print(f"\n[Fold {i}/{args.k}] n={len(fold_qas)}")
        agg = _eval_fold(fold_qas, kg, args.mode)
        per_fold.append(agg)
        for k, v in agg.items():
            if isinstance(v, float):
                print(f"   {k:>20s}: {v:.3f}")

    summary = _summarise(per_fold)
    print(f"\n{'='*60}")
    print(f"  K-fold summary (k={args.k}, mode={args.mode})")
    print(f"{'='*60}")
    print(f"  {'metric':<22}{'mean':>8}{'+/- std':>10}{'min':>8}{'max':>8}")
    for metric, stats in summary.items():
        print(f"  {metric:<22}{stats['mean']:>8.3f}{stats['std']:>10.3f}"
              f"{stats['min']:>8.3f}{stats['max']:>8.3f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            {"per_fold": per_fold, "summary": summary,
             "config": {"n": n, "k": args.k, "mode": args.mode, "seed": args.seed}},
            ensure_ascii=False, indent=2,
        ))
        print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
