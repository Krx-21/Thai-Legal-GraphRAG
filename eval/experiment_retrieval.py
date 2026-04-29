"""Retrieval Strategy Comparison Experiment

Tests multiple retrieval strategies on NitiBench-CCL and produces
a comparison table. All strategies operate on the same entity pool.

Strategies:
  1. TF-IDF (baseline)
  2. TF-IDF + [แพ่ง] filter
  3. BM25
  4. BM25 + [แพ่ง] filter
  5. Dense (sentence-transformers)
  6. Dense + [แพ่ง] filter
  7. TF-IDF + BM25 fusion
  8. Dense + BM25 fusion

Metrics per strategy: hit_rate, MRR, recall@k, precision@k, cit_f1
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from graphrag.graph_builder import KnowledgeGraph
from eval.eval_nitibench import (
    NitiBenchItem,
    load_nitibench_ccl,
    extract_gt_sections,
    _normalize_sec,
    _SEC_PAT,
)


# ── Strategy results ─────────────────────────────────────────────────

@dataclass
class StrategyResult:
    strategy: str
    n: int
    hit_rate: float
    mrr: float
    recall_at_10: float
    precision_at_10: float
    recall_at_20: float
    cit_macro_f1: float        # F1 using top-10 as "citations"
    avg_latency: float


# ── Build retrieval indices ──────────────────────────────────────────

class EntityIndex:
    """Pre-built indices for all retrieval strategies over a KnowledgeGraph."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

        # Collect entity data
        self.names: list[str] = []
        self.types: list[str] = []
        self.descriptions: list[str] = []
        self.texts: list[str] = []  # type: name - description

        for name, ent in kg.entities.items():
            self.names.append(name)
            self.types.append(ent.entity_type)
            self.descriptions.append(ent.description)
            self.texts.append(f"{ent.entity_type}: {name} - {ent.description}")

        self.n = len(self.names)
        print(f"[EntityIndex] {self.n} entities")

        # Pre-tokenize with pythainlp
        try:
            from pythainlp.tokenize import word_tokenize
            self._tokenize = lambda t: word_tokenize(t, engine="newmm")
        except ImportError:
            self._tokenize = lambda t: t.split()

        self.tokenized_texts = [self._tokenize(t) for t in self.texts]

        # Build indices
        self._build_tfidf()
        self._build_bm25()
        self._build_dense()

        # [แพ่ง] mask
        self.paeng_mask = np.array(["[แพ่ง]" in n for n in self.names])
        print(f"[EntityIndex] [แพ่ง] entities: {self.paeng_mask.sum()}")

    def _build_tfidf(self):
        """Build TF-IDF matrix from entity texts."""
        print("[EntityIndex] Building TF-IDF index ...")
        from sklearn.feature_extraction.text import TfidfVectorizer

        joined = [" ".join(toks) for toks in self.tokenized_texts]
        self.tfidf_vec = TfidfVectorizer(max_features=512)
        self.tfidf_matrix = self.tfidf_vec.fit_transform(joined).toarray().astype(np.float32)
        print(f"  TF-IDF dim={self.tfidf_matrix.shape[1]}")

    def _build_bm25(self):
        """Build BM25 index from entity texts."""
        print("[EntityIndex] Building BM25 index ...")
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.tokenized_texts)
        print("  BM25 ready")

    def _build_dense(self):
        """Build dense embeddings using sentence-transformers."""
        print("[EntityIndex] Building dense embeddings (this may take a minute) ...")
        from sentence_transformers import SentenceTransformer

        # Use a lightweight multilingual model
        self.dense_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.dense_matrix = self.dense_model.encode(
            self.texts, show_progress_bar=True, batch_size=64,
            normalize_embeddings=True,
        ).astype(np.float32)
        print(f"  Dense dim={self.dense_matrix.shape[1]}")

    # ── Query methods ────────────────────────────────────────────────

    def query_tfidf(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return [(entity_name, score)] ranked by TF-IDF cosine similarity."""
        q_tok = " ".join(self._tokenize(query))
        q_vec = self.tfidf_vec.transform([q_tok]).toarray().astype(np.float32)
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.names[i], float(sims[i])) for i in top_idx]

    def query_bm25(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return [(entity_name, score)] ranked by BM25."""
        q_toks = self._tokenize(query)
        scores = self.bm25.get_scores(q_toks)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.names[i], float(scores[i])) for i in top_idx]

    def query_dense(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return [(entity_name, score)] ranked by dense cosine similarity."""
        q_vec = self.dense_model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        sims = cosine_similarity(q_vec, self.dense_matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.names[i], float(sims[i])) for i in top_idx]

    def query_tfidf_bm25_fusion(self, query: str, top_k: int = 100,
                                 alpha: float = 0.5) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion of TF-IDF + BM25."""
        tfidf_ranked = self.query_tfidf(query, top_k=200)
        bm25_ranked = self.query_bm25(query, top_k=200)
        return self._rrf_fuse(tfidf_ranked, bm25_ranked, top_k)

    def query_dense_bm25_fusion(self, query: str, top_k: int = 100,
                                 alpha: float = 0.5) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion of Dense + BM25."""
        dense_ranked = self.query_dense(query, top_k=200)
        bm25_ranked = self.query_bm25(query, top_k=200)
        return self._rrf_fuse(dense_ranked, bm25_ranked, top_k)

    @staticmethod
    def _rrf_fuse(
        list_a: list[tuple[str, float]],
        list_b: list[tuple[str, float]],
        top_k: int,
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion (RRF) of two ranked lists."""
        scores: dict[str, float] = {}
        for rank, (name, _) in enumerate(list_a, 1):
            scores[name] = scores.get(name, 0.0) + 1.0 / (k + rank)
        for rank, (name, _) in enumerate(list_b, 1):
            scores[name] = scores.get(name, 0.0) + 1.0 / (k + rank)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def filter_paeng(
        self, results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """Keep only [แพ่ง] entities from a ranked result list."""
        return [(name, score) for name, score in results if "[แพ่ง]" in name]


# ── Section extraction from ranked results ───────────────────────────

def ranked_sections(results: list[tuple[str, float]]) -> list[str]:
    """Extract section numbers from ranked entity names, preserving order."""
    sections = []
    seen = set()
    for name, _ in results:
        m = _SEC_PAT.search(name)
        if m:
            sec = _normalize_sec(m.group(1))
            if sec not in seen:
                seen.add(sec)
                sections.append(sec)
    return sections


# ── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(
    gt_sections: list[str],
    pred_sections: list[str],
    k_cit: int = 10,
    k_recall: int = 20,
) -> dict:
    """Compute retrieval metrics for one query."""
    gt = set(gt_sections)
    if not gt:
        return {"hit": 0, "mrr": 0.0, "recall@10": 0.0, "prec@10": 0.0,
                "recall@20": 0.0, "f1@10": 0.0}

    # Hit & MRR
    hit = 0
    mrr = 0.0
    for i, s in enumerate(pred_sections, 1):
        if s in gt:
            hit = 1
            mrr = 1.0 / i
            break

    # Recall@k
    top10 = set(pred_sections[:k_cit])
    top20 = set(pred_sections[:k_recall])
    prec_10 = len(gt & top10) / k_cit if k_cit > 0 else 0.0
    rec_10 = len(gt & top10) / len(gt)
    rec_20 = len(gt & top20) / len(gt)
    f1_10 = (2 * prec_10 * rec_10 / (prec_10 + rec_10)
             if (prec_10 + rec_10) > 0 else 0.0)

    return {"hit": hit, "mrr": mrr, "recall@10": rec_10, "prec@10": prec_10,
            "recall@20": rec_20, "f1@10": f1_10}


# ── Run experiment ───────────────────────────────────────────────────

STRATEGIES = [
    ("TF-IDF",               "tfidf",           False),
    ("TF-IDF + [แพ่ง]",      "tfidf",           True),
    ("BM25",                  "bm25",            False),
    ("BM25 + [แพ่ง]",        "bm25",            True),
    ("Dense (MiniLM)",        "dense",           False),
    ("Dense + [แพ่ง]",       "dense",           True),
    ("TF-IDF+BM25 RRF",      "tfidf_bm25",      False),
    ("TF-IDF+BM25 + [แพ่ง]", "tfidf_bm25",      True),
    ("Dense+BM25 RRF",       "dense_bm25",      False),
    ("Dense+BM25 + [แพ่ง]",  "dense_bm25",      True),
]


def run_experiment(
    items: list[NitiBenchItem],
    idx: EntityIndex,
) -> list[StrategyResult]:
    """Run all strategies and return summary results."""
    results: list[StrategyResult] = []

    for strat_name, method, use_filter in STRATEGIES:
        print(f"\n{'─'*60}")
        print(f"  Strategy: {strat_name}")
        print(f"{'─'*60}")

        query_fn = {
            "tfidf":      idx.query_tfidf,
            "bm25":       idx.query_bm25,
            "dense":      idx.query_dense,
            "tfidf_bm25": idx.query_tfidf_bm25_fusion,
            "dense_bm25": idx.query_dense_bm25_fusion,
        }[method]

        all_metrics: list[dict] = []
        total_time = 0.0

        for i, item in enumerate(items, 1):
            gt = extract_gt_sections(item)
            if not gt:
                continue

            t0 = time.time()
            ranked = query_fn(item.question, top_k=200)
            if use_filter:
                ranked = idx.filter_paeng(ranked)
            elapsed = time.time() - t0
            total_time += elapsed

            sections = ranked_sections(ranked)
            m = compute_metrics(gt, sections)
            all_metrics.append(m)

            if i % 20 == 0 or i == len(items):
                avg_hit = sum(m["hit"] for m in all_metrics) / len(all_metrics)
                avg_mrr = sum(m["mrr"] for m in all_metrics) / len(all_metrics)
                print(f"  [{i}/{len(items)}] hit={avg_hit:.3f} mrr={avg_mrr:.3f}")

        n = len(all_metrics)
        if n == 0:
            continue

        sr = StrategyResult(
            strategy=strat_name,
            n=n,
            hit_rate=sum(m["hit"] for m in all_metrics) / n,
            mrr=sum(m["mrr"] for m in all_metrics) / n,
            recall_at_10=sum(m["recall@10"] for m in all_metrics) / n,
            precision_at_10=sum(m["prec@10"] for m in all_metrics) / n,
            recall_at_20=sum(m["recall@20"] for m in all_metrics) / n,
            cit_macro_f1=sum(m["f1@10"] for m in all_metrics) / n,
            avg_latency=total_time / n,
        )
        results.append(sr)

    return results


def print_comparison(results: list[StrategyResult]):
    """Print a nice comparison table."""
    print(f"\n{'='*100}")
    print("  RETRIEVAL STRATEGY COMPARISON — NitiBench-CCL (ประมวลกฎหมายแพ่งและพาณิชย์)")
    print(f"{'='*100}")
    header = (
        f"{'Strategy':<28s} {'N':>4s} {'Hit%':>6s} {'MRR':>6s} "
        f"{'R@10':>6s} {'P@10':>6s} {'R@20':>6s} {'F1@10':>6s} {'ms':>6s}"
    )
    print(header)
    print("─" * 100)
    for r in results:
        line = (
            f"{r.strategy:<28s} {r.n:>4d} {r.hit_rate:>6.1%} {r.mrr:>6.3f} "
            f"{r.recall_at_10:>6.1%} {r.precision_at_10:>6.3f} "
            f"{r.recall_at_20:>6.1%} {r.cit_macro_f1:>6.3f} "
            f"{r.avg_latency*1000:>6.1f}"
        )
        print(line)
    print("─" * 100)

    # Find best
    best_hit = max(results, key=lambda r: r.hit_rate)
    best_mrr = max(results, key=lambda r: r.mrr)
    best_f1 = max(results, key=lambda r: r.cit_macro_f1)
    print(f"\n  Best Hit Rate:  {best_hit.strategy} ({best_hit.hit_rate:.1%})")
    print(f"  Best MRR:       {best_mrr.strategy} ({best_mrr.mrr:.3f})")
    print(f"  Best F1@10:     {best_f1.strategy} ({best_f1.cit_macro_f1:.3f})")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retrieval Strategy Comparison")
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--output", default="output/results/retrieval_comparison.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  Retrieval Strategy Comparison Experiment")
    print("=" * 60)

    # 1. Load NitiBench
    print("\n[1] Loading NitiBench-CCL ...")
    items = load_nitibench_ccl(max_items=args.max_items)
    print(f"    {len(items)} items")

    # 2. Load KG
    print("\n[2] Loading knowledge graph ...")
    kg = KnowledgeGraph()
    kg.load_parquet()

    # 3. Build indices
    print("\n[3] Building retrieval indices ...")
    idx = EntityIndex(kg)

    # 4. Run experiment
    print("\n[4] Running strategies ...")
    results = run_experiment(items, idx)

    # 5. Print comparison
    print_comparison(results)

    # 6. Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
