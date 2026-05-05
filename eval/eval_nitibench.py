"""NitiBench Adapter — evaluate our GraphRAG system using NitiBench-CCL dataset.

Downloads NitiBench-CCL from HuggingFace, filters questions related to
ประมวลกฎหมายแพ่งและพาณิชย์ (Civil and Commercial Code), runs our system,
and computes NitiBench-style metrics:
  - Citation: micro/macro precision, recall, F1
  - Retrieval: hit_rate, recall, MRR
  - E2E: faithfulness (keyword heuristic), answer_relevancy (keyword heuristic)

Reference: https://github.com/vistec-AI/nitibench
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from graphrag.graph_builder import KnowledgeGraph
from graphrag.query_engine import answer


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class NitiBenchItem:
    """A single NitiBench-CCL test item."""
    idx: int
    question: str
    ground_truth_answer: str
    reference_answer: str
    relevant_laws: list[dict]       # [{"law_name": ..., "section_content": ...}]
    reference_laws: list[dict]      # cross-referenced laws


@dataclass
class NitiBenchResult:
    idx: int
    question: str
    mode: str
    # Retrieval (all entities)
    hit: int
    mrr: float
    retrieval_recall: float
    # Citation (answer line only)
    citation_precision: float
    citation_recall: float
    citation_f1: float
    # E2E
    faithfulness: float
    answer_relevancy: float
    # Meta
    predicted_answer: str
    retrieved_sections: list[str]   # all sections from entities
    cited_sections: list[str]       # sections from answer citation line
    ground_truth_sections: list[str]
    latency: float


# ── Load NitiBench-CCL ───────────────────────────────────────────────

def load_nitibench_ccl(
    law_filter: str = "ประมวลกฎหมายแพ่งและพาณิชย์",
    max_items: int | None = None,
) -> list[NitiBenchItem]:
    """Download NitiBench-CCL from HuggingFace and filter for target law."""
    from datasets import load_dataset

    ds = load_dataset("VISAI-AI/nitibench", split="ccl")

    items: list[NitiBenchItem] = []
    for i, row in enumerate(ds):
        matching_laws = [
            rl for rl in row["relevant_laws"]
            if law_filter in rl.get("law_name", "")
        ]
        if not matching_laws:
            continue
        items.append(NitiBenchItem(
            idx=i,
            question=row["question"],
            ground_truth_answer=row["answer"],
            reference_answer=row.get("reference_answer", ""),
            relevant_laws=row["relevant_laws"],
            reference_laws=row.get("reference_laws", []),
        ))
        if max_items and len(items) >= max_items:
            break

    return items


# ── Section extraction helpers ───────────────────────────────────────

_THAI_NUM = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_SEC_PAT = re.compile(r"มาตรา\s*([\d๐-๙]+(?:/[\d๐-๙]+)?)")


def _normalize_sec(s: str) -> str:
    """Normalize a section number to Arabic digits."""
    return s.translate(_THAI_NUM).strip()


def extract_gt_sections(item: NitiBenchItem) -> list[str]:
    """Extract ground truth section numbers from relevant_laws."""
    sections = []
    for rl in item.relevant_laws:
        content = rl.get("section_content", "")
        m = _SEC_PAT.search(content)
        if m:
            sections.append(_normalize_sec(m.group(1)))
    return list(set(sections))


def extract_predicted_sections(answer_text: str, entities: list[dict],
                               law_prefix: str | None = None,
                               top_k: int | None = 20) -> list[str]:
    """Extract predicted section numbers from entities (for retrieval metrics).

    Ranks SECTION entities by score and returns top-K section numbers from
    the matching law. Excludes 1-hop graph-neighbour entities (which inflate
    the list to thousands) by ranking on score.

    If law_prefix is given (e.g. "[แพ่ง]"), only entities with that
    prefix in their name are considered. If top_k is None, all are returned.
    """
    # Filter to SECTION entities of the target law, sort by score desc
    sec_ents = []
    for ent in entities:
        name = ent.get("name", "")
        if law_prefix and law_prefix not in name:
            continue
        if not _SEC_PAT.search(name):
            continue
        sec_ents.append(ent)
    sec_ents.sort(key=lambda e: e.get("score", 0) or 0.0, reverse=True)
    if top_k is not None:
        sec_ents = sec_ents[:top_k]

    sections = []
    for ent in sec_ents:
        m = _SEC_PAT.search(ent.get("name", ""))
        if m:
            sections.append(_normalize_sec(m.group(1)))
    return list(dict.fromkeys(sections))  # preserve order, deduplicate


def extract_cited_sections(answer_text: str,
                           law_prefix: str | None = None) -> list[str]:
    """Extract sections from the citation line (for citation metrics).

    Looks at the 'มาตราที่เกี่ยวข้อง:' line first (the system's explicit
    citation list). If law_prefix is given, sections are accepted only if
    the body of the answer contains a `[law_prefix] มาตรา X` reference for
    that section number, ensuring we only count citations from the target law.
    """
    # Build set of section numbers that appear in the body with the target prefix
    body_sections: set[str] | None = None
    if law_prefix:
        body_sections = set()
        for m in re.finditer(
            re.escape(law_prefix) + r"\s*มาตรา\s*([\d๐-๙]+(?:/[\d๐-๙]+)?)",
            answer_text,
        ):
            body_sections.add(_normalize_sec(m.group(1)))

    # Try citation line
    for line in answer_text.split("\n"):
        if "มาตราที่เกี่ยวข้อง" in line:
            raw = re.findall(r"(\d+(?:/\d+)?)", line)
            cites = [_normalize_sec(s) for s in raw]
            if body_sections is not None:
                cites = [s for s in cites if s in body_sections]
            return list(dict.fromkeys(cites))

    # Fallback: use only sections referenced in body with target prefix
    if body_sections:
        return list(body_sections)

    # Legacy format fallback: [SECTION] lines
    if law_prefix:
        sections: list[str] = []
        for line in answer_text.split("\n"):
            if "[SECTION]" in line and law_prefix in line:
                m = _SEC_PAT.search(line)
                if m:
                    sections.append(_normalize_sec(m.group(1)))
        return list(dict.fromkeys(sections))

    return []


# ── NitiBench Metrics ────────────────────────────────────────────────

def citation_score_single(
    gt_sections: list[str],
    pred_sections: list[str],
) -> tuple[float, float, float]:
    """Compute citation precision, recall, F1 for a single query."""
    gt = set(gt_sections)
    pred = set(pred_sections)
    if not gt and not pred:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    if not gt:
        return 0.0, 0.0, 0.0
    tp = len(gt & pred)
    precision = tp / len(pred)
    recall = tp / len(gt)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def retrieval_hit(gt_sections: list[str], pred_sections: list[str]) -> int:
    """1 if any ground truth section is in predicted."""
    return 1 if set(gt_sections) & set(pred_sections) else 0


def retrieval_mrr(gt_sections: list[str], pred_sections: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant section."""
    gt = set(gt_sections)
    for i, s in enumerate(pred_sections, 1):
        if s in gt:
            return 1.0 / i
    return 0.0


def keyword_faithfulness(answer_text: str, reference: str) -> float:
    """Simple keyword overlap faithfulness measure."""
    if not reference or not answer_text:
        return 0.0
    ref_words = set(re.findall(r"[\u0E00-\u0E7Fa-zA-Z]+", reference.lower()))
    ans_words = set(re.findall(r"[\u0E00-\u0E7Fa-zA-Z]+", answer_text.lower()))
    if not ref_words:
        return 0.0
    overlap = len(ref_words & ans_words)
    return min(overlap / len(ref_words), 1.0)


def keyword_relevancy(answer_text: str, question: str) -> float:
    """Simple keyword overlap relevancy measure."""
    if not question or not answer_text:
        return 0.0
    q_words = set(re.findall(r"[\u0E00-\u0E7Fa-zA-Z]+", question.lower()))
    a_words = set(re.findall(r"[\u0E00-\u0E7Fa-zA-Z]+", answer_text.lower()))
    if not q_words:
        return 0.0
    overlap = len(q_words & a_words)
    return min(overlap / len(q_words), 1.0)


# ── Run evaluation ───────────────────────────────────────────────────

def run_nitibench_eval(
    items: list[NitiBenchItem],
    kg: KnowledgeGraph,
    mode: str = "hybrid",
    law_prefix: str | None = "[แพ่ง]",
) -> list[NitiBenchResult]:
    """Run our system against NitiBench items and compute metrics.

    law_prefix: if set, only count entities/citations from this law code.
    """
    results: list[NitiBenchResult] = []

    for i, item in enumerate(items, 1):
        t0 = time.time()
        res = answer(item.question, kg, mode=mode)
        elapsed = time.time() - t0

        # Extract sections — two views
        gt_sections = extract_gt_sections(item)
        sr = res["search_result"]
        retrieved_sections = extract_predicted_sections(
            res["answer"], sr.entities, law_prefix=law_prefix
        )
        cited_sections = extract_cited_sections(
            res["answer"], law_prefix=law_prefix
        )

        # Citation metrics (answer-line citations only — measures presentation)
        cit_p, cit_r, cit_f1 = citation_score_single(gt_sections, cited_sections)

        # Retrieval metrics (all entities — measures search quality)
        hit = retrieval_hit(gt_sections, retrieved_sections)
        mrr = retrieval_mrr(gt_sections, retrieved_sections)
        ret_recall = (len(set(gt_sections) & set(retrieved_sections)) / len(gt_sections)
                      if gt_sections else 0.0)

        # E2E heuristics
        faith = keyword_faithfulness(res["answer"], item.reference_answer)
        rel = keyword_relevancy(res["answer"], item.question)

        result = NitiBenchResult(
            idx=item.idx,
            question=item.question,
            mode=mode,
            hit=hit,
            mrr=mrr,
            retrieval_recall=ret_recall,
            citation_precision=cit_p,
            citation_recall=cit_r,
            citation_f1=cit_f1,
            faithfulness=faith,
            answer_relevancy=rel,
            predicted_answer=res["answer"],
            retrieved_sections=retrieved_sections,
            cited_sections=cited_sections,
            ground_truth_sections=gt_sections,
            latency=elapsed,
        )
        results.append(result)

        if i % 10 == 0 or i == len(items):
            avg_hit = sum(r.hit for r in results) / len(results)
            avg_cit_f1 = sum(r.citation_f1 for r in results) / len(results)
            avg_ret_rec = sum(r.retrieval_recall for r in results) / len(results)
            print(f"  [{i}/{len(items)}] hit={avg_hit:.3f} ret_recall={avg_ret_rec:.3f} cit_f1={avg_cit_f1:.3f}")

    return results


def aggregate_results(results: list[NitiBenchResult]) -> dict:
    """Compute NitiBench-style global metrics."""
    n = len(results)
    if n == 0:
        return {}

    # Citation (from answer line): micro and macro
    all_gt = [set(r.ground_truth_sections) for r in results]
    all_cited = [set(r.cited_sections) for r in results]

    cit_micro_tp = sum(len(g & p) for g, p in zip(all_gt, all_cited))
    cit_micro_fp = sum(len(p - g) for g, p in zip(all_gt, all_cited))
    cit_micro_fn = sum(len(g - p) for g, p in zip(all_gt, all_cited))

    cit_micro_prec = cit_micro_tp / (cit_micro_tp + cit_micro_fp) if (cit_micro_tp + cit_micro_fp) > 0 else 0.0
    cit_micro_rec = cit_micro_tp / (cit_micro_tp + cit_micro_fn) if (cit_micro_tp + cit_micro_fn) > 0 else 0.0
    cit_micro_f1 = (2 * cit_micro_prec * cit_micro_rec / (cit_micro_prec + cit_micro_rec)
                    if (cit_micro_prec + cit_micro_rec) > 0 else 0.0)

    cit_macro_prec = sum(r.citation_precision for r in results) / n
    cit_macro_rec = sum(r.citation_recall for r in results) / n
    cit_macro_f1 = sum(r.citation_f1 for r in results) / n

    # Retrieval (from all entities)
    all_retrieved = [set(r.retrieved_sections) for r in results]

    ret_micro_tp = sum(len(g & p) for g, p in zip(all_gt, all_retrieved))
    ret_micro_fp = sum(len(p - g) for g, p in zip(all_gt, all_retrieved))
    ret_micro_fn = sum(len(g - p) for g, p in zip(all_gt, all_retrieved))

    ret_micro_prec = ret_micro_tp / (ret_micro_tp + ret_micro_fp) if (ret_micro_tp + ret_micro_fp) > 0 else 0.0
    ret_micro_rec = ret_micro_tp / (ret_micro_tp + ret_micro_fn) if (ret_micro_tp + ret_micro_fn) > 0 else 0.0
    ret_micro_f1 = (2 * ret_micro_prec * ret_micro_rec / (ret_micro_prec + ret_micro_rec)
                    if (ret_micro_prec + ret_micro_rec) > 0 else 0.0)

    return {
        "n": n,
        # Retrieval (all entities)
        "retrieval_hit_rate": sum(r.hit for r in results) / n,
        "retrieval_mrr": sum(r.mrr for r in results) / n,
        "retrieval_recall": sum(r.retrieval_recall for r in results) / n,
        "retrieval_micro_precision": ret_micro_prec,
        "retrieval_micro_recall": ret_micro_rec,
        "retrieval_micro_f1": ret_micro_f1,
        # Citation (answer line)
        "citation_micro_precision": cit_micro_prec,
        "citation_micro_recall": cit_micro_rec,
        "citation_micro_f1": cit_micro_f1,
        "citation_macro_precision": cit_macro_prec,
        "citation_macro_recall": cit_macro_rec,
        "citation_macro_f1": cit_macro_f1,
        # E2E
        "faithfulness": sum(r.faithfulness for r in results) / n,
        "answer_relevancy": sum(r.answer_relevancy for r in results) / n,
        "avg_latency": sum(r.latency for r in results) / n,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate with NitiBench-CCL")
    parser.add_argument("--max-items", type=int, default=50,
                        help="Max NitiBench items to evaluate (default 50)")
    parser.add_argument("--mode", default="hybrid",
                        choices=["local", "global", "hybrid"],
                        help="Search mode")
    parser.add_argument("--all-modes", action="store_true",
                        help="Run all 3 modes")
    parser.add_argument("--output", default="output/results/nitibench_results.json",
                        help="Output path for results")
    args = parser.parse_args()

    print("=" * 60)
    print("  NitiBench-CCL Evaluation")
    print("=" * 60)

    # Load NitiBench data
    print("\n[1] Loading NitiBench-CCL (ประมวลกฎหมายแพ่งและพาณิชย์) ...")
    items = load_nitibench_ccl(max_items=args.max_items)
    print(f"    Loaded {len(items)} items")

    # Load our knowledge graph
    print("\n[2] Loading knowledge graph ...")
    kg = KnowledgeGraph()
    kg.load_parquet()
    kg.compute_embeddings()

    # Run evaluation
    modes = ["local", "global", "hybrid"] if args.all_modes else [args.mode]
    all_results: list[NitiBenchResult] = []

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"  Mode: {mode}")
        print(f"{'=' * 60}")
        results = run_nitibench_eval(items, kg, mode=mode)
        all_results.extend(results)

        agg = aggregate_results(results)
        print(f"\n  [{mode}] NitiBench Metrics:")
        for k, v in agg.items():
            if isinstance(v, float):
                print(f"    {k:>30s}: {v:.4f}")
            else:
                print(f"    {k:>30s}: {v}")

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {len(all_results)} results -> {out_path}")

    # Overall summary
    if len(modes) > 1:
        print(f"\n{'=' * 60}")
        print("  Overall Summary")
        print(f"{'=' * 60}")
        for mode in modes:
            mode_results = [r for r in all_results if r.mode == mode]
            agg = aggregate_results(mode_results)
            hit = agg.get("retrieval_hit_rate", 0)
            mrr = agg.get("retrieval_mrr", 0)
            ret_rec = agg.get("retrieval_recall", 0)
            cit_f1 = agg.get("citation_macro_f1", 0)
            faith = agg.get("faithfulness", 0)
            print(f"  {mode:>8s}: hit={hit:.3f} mrr={mrr:.3f} ret_recall={ret_rec:.3f} cit_f1={cit_f1:.3f} faith={faith:.3f}")


if __name__ == "__main__":
    main()
