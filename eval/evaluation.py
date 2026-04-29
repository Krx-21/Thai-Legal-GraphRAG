"""Evaluation module – RAGAS-style metrics for Thai Legal GraphRAG.

Metrics:
  Retrieval  → Hit Rate, Context Precision, Context Recall
  Generation → Faithfulness, Answer Relevancy, Citation F1

When LLM is unavailable, Faithfulness and Answer Relevancy use
keyword-overlap heuristics instead of LLM-judge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import config
from graphrag import llm_utils


@dataclass
class QAPair:
    question: str
    ground_truth: str
    relevant_sections: list[str]


@dataclass
class EvalResult:
    question: str
    mode: str
    # Retrieval
    hit_rate: float
    context_precision: float
    context_recall: float
    # Generation
    faithfulness: float
    answer_relevancy: float
    citation_f1: float
    # Meta
    predicted_answer: str = ""
    predicted_sections: list[str] | None = None
    latency_seconds: float = 0.0


# ── Citation F1 ─────────────────────────────────────────────────────

def _extract_sections(text: str) -> set[str]:
    return set(re.findall(r"มาตรา\s*(\d+(?:/\d+)?)", text))


def _extract_cited_sections(answer_text: str) -> set[str]:
    """Extract sections only from the citation line (มาตราที่เกี่ยวข้อง)."""
    for line in answer_text.split("\n"):
        if "มาตราที่เกี่ยวข้อง" in line:
            # Require explicit "มาตรา" prefix to avoid matching B.E. years etc.
            return set(re.findall(r"มาตรา\s*(\d+(?:/\d+)?)", line))
    # Fallback: extract from full text
    return _extract_sections(answer_text)


def citation_f1(predicted: str, ground_truth_sections: list[str]) -> float:
    pred = _extract_cited_sections(predicted)
    gt = set(ground_truth_sections)
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    precision = tp / len(pred)
    recall = tp / len(gt)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _citation_f1_from_lists(pred_sections: list[str], ground_truth_sections: list[str]) -> float:
    pred = set(pred_sections)
    gt = set(ground_truth_sections)
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    precision = tp / len(pred)
    recall = tp / len(gt)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Retrieval Metrics ───────────────────────────────────────────────

def hit_rate(retrieved_sections: list[str], relevant_sections: list[str]) -> float:
    """1 if at least one relevant section was retrieved, else 0."""
    return 1.0 if set(retrieved_sections) & set(relevant_sections) else 0.0


def context_precision(retrieved_sections: list[str], relevant_sections: list[str]) -> float:
    rel = set(relevant_sections)
    if not retrieved_sections:
        return 0.0
    return len(set(retrieved_sections) & rel) / len(retrieved_sections)


def context_recall(retrieved_sections: list[str], relevant_sections: list[str]) -> float:
    rel = set(relevant_sections)
    if not rel:
        return 1.0
    return len(set(retrieved_sections) & rel) / len(rel)


# ── LLM-Judge Metrics ──────────────────────────────────────────────

def _llm_judge(prompt: str) -> float:
    """Ask LLM to score 0.0-1.0. Returns None if LLM unavailable."""
    try:
        resp = llm_utils.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        # Extract first float from response
        match = re.search(r"(\d+\.\d+|\d+)", resp)
        if match:
            val = float(match.group())
            return min(max(val, 0.0), 1.0)
    except Exception:
        pass
    return None


# ── Embedding-based similarity (semantic fallback) ─────────────────

_EMBED_MODEL = None  # lazy-loaded SentenceTransformer

def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        except Exception:
            _EMBED_MODEL = False  # mark as unavailable
    return _EMBED_MODEL or None


def _embed_similarity(text_a: str, text_b: str) -> float | None:
    """Cosine similarity between two texts using multilingual MiniLM.

    Returns a value in [0, 1] (negative cosines clipped to 0), or None if
    the embedding backend is unavailable.
    """
    model = _get_embed_model()
    if model is None:
        return None
    if not text_a.strip() or not text_b.strip():
        return 0.0
    try:
        import numpy as np
        vecs = model.encode([text_a[:2000], text_b[:2000]], convert_to_numpy=True, show_progress_bar=False)
        a, b = vecs[0], vecs[1]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        sim = float(np.dot(a, b) / denom)
        return max(0.0, min(1.0, sim))
    except Exception:
        return None


def faithfulness(predicted: str, ground_truth: str) -> float:
    if llm_utils.is_available():
        prompt = f"""ให้คะแนนความถูกต้อง (faithfulness) ของคำตอบเทียบกับเฉลย 0.0-1.0
- 1.0 = คำตอบตรงกับเฉลยทั้งหมด ไม่มีข้อมูลผิด
- 0.5 = คำตอบถูกบางส่วน
- 0.0 = คำตอบผิดทั้งหมดหรือไม่เกี่ยวข้อง

คำตอบ: {predicted[:2000]}

เฉลย: {ground_truth[:2000]}

ตอบเป็นตัวเลข 0.0-1.0 เท่านั้น:"""
        score = _llm_judge(prompt)
        if score is not None:
            return score
    # Semantic fallback: embedding cosine, blended with keyword overlap.
    # Embedding captures paraphrase; overlap penalizes hallucinated specifics.
    emb = _embed_similarity(predicted, ground_truth)
    kw = _keyword_overlap(predicted, ground_truth)
    if emb is None:
        return kw
    return 0.6 * emb + 0.4 * kw


def answer_relevancy(predicted: str, question: str) -> float:
    if llm_utils.is_available():
        prompt = f"""ให้คะแนนความเกี่ยวข้อง (relevancy) ของคำตอบกับคำถาม 0.0-1.0
- 1.0 = คำตอบตรงประเด็น ตอบคำถามได้ครบถ้วน
- 0.5 = คำตอบเกี่ยวข้องบ้าง แต่ไม่ครบ
- 0.0 = คำตอบไม่เกี่ยวข้องกับคำถาม

คำถาม: {question}

คำตอบ: {predicted[:2000]}

ตอบเป็นตัวเลข 0.0-1.0 เท่านั้น:"""
        score = _llm_judge(prompt)
        if score is not None:
            return score
    # Semantic fallback. Strip echoed question line first to avoid leakage.
    answer_body = re.sub(r"^คำถาม:.*\n?", "", predicted, count=1).strip()
    emb = _embed_similarity(answer_body, question)
    kw = _keyword_overlap(answer_body, question)
    if emb is None:
        return kw
    return 0.6 * emb + 0.4 * kw


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Token-overlap score. Used as a component of LLM-judge fallback for
    faithfulness / relevancy.

    Score = |A ∩ B| / |B| (asymmetric: how much of B appears in A), so when
    B is the question/ground-truth we measure coverage rather than verbosity.
    Section numbers are added as pseudo-tokens (`§334`) so citations count
    even when the prose phrasing differs.
    """
    a_tok = _tokens(text_a)
    b_tok = _tokens(text_b)
    if not a_tok or not b_tok:
        return 0.0
    overlap = len(a_tok & b_tok)
    return min(overlap / max(len(b_tok), 1), 1.0)


# Stop-word set for `_keyword_overlap` (module-level so we don't rebuild per
# call). Three categories:
#   1. Function words / particles — universal Thai noise.
#   2. Nominalizers ("การ", "ความ") — strip so they don't dominate overlap.
#   3. Template / markdown tokens emitted by the answer formatter or newmm
#      tokenizer when fed markdown ("##", "**", whitespace, punctuation, and
#      generic phrasing like "บทบัญญัติ", "เกี่ยวข้อง" that appear in every
#      answer regardless of question).
# Words like "ลักทรัพย์", "ฉ้อโกง", "อายุความ", "นิติกรรม" are *not* listed
# here — they carry legal meaning and must contribute to overlap.
_STOP_WORDS: set[str] = {
    # function words
    "ของ", "ใน", "ที่", "และ", "หรือ", "ได้", "ไม่", "เป็น", "มี",
    "แห่ง", "ตาม", "โดย", "ให้", "กับ", "จาก", "ถึง", "แต่", "นั้น",
    "นี้", "จะ", "ต้อง", "อย่าง", "ไว้", "เมื่อ", "ผู้", "แก่", "ว่า",
    "ซึ่ง", "อัน", "ก็", "กัน", "ดัง", "คือ", "เพื่อ", "หาก", "ถ้า",
    "เช่น", "เพียง", "ยัง", "อาจ", "ทั้ง",
    # nominalizers
    "การ", "ความ",
    # template / markdown / formatter noise
    "##", "**", "***", "---", "[", "]", "(", ")", ":", " ", "\n", "\t",
    "บทบัญญัติ", "เกี่ยวข้อง", "ข้อมูล", "เพิ่มเติม", "สรุป", "อธิบาย",
    "หมายความ", "ดังนี้", "ดังกล่าว",
}


def _tokens(text: str) -> set[str]:
    """Tokenize Thai text and append section pseudo-tokens.

    Returns the set of content tokens after stop-word filtering. A section
    citation like "มาตรา 334" yields "§334" so numeric matches survive even
    if the surrounding wording diverges.
    """
    try:
        from pythainlp.tokenize import word_tokenize
        words = set(word_tokenize(text, engine="newmm"))
    except ImportError:
        words = set(re.findall(r"[\u0E00-\u0E7F]{2,}", text))
    sections = set(re.findall(r"มาตรา\s*(\d+(?:/\d+)?)", text))
    words.update(f"§{s}" for s in sections)
    return {w for w in words if len(w) >= 2 and w not in _STOP_WORDS}


# ── Full Evaluation ─────────────────────────────────────────────────

def evaluate_single(
    qa: QAPair,
    predicted_answer: str,
    retrieved_sections: list[str],
    mode: str,
    latency: float = 0.0,
) -> EvalResult:
    # Use sections from the citation line for citation F1
    pred_sections = list(_extract_cited_sections(predicted_answer))
    return EvalResult(
        question=qa.question,
        mode=mode,
        hit_rate=hit_rate(retrieved_sections, qa.relevant_sections),
        context_precision=context_precision(retrieved_sections, qa.relevant_sections),
        context_recall=context_recall(retrieved_sections, qa.relevant_sections),
        faithfulness=faithfulness(predicted_answer, qa.ground_truth),
        answer_relevancy=answer_relevancy(predicted_answer, qa.question),
        citation_f1=_citation_f1_from_lists(pred_sections, qa.relevant_sections),
        predicted_answer=predicted_answer,
        predicted_sections=pred_sections,
        latency_seconds=latency,
    )


def evaluate_batch(
    results: list[EvalResult],
) -> dict:
    """Aggregate metrics across all evaluation results."""
    n = len(results)
    if n == 0:
        return {}
    return {
        "n": n,
        "hit_rate": sum(r.hit_rate for r in results) / n,
        "context_precision": sum(r.context_precision for r in results) / n,
        "context_recall": sum(r.context_recall for r in results) / n,
        "faithfulness": sum(r.faithfulness for r in results) / n,
        "answer_relevancy": sum(r.answer_relevancy for r in results) / n,
        "citation_f1": sum(r.citation_f1 for r in results) / n,
        "avg_latency": sum(r.latency_seconds for r in results) / n,
    }


def save_results(results: list[EvalResult], path: Path | None = None):
    path = path or config.RESULTS_DIR / "eval_results.json"
    data = [asdict(r) for r in results]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Eval] Saved {len(results)} results -> {path}")


def load_qa_pairs(path: Path) -> list[QAPair]:
    """Load QA pairs from JSON file.

    Expected format:
    [
      {
        "question": "...",
        "ground_truth": "...",
        "relevant_sections": ["334", "335"]
      }
    ]
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [QAPair(**item) for item in data]
