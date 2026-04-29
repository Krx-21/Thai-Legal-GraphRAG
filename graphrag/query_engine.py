"""Query Engine – answer generation with citations."""

from __future__ import annotations

import re

import config
from graphrag import llm_utils
from graphrag.graph_builder import KnowledgeGraph
from graphrag.search_engine import SearchResult, local_search, global_search, hybrid_search


_SYSTEM_PROMPT = (
    "คุณเป็นผู้เชี่ยวชาญด้านกฎหมายไทย ตอบคำถามตามข้อมูลที่ให้เท่านั้น\n"
    "ตอบเป็นภาษาไทย กระชับ และอ้างอิงเฉพาะมาตราที่ปรากฏในบริบท\n"
    "บรรทัดสุดท้ายของคำตอบต้องเป็น `**มาตราที่เกี่ยวข้อง:** มาตรา X, มาตรา Y, ...` "
    "(ระบุเฉพาะหมายเลขมาตราที่ใช้จริง คั่นด้วยจุลภาค)\n"
    "ถ้าไม่มีข้อมูลเพียงพอ ให้ระบุว่าไม่สามารถตอบได้จากข้อมูลที่มี"
)


def answer(
    query: str,
    kg: KnowledgeGraph,
    mode: str = "hybrid",
) -> dict:
    """Generate an answer for *query* using the specified search mode.

    Returns dict with keys: answer, search_result, mode
    """
    # 1. Retrieve context
    if mode == "local":
        search_result = local_search(query, kg)
    elif mode == "global":
        search_result = global_search(query, kg)
    else:
        search_result = hybrid_search(query, kg)

    # 2. Fallback if context is effectively empty
    if len(search_result.context.strip()) < 20:
        return {
            "answer": "ไม่พบข้อมูลที่เกี่ยวข้องในฐานความรู้ กรุณาลองใช้คำถามอื่น",
            "search_result": search_result,
            "mode": mode,
        }

    # 3. Generate answer — try LLM first, fall back to context-based answer
    user_msg = f"## Context\n{search_result.context}\n\n## คำถาม\n{query}"
    try:
        answer_text = llm_utils.chat(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=config.MAX_GENERATION_TOKENS,
        )
        if answer_text.strip():
            return {
                "answer": answer_text,
                "search_result": search_result,
                "mode": mode,
            }
    except Exception:
        pass

    # Fallback: return structured context as the answer (no LLM needed)
    answer_text = _format_context_answer(query, search_result, kg)
    return {
        "answer": answer_text,
        "search_result": search_result,
        "mode": mode,
    }


# ── Citation Extraction ─────────────────────────────────────────────

def extract_cited_sections(text: str) -> set[str]:
    """Extract all section numbers (มาตรา) cited in the answer."""
    return set(re.findall(r"มาตรา\s*(\d+(?:/\d+)?)", text))


def _truncate_at_sentence(text: str, max_len: int = 300) -> str:
    """Truncate text at a sentence boundary near max_len."""
    if len(text) <= max_len:
        return text
    # Cut before the start of a new มาตรา (Thai or Arabic digits)
    for m in re.finditer(r"\s+มาตรา\s+[๐-๙\d]", text):
        pos = m.start()
        if pos > max_len * 0.3 and pos <= max_len:
            return text[:pos].rstrip() + " ..."
        if pos > max_len:
            return text[:pos].rstrip() + " ..."
    # Fallback: cut at last space
    cut = text.rfind(" ", 0, max_len)
    if cut > max_len * 0.5:
        return text[:cut].rstrip() + " ..."
    return text[:max_len] + "..."


def _select_top_sections(sections: list[dict], max_sections: int | None = None) -> list[dict]:
    """Select only the sections that are *clearly* relevant.

    Tunable via env vars (for eval sweeps):
      GRAPHRAG_MAX_SECTIONS   (int,   default 3)
      GRAPHRAG_REL_FLOOR      (float, default 0.55)  ratio of top_score
      GRAPHRAG_GAP_THRESHOLD  (float, default 0.15)  absolute decisive gap

    Rules:
    1. Always keep the top-1.
    2. Keep subsequent sections only if score >= REL_FLOOR * top_score.
    3. Cut at the largest score gap only if gap >= GAP_THRESHOLD.
    4. Hard cap at MAX_SECTIONS.
    """
    import os
    if max_sections is None:
        max_sections = int(os.getenv("GRAPHRAG_MAX_SECTIONS", "3"))
    rel_floor_ratio = float(os.getenv("GRAPHRAG_REL_FLOOR", "0.55"))
    gap_threshold = float(os.getenv("GRAPHRAG_GAP_THRESHOLD", "0.15"))

    if not sections:
        return []
    if len(sections) == 1:
        return sections[:1]

    top_score = sections[0].get("score", 0) or 0.0
    rel_floor = rel_floor_ratio * top_score  # adaptive threshold

    # 1) Keep only entries above the adaptive floor (up to max_sections)
    kept: list[dict] = [sections[0]]
    for s in sections[1:max_sections]:
        if (s.get("score", 0) or 0.0) >= rel_floor:
            kept.append(s)
        else:
            break  # scores are sorted desc; once we drop below, stop

    if len(kept) <= 1:
        return kept

    # 2) Within kept, cut at the largest decisive gap only
    best_gap = 0.0
    best_cut = len(kept)
    for i in range(len(kept) - 1):
        gap = (kept[i].get("score", 0) or 0.0) - (kept[i + 1].get("score", 0) or 0.0)
        if gap > best_gap and gap >= gap_threshold:
            best_gap = gap
            best_cut = i + 1

    return kept[:best_cut]


def _trim_section_desc(name: str, desc: str) -> str:
    """Trim a SECTION description: cut at the first reference to a different มาตรา.

    Handles both Thai and Arabic digit forms.
    """
    _DIG = r"[\d" + _THAI_DIGITS + r"]+(?:/[\d" + _THAI_DIGITS + r"]+)?"
    # Extract this section's own number (normalized to Arabic)
    own_num = _section_num_from_name(name) or ""
    for m in re.finditer(r"\s+มาตรา\s+(" + _DIG + r")", desc):
        if m.start() < 10:
            continue
        hit = m.group(1).translate(_TO_ARABIC)
        if hit != own_num and m.start() > 40:
            return desc[:m.start()].rstrip()
    # Fallback: limit length
    if len(desc) > 1200:
        cut = desc.rfind(" ", 0, 1200)
        return desc[:cut].rstrip() + " ..." if cut > 400 else desc[:1200] + "..."
    return desc


def _query_entity_relevant(query: str, ent: dict) -> bool:
    """Check if an OFFENSE/PENALTY/LEGAL_CONCEPT entity is relevant to the query."""
    q = query.lower()
    name = ent["name"].lower()
    desc = ent.get("description", "").lower()
    # Check keyword overlap between entity name and query
    for word in re.findall(r"[\u0E00-\u0E7F]{3,}", name):
        if word in q:
            return True
    # Check if entity description mentions query keywords
    for word in re.findall(r"[\u0E00-\u0E7F]{3,}", q):
        if word in name or word in desc[:100]:
            return True
    return False


def _extract_chapter(ent: dict) -> str:
    """Extract chapter name from entity's source chunk IDs (format: law__chapter__idx)."""
    for cid in ent.get("source_chunk_ids", []):
        parts = cid.split("__")
        if len(parts) >= 2:
            return parts[1].replace("_", " ")
    return ""


# ── Thai ↔ Arabic digit helpers ───────────────────────────────────
_THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
_TO_ARABIC = str.maketrans(_THAI_DIGITS, "0123456789")


def _section_num_from_name(name: str) -> str | None:
    """Extract 'NNN' (Arabic) from an entity name like '[อาญา] มาตรา 334'."""
    m = re.search(r"มาตรา\s*([\d" + _THAI_DIGITS + r"]+(?:/[\d" + _THAI_DIGITS + r"]+)?)", name)
    return m.group(1).translate(_TO_ARABIC) if m else None


def _resolve_section_text(ent: dict, kg) -> str:
    """Look up the real statute text for a SECTION entity from kg.text_chunks.

    Strategy:
      1. Find this section's number from entity name.
      2. For each source chunk, locate 'มาตรา <num>' in the chunk text.
      3. Return text from that point up to the next 'มาตรา <other_num>' or end.
    Falls back to the original entity description.
    """
    if kg is None:
        return ent.get("description", "")
    own_num = _section_num_from_name(ent["name"])
    if not own_num:
        return ent.get("description", "")

    chunks = getattr(kg, "text_chunks", None) or {}
    # Pattern to find any มาตรา X header (Thai or Arabic digits)
    header_pat = re.compile(r"มาตรา\s*([\d" + _THAI_DIGITS + r"]+(?:/[\d" + _THAI_DIGITS + r"]+)?)")

    for cid in ent.get("source_chunk_ids", []):
        chunk = chunks.get(cid)
        if chunk is None:
            continue
        text = getattr(chunk, "text", "") or ""
        if not text:
            continue
        # Locate start of own section
        start = None
        for m in header_pat.finditer(text):
            hit_num = m.group(1).translate(_TO_ARABIC)
            if hit_num == own_num:
                start = m.start()
                break
        if start is None:
            continue
        # Locate next different-section header
        end = len(text)
        for m in header_pat.finditer(text, start + 4):
            hit_num = m.group(1).translate(_TO_ARABIC)
            if hit_num != own_num:
                end = m.start()
                break
        snippet = text[start:end].strip()
        # Normalize whitespace but preserve paragraph breaks
        snippet = re.sub(r"[ \t]+", " ", snippet)
        snippet = re.sub(r"\n{2,}", "\n", snippet)
        if snippet:
            return snippet

    return ent.get("description", "")


def _penalty_query(query: str) -> bool:
    """True if query asks about penalties."""
    q = query
    return any(k in q for k in ("โทษ", "ระวาง", "จำคุก", "ปรับ", "ประหาร"))


# Real penalty descriptions should contain these statutory phrases
_PENALTY_GOOD = re.compile(
    r"ระวางโทษ|จำคุก|ปรับไม่เกิน|ปรับตั้งแต่|ประหาร|กักขัง|ริบทรัพย์|โทษ"
)
# Noise patterns that indicate regex-extractor over-matched (e.g. "ปรับปรุง")
_PENALTY_NOISE = re.compile(r"ปรับปรุง|พัฒนา|โครงสร้าง|แผน|นโยบาย|ยุทธศาสตร์")


def _valid_penalty_desc(desc: str) -> bool:
    """Penalty description must look statutory, not management jargon."""
    if _PENALTY_NOISE.search(desc):
        return False
    return bool(_PENALTY_GOOD.search(desc))


def _law_boost_for_query(query: str) -> dict[str, float]:
    """Return multiplicative score boost per law prefix based on query signals.

    Helps disambiguate when retrieval pulls sections from the wrong code
    (e.g. Constitution sections dominate for a criminal-law question).
    """
    q = query
    boosts = {"อาญา": 1.0, "แพ่ง": 1.0, "รธน": 1.0}
    # Criminal-law signals
    if any(k in q for k in ("อาญา", "ระวางโทษ", "ระวาง", "จำคุก", "ประหาร",
                            "ลักทรัพย์", "ฉ้อโกง", "ยักยอก", "ทำร้าย", "ฆ่า",
                            "หมิ่น", "ปล้น", "ชิงทรัพย์", "ข่มขืน", "ริบทรัพย์",
                            "บันดาล", "เจตนา", "ประมาท", "ความผิด", "ปรับไม่เกิน")):
        boosts["อาญา"] = 1.25
        boosts["รธน"] = 0.85
    # Civil-law signals
    if any(k in q for k in ("แพ่ง", "นิติกรรม", "สัญญา", "ละเมิด", "มรดก",
                            "บุริมสิทธิ", "อายุความ", "หนี้", "ภูมิลำเนา",
                            "นิติภาวะ", "สาบสูญ", "โมฆะ", "โมฆียะ", "ข่มขู่")):
        boosts["แพ่ง"] = 1.25
        boosts["รธน"] = 0.9
    # Constitutional signals
    if any(k in q for k in ("รัฐธรรมนูญ", "อธิปไตย", "รัฐสภา", "นายก",
                            "พระมหากษัตริย์", "คณะรัฐมนตรี", "เลือกตั้ง",
                            "สิทธิเสรีภาพ")):
        # Only boost if also NOT a penal-king query
        if not any(k in q for k in ("ปลงพระ", "ประทุษร้าย", "หมิ่นพระ")):
            boosts["รธน"] = 1.2
    return boosts


def _apply_law_boost(sections: list[dict], query: str) -> None:
    """In-place: multiply section scores by law-prefix boost from query."""
    boosts = _law_boost_for_query(query)
    for s in sections:
        name = s.get("name", "")
        m = re.match(r"\[([^\]]+)\]", name)
        if m:
            prefix = m.group(1)
            b = boosts.get(prefix, 1.0)
            if b != 1.0:
                s["score"] = (s.get("score", 0) or 0.0) * b


def _format_context_answer(query: str, sr: SearchResult, kg=None) -> str:
    """Build a structured, readable answer from search context (no LLM required)."""
    parts: list[str] = []

    # Collect entities by type
    sections: list[dict] = []
    other_entities: list[dict] = []
    seen = set()

    for ent in sr.entities:
        key = ent["name"]
        if key in seen:
            continue
        seen.add(key)
        if ent["type"] == "SECTION":
            sections.append(ent)
        elif ent["type"] in ("OFFENSE", "PENALTY", "LEGAL_CONCEPT"):
            # Drop PENALTY entries whose description is clearly noise
            if ent["type"] == "PENALTY" and not _valid_penalty_desc(ent.get("description", "")):
                continue
            other_entities.append(ent)

    # Re-rank sections based on law-domain signals in the query (mutates scores)
    _apply_law_boost(sections, query)

    # Sort by score
    sections.sort(key=lambda x: x.get("score", 0), reverse=True)
    other_entities.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_sections = _select_top_sections(sections)

    # High-relevance OFFENSE/PENALTY/LEGAL_CONCEPT — must also be relevant to query
    section_min_score = min((s.get("score", 0) for s in top_sections), default=0.5)
    # For penalty queries, always include PENALTY entities regardless of relevance heuristic
    is_pen_q = _penalty_query(query)
    primary_extras = [
        e for e in other_entities
        if e.get("score", 0) >= section_min_score
        and (_query_entity_relevant(query, e) or (is_pen_q and e["type"] == "PENALTY"))
    ][:4]
    secondary_extras = [
        e for e in other_entities
        if e.get("score", 0) < section_min_score
        and e.get("score", 0) > 0.3
        and (_query_entity_relevant(query, e) or (is_pen_q and e["type"] == "PENALTY"))
    ][:3]

    _TYPE_LABELS = {"OFFENSE": "ความผิด", "PENALTY": "โทษ", "LEGAL_CONCEPT": "หลักกฎหมาย"}

    if primary_extras:
        parts.append("## สาระสำคัญ\n")
        for ent in primary_extras:
            desc = re.sub(r"\s+", " ", ent["description"]).strip()
            label = _TYPE_LABELS.get(ent["type"], ent["type"])
            if len(desc) > 400:
                desc = desc[:400] + "..."
            parts.append(f"**{ent['name']}** ({label})\n{desc}\n")

    # Build citation set — cite EVERY section shown in "บทบัญญัติที่เกี่ยวข้อง"
    # so the citation list is consistent with what the reader sees.
    # (Previously we cited only top-1, which collapsed citation_f1 to ~0 whenever
    # the gold section sat at rank 2-3.)
    cited_sections: list[tuple[str, float]] = []  # (section_num, score)
    for sec in top_sections:
        m = re.search(r"มาตรา\s*(\d+(?:/\d+)?(?:\s*(?:ทวิ|ตรี|จัตวา))?)", sec["name"])
        if m:
            cited_sections.append((m.group(1), sec.get("score", 0) or 0.0))

    if top_sections:
        parts.append("## บทบัญญัติที่เกี่ยวข้อง\n")
        for sec in top_sections:
            # Prefer the real statute text from source chunks over the stub description
            real_text = _resolve_section_text(sec, kg)
            # Strip gazette footnote markers like "[303]"
            real_text = re.sub(r"\[\d+\]", "", real_text)
            desc = re.sub(r"[ \t]+", " ", real_text).strip()
            desc = _trim_section_desc(sec["name"], desc)
            # Render as blockquote; keep line breaks readable
            quoted = "\n".join("> " + ln if ln.strip() else ">" for ln in desc.splitlines())
            chapter_ctx = _extract_chapter(sec)
            header = f"**{sec['name']}**"
            if chapter_ctx:
                header += f"  _({chapter_ctx})_"
            parts.append(f"{header}\n{quoted}\n")

    # Lower-relevance extras
    if secondary_extras:
        parts.append("## ข้อมูลเพิ่มเติม\n")
        for ent in secondary_extras:
            desc = re.sub(r"\s+", " ", ent["description"]).strip()[:200]
            label = _TYPE_LABELS.get(ent["type"], ent["type"])
            parts.append(f"- **{ent['name']}** ({label}): {desc}")

    if cited_sections:
        sorted_secs = sorted(cited_sections, key=lambda x: x[1], reverse=True)
        cite_nums = [s[0] for s in sorted_secs]
        parts.append(f"\n**มาตราที่เกี่ยวข้อง:** {', '.join('มาตรา ' + s for s in cite_nums)}")

    if sr.community_reports:
        # Prefer findings that overlap query keywords; fall back to summary/first findings.
        q_words = set(re.findall(r"[\u0E00-\u0E7F]{3,}", query.lower()))
        shown_reports = False
        report_parts: list[str] = []
        for rpt in sr.community_reports[:3]:
            title = rpt["title"]
            findings = rpt.get("findings", [])
            relevant_findings = [f for f in findings if any(w in f.lower() for w in q_words)]
            # Fallback: if no keyword-matched findings but the report exists,
            # show summary + up to 2 top findings so the user sees SOMETHING.
            if not relevant_findings and findings:
                relevant_findings = findings[:2]
            if relevant_findings:
                report_parts.append(f"**{title}**")
                summary = re.sub(r"\s+", " ", rpt.get("summary", "")).strip()
                if summary:
                    report_parts.append(_truncate_at_sentence(summary, max_len=220))
                for f in relevant_findings[:3]:
                    f_text = re.sub(r"\s+", " ", f).strip()
                    f_text = _truncate_at_sentence(f_text, max_len=220)
                    report_parts.append(f"- {f_text}")
                report_parts.append("")
                shown_reports = True
            elif rpt.get("summary"):
                summary = re.sub(r"\s+", " ", rpt["summary"]).strip()
                summary = _truncate_at_sentence(summary, max_len=220)
                report_parts.append(f"- **{title}**: {summary}")
                shown_reports = True
        if shown_reports:
            parts.append("\n## สรุปจากรายงานกลุ่ม\n")
            parts.extend(report_parts)

    if not top_sections and not primary_extras and not sr.community_reports:
        parts.append("ไม่พบข้อมูลที่ตรงกับคำถามในฐานความรู้")

    return "\n".join(parts)
