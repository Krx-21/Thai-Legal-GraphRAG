"""Regex-based entity & relationship extraction for Thai legal texts.

Extracts structured knowledge from Thai law documents using pattern matching
instead of LLM calls. Works well because Thai legal texts follow consistent
formatting conventions.

Fixes applied over v1:
  - Section IDs are prefixed with law abbreviation to avoid cross-law merge
  - PART_OF / ADJACENT_TO edges are deduplicated per chunk
  - PENALTY names are cleaned (no newlines, capped length)
  - CONCEPT / ORG / PERSON_TYPE link to *nearest* section, not just first
  - OFFENSE regex-captured names are normalized via keyword dict
"""

from __future__ import annotations

import re
from graphrag.preprocessor import TextChunk

# ═══════════════════════════════════════════════════════════════════════
# Patterns
# ═══════════════════════════════════════════════════════════════════════

# มาตรา with optional ทวิ/ตรี/จัตวา suffix and [footnote]
_SECTION_PAT = re.compile(
    r"มาตรา\s*([\u0E50-\u0E59\d]+(?:/[\u0E50-\u0E59\d]+)?(?:\s*(?:ทวิ|ตรี|จัตวา))?)"
)

# Section header at start of line (not inline cross-references)
_SECTION_HEADER_PAT = re.compile(
    r"(?:^|\n)\s*มาตรา\s*[\u0E50-\u0E59\d]+(?:/[\u0E50-\u0E59\d]+)?(?:\s*(?:ทวิ|ตรี|จัตวา))?"
)

# Penalty clauses — match the full penalty description
_PENALTY_PATTERNS = [
    re.compile(r"ต้องระวางโทษ[^.]*?ประหารชีวิต"),
    re.compile(r"ต้องระวางโทษ[^.]*?จำคุก[^.]*?(?:ปี|เดือน|ตลอดชีวิต)[^.]*"),
    re.compile(r"ต้องระวางโทษ[^.]*?ปรับ[^.]*?บาท[^.]*"),
    re.compile(r"จำคุก(?:ไม่เกิน|ตั้งแต่|ตลอดชีวิต)[^.]*?(?:ปี|เดือน)[^.]*"),
    re.compile(r"ปรับ(?:ไม่เกิน|ตั้งแต่)[^.]*?บาท"),
]

# Offense naming patterns — capture the offense name
_OFFENSE_PAT = re.compile(
    r"กระทำความผิดฐาน([\u0E00-\u0E7F\s]+?)(?:\s+ต้องระวาง|$|[\n,])"
)
# Extended pattern to grab offense + penalty context
_OFFENSE_CONTEXT_PAT = re.compile(
    r"กระทำความผิดฐาน[\u0E00-\u0E7F\s]+?ต้องระวาง[^\n]{0,200}"
)

# Cross-references to other sections
_XREF_PATTERNS = [
    re.compile(r"(?:ตาม|ใน|แห่ง|ดังที่บัญญัติไว้ใน)?มาตรา\s*([\u0E50-\u0E59\d]+(?:/[\u0E50-\u0E59\d]+)?(?:\s*(?:ทวิ|ตรี|จัตวา))?)"),
]

# Common legal concepts (keyword → normalized name)
_LEGAL_CONCEPTS: dict[str, str] = {
    "ละเมิด": "ละเมิด",
    "สัญญา": "สัญญา",
    "นิติกรรม": "นิติกรรม",
    "อายุความ": "อายุความ",
    "ค่าสินไหมทดแทน": "ค่าสินไหมทดแทน",
    "ค่าเสียหาย": "ค่าเสียหาย",
    "สิทธิ": "สิทธิ",
    "เสรีภาพ": "เสรีภาพ",
    "ทรัพย์สิน": "ทรัพย์สิน",
    "กรรมสิทธิ์": "กรรมสิทธิ์",
    "ครอบครอง": "การครอบครอง",
    "มรดก": "มรดก",
    "จำนอง": "จำนอง",
    "จำนำ": "จำนำ",
    "ค้ำประกัน": "ค้ำประกัน",
    "โอนสิทธิ": "การโอนสิทธิ",
    "หนี้": "หนี้",
    "เจตนา": "เจตนา",
    "ประมาท": "ความประมาท",
    "ทุจริต": "ทุจริต",
    "สุจริต": "สุจริต",
    "ข่มขืนใจ": "การข่มขืนใจ",
    "กำลังประทุษร้าย": "การใช้กำลังประทุษร้าย",
    "ยินยอม": "ความยินยอม",
    "ป้องกัน": "การป้องกัน",
    "จำเป็น": "ความจำเป็น",
    "บรรเทาโทษ": "การบรรเทาโทษ",
    "รอการลงโทษ": "การรอการลงโทษ",
    "ริบทรัพย์": "การริบทรัพย์",
}

# Organizations
_ORGANIZATIONS: dict[str, str] = {
    "ศาล": "ศาล",
    "ศาลยุติธรรม": "ศาลยุติธรรม",
    "ศาลปกครอง": "ศาลปกครอง",
    "ศาลทหาร": "ศาลทหาร",
    "ศาลรัฐธรรมนูญ": "ศาลรัฐธรรมนูญ",
    "ศาลฎีกา": "ศาลฎีกา",
    "ศาลอุทธรณ์": "ศาลอุทธรณ์",
    "คณะรัฐมนตรี": "คณะรัฐมนตรี",
    "รัฐสภา": "รัฐสภา",
    "สภาผู้แทนราษฎร": "สภาผู้แทนราษฎร",
    "วุฒิสภา": "วุฒิสภา",
    "คณะกรรมการการเลือกตั้ง": "คณะกรรมการการเลือกตั้ง",
    "คณะกรรมการป้องกันและปราบปรามการทุจริตแห่งชาติ": "คณะกรรมการ ป.ป.ช.",
    "สำนักงานตำรวจแห่งชาติ": "สำนักงานตำรวจแห่งชาติ",
    "อัยการ": "อัยการ",
    "พนักงานสอบสวน": "พนักงานสอบสวน",
}

# Person types
_PERSON_TYPES: dict[str, str] = {
    "ผู้เสียหาย": "ผู้เสียหาย",
    "ผู้กระทำผิด": "ผู้กระทำผิด",
    "ผู้ต้องหา": "ผู้ต้องหา",
    "จำเลย": "จำเลย",
    "โจทก์": "โจทก์",
    "ผู้เยาว์": "ผู้เยาว์",
    "นายจ้าง": "นายจ้าง",
    "ลูกจ้าง": "ลูกจ้าง",
    "เจ้าหนี้": "เจ้าหนี้",
    "ลูกหนี้": "ลูกหนี้",
    "ผู้ซื้อ": "ผู้ซื้อ",
    "ผู้ขาย": "ผู้ขาย",
    "ผู้ให้เช่า": "ผู้ให้เช่า",
    "ผู้เช่า": "ผู้เช่า",
    "ผู้รับจำนอง": "ผู้รับจำนอง",
    "ผู้จำนอง": "ผู้จำนอง",
    "ตัวการ": "ตัวการ",
    "ตัวแทน": "ตัวแทน",
    "ผู้ค้ำประกัน": "ผู้ค้ำประกัน",
    "เจ้าของ": "เจ้าของ",
    "ผู้ครอง": "ผู้ครอง",
    "ผู้อนุบาล": "ผู้อนุบาล",
    "ผู้พิทักษ์": "ผู้พิทักษ์",
    "บิดามารดา": "บิดามารดา",
    "ทายาท": "ทายาท",
    "เจ้าพนักงาน": "เจ้าพนักงาน",
    "พระมหากษัตริย์": "พระมหากษัตริย์",
}

# Common offense names (for direct matching)
_OFFENSE_KEYWORDS: dict[str, str] = {
    "ลักทรัพย์": "ความผิดฐานลักทรัพย์",
    "วิ่งราวทรัพย์": "ความผิดฐานวิ่งราวทรัพย์",
    "ชิงทรัพย์": "ความผิดฐานชิงทรัพย์",
    "ปล้นทรัพย์": "ความผิดฐานปล้นทรัพย์",
    "กรรโชก": "ความผิดฐานกรรโชก",
    "รีดเอาทรัพย์": "ความผิดฐานรีดเอาทรัพย์",
    "ฉ้อโกง": "ความผิดฐานฉ้อโกง",
    "ยักยอก": "ความผิดฐานยักยอก",
    "รับของโจร": "ความผิดฐานรับของโจร",
    "ทำให้เสียทรัพย์": "ความผิดฐานทำให้เสียทรัพย์",
    "บุกรุก": "ความผิดฐานบุกรุก",
    "ฆ่าผู้อื่น": "ความผิดฐานฆ่าผู้อื่น",
    "ฆ่าคนตาย": "ความผิดฐานฆ่าคนตาย",
    "ทำร้ายร่างกาย": "ความผิดฐานทำร้ายร่างกาย",
    "ทำร้ายจนเป็นเหตุให้ถึงแก่ความตาย": "ความผิดฐานทำร้ายเป็นเหตุให้ตาย",
    "ข่มขืนกระทำชำเรา": "ความผิดฐานข่มขืนกระทำชำเรา",
    "หมิ่นประมาท": "ความผิดฐานหมิ่นประมาท",
    "หมิ่นพระบรมเดชานุภาพ": "ความผิดฐานหมิ่นพระบรมเดชานุภาพ",
    "กบฏ": "ความผิดฐานกบฏ",
    "ปลอมเอกสาร": "ความผิดฐานปลอมเอกสาร",
    "ปลอมแปลง": "ความผิดฐานปลอมแปลง",
    "ทุจริตต่อหน้าที่": "ความผิดฐานทุจริตต่อหน้าที่",
    "หน่วงเหนี่ยว": "ความผิดฐานหน่วงเหนี่ยว",
    "กักขัง": "ความผิดฐานกักขังหน่วงเหนี่ยว",
    "วางเพลิง": "ความผิดฐานวางเพลิง",
}

# Chapter / topic headers
_CHAPTER_PAT = re.compile(
    r"(?:หมวด|ลักษณะ|ภาค|บรรพ)\s*[\u0E50-\u0E59\d]+\s*\n?([\u0E00-\u0E7F\s]+?)(?:\n|$)"
)


# ═══════════════════════════════════════════════════════════════════════
# Thai numeral helpers
# ═══════════════════════════════════════════════════════════════════════

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# Short abbreviation used as prefix in section IDs
_LAW_ABBREV: dict[str, str] = {
    "ประมวลกฎหมายอาญา":                 "อาญา",
    "ประมวลกฎหมายแพ่งและพาณิชย์":       "แพ่ง",
    "รัฐธรรมนูญแห่งราชอาณาจักรไทย":     "รธน",
    "รัฐธรรมนูญแห่งราชอาณาจักรไทย พุทธศักราช ๒๕๖๐": "รธน",
}


def _thai_to_arabic(s: str) -> str:
    """Convert Thai numeral string to Arabic."""
    return s.translate(_THAI_DIGITS)


def _law_abbr(law_name: str) -> str:
    """Return short abbreviation for a law name."""
    for key, abbr in _LAW_ABBREV.items():
        if key in law_name:
            return abbr
    return law_name[:4]  # fallback: first 4 chars


def _normalize_section(s: str, law_name: str = "") -> str:
    """Normalize section: '๓๓๔' → '[อาญา] มาตรา 334'."""
    base = "มาตรา " + _thai_to_arabic(s.strip())
    if law_name:
        return f"[{_law_abbr(law_name)}] {base}"
    return base


# ═══════════════════════════════════════════════════════════════════════
# Extraction
# ═══════════════════════════════════════════════════════════════════════

def _extract_penalty_text(text: str) -> list[str]:
    """Extract all penalty descriptions from text."""
    penalties: list[str] = []
    for pat in _PENALTY_PATTERNS:
        for m in pat.finditer(text):
            p = m.group(0).strip()
            # Clean: remove newlines and cap length
            p = re.sub(r"\s+", " ", p)
            if p and len(p) > 5 and p not in penalties:
                penalties.append(p)
    return penalties


def _normalize_offense_name(raw: str) -> str | None:
    """Normalize a regex-captured offense name via the keyword dict.
    
    Returns the canonical name if it matches a known offense keyword,
    or constructs a cleaned name.  Returns None if too short / junk.
    """
    raw = raw.strip()
    if len(raw) < 3 or len(raw) > 60:
        return None
    # Check if the raw text matches any known keyword → use canonical name
    for keyword, canonical in _OFFENSE_KEYWORDS.items():
        if keyword in raw:
            return canonical
    return f"ความผิดฐาน{raw}"


def extract_from_chunk(chunk: TextChunk) -> tuple[list[dict], list[dict]]:
    """Extract entities and relationships from a single TextChunk.

    Returns:
        (entities, relationships) where each is a list of dicts
        matching the format used by graph_builder.
    """
    text = chunk.text
    chunk_id = chunk.chunk_id
    entities: dict[str, dict] = {}   # name → entity dict
    relationships: list[dict] = []
    _seen_edges: set[tuple[str, str, str]] = set()  # (src, tgt, type) dedup

    def _add_rel(src: str, tgt: str, rtype: str, desc: str,
                 weight: float = 1.0) -> None:
        key = (src, tgt, rtype)
        if key not in _seen_edges:
            _seen_edges.add(key)
            relationships.append({
                "source": src, "target": tgt, "type": rtype,
                "description": desc, "weight": weight,
                "source_chunk_ids": [chunk_id],
            })

    # Determine the law name from chunk fields
    law_name = chunk.law_name or ""
    if not law_name:
        chapter = chunk.chapter or ""
        if "อาญา" in chapter:
            law_name = "ประมวลกฎหมายอาญา"
        elif "แพ่ง" in chapter or "พาณิชย์" in chapter:
            law_name = "ประมวลกฎหมายแพ่งและพาณิชย์"
        elif "รัฐธรรมนูญ" in chapter:
            law_name = "รัฐธรรมนูญแห่งราชอาณาจักรไทย พุทธศักราช ๒๕๖๐"

    # ── LAW entity ──────────────────────────────────────────────────
    if law_name and law_name not in entities:
        entities[law_name] = {
            "name": law_name,
            "type": "LAW",
            "description": law_name,
            "source_chunk_ids": [chunk_id],
        }

    # ── SECTION entities ────────────────────────────────────────────
    sections_in_chunk: list[str] = []          # prefixed names
    section_positions: list[tuple[str, int]] = []  # (prefixed_name, char_pos)
    for m in _SECTION_PAT.finditer(text):
        sec_name = _normalize_section(m.group(1), law_name)
        sections_in_chunk.append(sec_name)
        section_positions.append((sec_name, m.start()))

        start = m.start()

        # Check if this มาตรา is at start of a line (section header)
        # vs inline cross-reference (e.g. "ภายใต้บังคับมาตรา ๒๕๕")
        line_start_pos = text.rfind('\n', 0, m.start())
        prefix_on_line = text[line_start_pos + 1:m.start()].strip() if line_start_pos >= 0 else text[:m.start()].strip()
        is_at_line_start = len(prefix_on_line) == 0

        if is_at_line_start:
            # Real section header — extract full snippet
            next_header = _SECTION_HEADER_PAT.search(text, m.end() + 1)
            end_boundary = next_header.start() if next_header else len(text)
            end = min(start + 1200, end_boundary)
            snippet = text[start:end].replace("\n", " ").strip()[:900]
        else:
            # Inline cross-reference — use minimal description
            snippet = text[start:start + len(m.group(0)) + 5].strip()

        after_num = text[m.end():m.end()+20].strip()
        is_definition = is_at_line_start and not after_num.startswith("ถึง") and (
            re.match(r"[\[\(]?\d", after_num) or      # e.g. [303]
            re.match(r"[\u0E00-\u0E7F]", after_num)    # Thai text follows
        )

        if sec_name not in entities:
            entities[sec_name] = {
                "name": sec_name,
                "type": "SECTION",
                "description": snippet,
                "source_chunk_ids": [chunk_id],
                "_is_definition": is_definition,
            }
        elif is_definition:
            # Overwrite if previous was non-definition, or if new is more authoritative
            old_desc = entities[sec_name].get("description", "")
            should_overwrite = (
                not entities[sec_name].get("_is_definition")
                or ("หมายความว่า" in snippet and "หมายความว่า" not in old_desc)
                or ("ในประมวลกฎหมายนี้" in snippet and "พระราชบัญญัตินี้" in old_desc)
            )
            if should_overwrite:
                entities[sec_name]["description"] = snippet
                entities[sec_name]["_is_definition"] = True
                entities[sec_name]["source_chunk_ids"].append(chunk_id)

        # Link section → law  (deduplicated)
        if law_name:
            _add_rel(sec_name, law_name, "PART_OF",
                     f"{sec_name} เป็นส่วนหนึ่งของ{law_name}", 1.0)

    # Helper: find nearest section *before* a character position
    def _nearest(pos: int) -> str | None:
        best, best_dist = None, float("inf")
        for name, spos in section_positions:
            dist = pos - spos
            if 0 <= dist < best_dist:
                best_dist = dist
                best = name
        return best or (sections_in_chunk[0] if sections_in_chunk else None)

    # Deduplicated unique section names in chunk
    unique_sections = list(dict.fromkeys(sections_in_chunk))

    # ── OFFENSE entities ────────────────────────────────────────────
    for m in _OFFENSE_PAT.finditer(text):
        offense_name = _normalize_offense_name(m.group(1))
        if not offense_name:
            continue
        if offense_name not in entities:
            # Try to capture the full offense + penalty context
            ctx_match = _OFFENSE_CONTEXT_PAT.search(text, m.start())
            if ctx_match:
                desc = ctx_match.group(0).strip()[:300]
            else:
                desc = m.group(0).strip()[:200]
            entities[offense_name] = {
                "name": offense_name,
                "type": "OFFENSE",
                "description": desc,
                "source_chunk_ids": [chunk_id],
                "_is_definition": True,  # regex match = authoritative
            }
        nearest = _nearest(m.start())
        if nearest:
            _add_rel(nearest, offense_name, "DEFINES_OFFENSE",
                     f"{nearest} บัญญัติ{offense_name}", 0.9)

    # Keyword-based offense detection
    for keyword, offense_name in _OFFENSE_KEYWORDS.items():
        if keyword in text and offense_name not in entities:
            # Extract actual context around the keyword
            pos = text.find(keyword)
            ctx_start = max(0, pos - 40)
            ctx_end = min(pos + 250, len(text))
            raw_ctx = text[ctx_start:ctx_end].replace("\n", " ").strip()
            # Prefer context that includes penalty text
            desc = re.sub(r"\s+", " ", raw_ctx)[:250] if len(raw_ctx) > 20 else f"ความผิดเกี่ยวกับ{keyword}"
            entities[offense_name] = {
                "name": offense_name,
                "type": "OFFENSE",
                "description": desc,
                "source_chunk_ids": [chunk_id],
            }
            nearest = _nearest(pos)
            if nearest:
                _add_rel(nearest, offense_name, "RELATES_TO",
                         f"{nearest} เกี่ยวข้องกับ{offense_name}", 0.7)

    # ── PENALTY entities ────────────────────────────────────────────
    penalty_texts = _extract_penalty_text(text)
    for pt in penalty_texts:
        penalty_name = _make_penalty_name(pt)
        if penalty_name and penalty_name not in entities:
            entities[penalty_name] = {
                "name": penalty_name,
                "type": "PENALTY",
                "description": re.sub(r"\s+", " ", pt[:200]),
                "source_chunk_ids": [chunk_id],
            }
            pos = text.find(pt[:30])  # find by prefix (cleaned text may differ)
            nearest = _nearest(pos if pos >= 0 else 0)
            if nearest:
                _add_rel(nearest, penalty_name, "HAS_PENALTY",
                         f"{nearest} กำหนด{penalty_name}", 0.9)

    # ── Cross-references ────────────────────────────────────────────
    for pat in _XREF_PATTERNS:
        for m in pat.finditer(text):
            # Cross-refs default to same law
            ref_sec = _normalize_section(m.group(1), law_name)
            pos = m.start()
            source_sec = _nearest(pos)
            if source_sec and source_sec != ref_sec:
                _add_rel(source_sec, ref_sec, "REFERENCES",
                         f"{source_sec} อ้างอิง{ref_sec}", 0.8)
                # Ensure referenced section entity exists
                if ref_sec not in entities:
                    entities[ref_sec] = {
                        "name": ref_sec,
                        "type": "SECTION",
                        "description": f"{ref_sec} (อ้างอิงจาก{source_sec})",
                        "source_chunk_ids": [chunk_id],
                    }

    # ── LEGAL_CONCEPT entities ──────────────────────────────────────
    for keyword, concept_name in _LEGAL_CONCEPTS.items():
        if keyword in text and concept_name not in entities:
            # Extract actual context around the keyword
            pos = text.find(keyword)
            ctx_start = max(0, pos - 30)
            ctx_end = min(pos + 200, len(text))
            raw_ctx = text[ctx_start:ctx_end].replace("\n", " ").strip()
            desc = re.sub(r"\s+", " ", raw_ctx)[:200] if len(raw_ctx) > 20 else f"แนวคิดทางกฎหมาย: {concept_name}"
            entities[concept_name] = {
                "name": concept_name,
                "type": "LEGAL_CONCEPT",
                "description": desc,
                "source_chunk_ids": [chunk_id],
            }
            nearest = _nearest(pos)
            if nearest:
                _add_rel(nearest, concept_name, "INVOLVES_CONCEPT",
                         f"{nearest} เกี่ยวข้องกับ{concept_name}", 0.5)

    # ── ORGANIZATION entities ───────────────────────────────────────
    for keyword, org_name in _ORGANIZATIONS.items():
        if keyword in text and org_name not in entities:
            entities[org_name] = {
                "name": org_name,
                "type": "ORGANIZATION",
                "description": f"องค์กร: {org_name}",
                "source_chunk_ids": [chunk_id],
            }
            pos = text.find(keyword)
            nearest = _nearest(pos)
            if nearest:
                _add_rel(nearest, org_name, "MENTIONS",
                         f"{nearest} กล่าวถึง{org_name}", 0.4)

    # ── PERSON_TYPE entities ────────────────────────────────────────
    for keyword, person_name in _PERSON_TYPES.items():
        if keyword in text and person_name not in entities:
            entities[person_name] = {
                "name": person_name,
                "type": "PERSON_TYPE",
                "description": f"ประเภทบุคคล: {person_name}",
                "source_chunk_ids": [chunk_id],
            }
            pos = text.find(keyword)
            nearest = _nearest(pos)
            if nearest:
                _add_rel(nearest, person_name, "MENTIONS",
                         f"{nearest} กล่าวถึง{person_name}", 0.3)

    # ── Intra-chunk section adjacency (deduplicated) ────────────────
    for i in range(len(unique_sections) - 1):
        s1, s2 = unique_sections[i], unique_sections[i + 1]
        _add_rel(s1, s2, "ADJACENT_TO",
                 f"{s1} อยู่ถัดจาก{s2}", 0.6)

    return list(entities.values()), relationships


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_penalty_name(penalty_text: str) -> str:
    """Create a short, clean name for a penalty entity."""
    # Clean whitespace first
    penalty_text = re.sub(r"\s+", " ", penalty_text).strip()
    parts: list[str] = []
    if "ประหารชีวิต" in penalty_text:
        parts.append("ประหารชีวิต")
    if "จำคุกตลอดชีวิต" in penalty_text:
        parts.append("จำคุกตลอดชีวิต")
    elif "จำคุก" in penalty_text:
        m = re.search(r"จำคุก[^,]*?(?:ปี|เดือน|ตลอดชีวิต)", penalty_text)
        if m:
            parts.append(m.group(0).strip())
        else:
            parts.append("จำคุก")
    if "ปรับ" in penalty_text:
        m = re.search(r"ปรับ[^,]*?บาท", penalty_text)
        if m:
            parts.append(m.group(0).strip())
        else:
            parts.append("ปรับ")
    if not parts:
        return ""
    name = "โทษ" + " และ".join(parts)
    # Cap at 80 characters
    if len(name) > 80:
        name = name[:77] + "…"
    return name
