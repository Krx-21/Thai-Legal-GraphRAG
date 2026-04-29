"""Thai Legal Text Preprocessor

Handles:
  - Thai text normalization (BOM, zero-width chars, Unicode)
  - Section (มาตรา) extraction via regex
  - Tokenization with PyThaiNLP
  - Chunking that preserves section boundaries
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path

import tiktoken
from pythainlp.tokenize import word_tokenize

import config

# ── Regex Patterns ──────────────────────────────────────────────────────
_SECTION_PATTERN = re.compile(r"^[ \t]*มาตรา\s*(\d+(?:/\d+)?)", re.MULTILINE)
_BOM = "\ufeff"
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")


@dataclass
class LegalSection:
    """A single legal section extracted from raw text."""
    law_name: str
    chapter: str
    section_number: str
    raw_text: str
    tokenized_text: str = ""
    char_count: int = 0


@dataclass
class TextChunk:
    """A chunk of text ready for GraphRAG indexing."""
    chunk_id: str
    law_name: str
    chapter: str
    sections: list[str] = field(default_factory=list)
    text: str = ""
    token_count: int = 0


# ── Text Normalization ──────────────────────────────────────────────────

def normalize_thai(text: str) -> str:
    """Remove BOM, zero-width chars, normalize Unicode."""
    text = text.replace(_BOM, "")
    text = _ZERO_WIDTH.sub("", text)
    text = unicodedata.normalize("NFC", text)
    # collapse multiple spaces / blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Section Extraction ──────────────────────────────────────────────────

def extract_sections(text: str, law_name: str, chapter: str) -> list[LegalSection]:
    """Split raw law text into individual sections (มาตรา)."""
    text = normalize_thai(text)
    matches = list(_SECTION_PATTERN.finditer(text))
    if not matches:
        # Whole file is one section-less block (e.g. preamble)
        return [LegalSection(
            law_name=law_name,
            chapter=chapter,
            section_number="0",
            raw_text=text,
            char_count=len(text),
        )]

    sections: list[LegalSection] = []

    # Text before first มาตรา (chapter header, etc.)
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(LegalSection(
            law_name=law_name,
            chapter=chapter,
            section_number="preamble",
            raw_text=preamble,
            char_count=len(preamble),
        ))

    for i, m in enumerate(matches):
        sec_num = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sec_text = text[start:end].strip()
        sections.append(LegalSection(
            law_name=law_name,
            chapter=chapter,
            section_number=sec_num,
            raw_text=sec_text,
            char_count=len(sec_text),
        ))

    return sections


def tokenize_section(section: LegalSection) -> LegalSection:
    """Add PyThaiNLP word-tokenized version of the text."""
    tokens = word_tokenize(section.raw_text, engine="newmm")
    section.tokenized_text = "|".join(tokens)
    return section


# ── Chunking ────────────────────────────────────────────────────────────

_enc = tiktoken.get_encoding(config.TOKEN_ENCODING)


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def chunk_sections(
    sections: list[LegalSection],
    max_tokens: int = config.CHUNK_SIZE,
    overlap_tokens: int = config.CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Merge sections into token-limited chunks preserving section boundaries."""
    chunks: list[TextChunk] = []
    current_texts: list[str] = []
    current_sections: list[str] = []
    current_tokens = 0
    law_name = sections[0].law_name if sections else ""
    chapter = sections[0].chapter if sections else ""
    idx = 0

    def _flush():
        nonlocal current_texts, current_sections, current_tokens, idx
        if not current_texts:
            return
        chunks.append(TextChunk(
            chunk_id=f"{law_name}__{chapter}__{idx}",
            law_name=law_name,
            chapter=chapter,
            sections=list(current_sections),
            text="\n\n".join(current_texts),
            token_count=current_tokens,
        ))
        idx += 1

        # keep last section as overlap
        if overlap_tokens > 0 and current_texts:
            last = current_texts[-1]
            last_sec = current_sections[-1]
            last_tok = _count_tokens(last)
            current_texts = [last]
            current_sections = [last_sec]
            current_tokens = last_tok
        else:
            current_texts = []
            current_sections = []
            current_tokens = 0

    for sec in sections:
        sec_tokens = _count_tokens(sec.raw_text)
        if sec_tokens > max_tokens:
            # Section itself exceeds chunk size → emit as-is
            _flush()
            chunks.append(TextChunk(
                chunk_id=f"{law_name}__{chapter}__{idx}",
                law_name=law_name,
                chapter=chapter,
                sections=[sec.section_number],
                text=sec.raw_text,
                token_count=sec_tokens,
            ))
            idx += 1
            current_texts = []
            current_sections = []
            current_tokens = 0
            continue

        if current_tokens + sec_tokens > max_tokens:
            _flush()

        current_texts.append(sec.raw_text)
        current_sections.append(sec.section_number)
        current_tokens += sec_tokens

    _flush()
    return chunks


# ── File-level Processing ───────────────────────────────────────────────

def process_law_file(filepath: Path) -> list[TextChunk]:
    """Read a single .txt law file and return chunks."""
    text = filepath.read_text(encoding="utf-8")
    # Derive law_name from parent directory, chapter from filename stem
    law_name = filepath.parent.name
    chapter = filepath.stem
    sections = extract_sections(text, law_name, chapter)
    sections = [tokenize_section(s) for s in sections]
    return chunk_sections(sections)


def process_all_laws(data_dir: Path | None = None) -> list[TextChunk]:
    """Process every .txt file under the law data directory."""
    data_dir = data_dir or config.LAW_DATA_DIR
    all_chunks: list[TextChunk] = []
    _SKIP_STEMS = {"หมายเหตุ", "เชิงอรรถ", "พระราชบัญญัติแก้ไขเพิ่มเติม", "พระราชบัญญัติให้ใช้ประมวลกฎหมายอาญา", "พระราชปรารภประกาศใช้รัฐธรรมนูญ"}
    txt_files = sorted(f for f in data_dir.rglob("*.txt") if f.stem not in _SKIP_STEMS)
    print(f"[Preprocessor] Found {len(txt_files)} law text files")

    for fp in txt_files:
        try:
            chunks = process_law_file(fp)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  [!] Error processing {fp.name}: {e}")

    print(f"[Preprocessor] Total chunks: {len(all_chunks)}")
    return all_chunks


def save_chunks_metadata(chunks: list[TextChunk], output_path: Path | None = None):
    """Persist chunk metadata as JSON."""
    output_path = output_path or config.OUTPUT_DIR / "chunks_metadata.json"
    data = [asdict(c) for c in chunks]
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Preprocessor] Saved metadata -> {output_path}")


# ── Convenience ─────────────────────────────────────────────────────────

def prepare_graphrag_input(chunks: list[TextChunk], output_dir: Path | None = None):
    """Write each chunk as a separate .txt file for the indexing pipeline."""
    output_dir = output_dir or config.OUTPUT_DIR / "graphrag_input"
    output_dir.mkdir(parents=True, exist_ok=True)
    for c in chunks:
        out_file = output_dir / f"{c.chunk_id}.txt"
        out_file.write_text(c.text, encoding="utf-8")
    print(f"[Preprocessor] Wrote {len(chunks)} input files -> {output_dir}")
