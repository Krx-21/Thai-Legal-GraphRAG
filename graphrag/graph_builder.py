"""Knowledge Graph Builder

1. LLM-driven entity & relationship extraction from Thai legal text chunks
2. Leiden community detection
3. Community summarisation
4. Artifact storage (Parquet + in-memory)
"""

from __future__ import annotations

import asyncio
import json
import re
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from graphrag import llm_utils
from graphrag.preprocessor import TextChunk

# ═══════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Entity:
    name: str
    entity_type: str
    description: str
    source_chunk_ids: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    is_definition: bool = False

    @property
    def id(self) -> str:
        return hashlib.md5(f"{self.entity_type}::{self.name}".encode()).hexdigest()


@dataclass
class Relationship:
    source: str
    target: str
    rel_type: str
    description: str
    weight: float = 1.0
    source_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class Community:
    community_id: int
    level: int
    entity_ids: list[str] = field(default_factory=list)
    title: str = ""
    summary: str = ""
    importance: float = 0.0
    findings: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Extraction Prompt
# ═══════════════════════════════════════════════════════════════════════

_EXTRACT_PROMPT = """คุณเป็นผู้เชี่ยวชาญด้านกฎหมายไทย ทำหน้าที่สกัด entity และ relationship จากข้อความกฎหมาย
จงสกัด entity และ relationship จากข้อความต่อไปนี้

## Entity Types ที่ต้องการ
{entity_types}

## Relationship Types ที่ต้องการ
{relationship_types}

## กฎการสกัด
1. สกัด entity ทุกตัวที่พบ โดยระบุ name, type, description
2. สกัด relationship ทุกคู่ที่พบ โดยระบุ source, target, type, description, weight (0-1)
3. สำหรับ SECTION entity ให้ใช้รูปแบบ "มาตรา XXX" เป็น name
4. สำหรับ OFFENSE entity ให้ใช้ชื่อความผิดภาษาไทย
5. สำหรับ PENALTY entity ให้ระบุโทษที่กำหนด

## ข้อความ
{text}

## Output Format (JSON)
ตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่น:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "..."}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "type": "...", "description": "...", "weight": 0.8}}
  ]
}}"""

_GLEANING_PROMPT = """ข้อความเดิม:
{text}

Entity และ Relationship ที่สกัดได้แล้ว:
{previous}

จงตรวจสอบอีกครั้งว่ามี entity หรือ relationship ใดที่ตกหล่นไป
ตอบเป็น JSON เท่านั้น ในรูปแบบเดียวกัน หรือ {{"entities":[],"relationships":[]}} ถ้าไม่มีอะไรเพิ่ม"""


# ═══════════════════════════════════════════════════════════════════════
# Extraction Logic
# ═══════════════════════════════════════════════════════════════════════

def _parse_json_response(text: str) -> dict:
    """Robustly parse JSON from LLM output."""
    # Try direct parse
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"entities": [], "relationships": []}


def _parse_extraction(data: dict, chunk_id: str) -> tuple[list[Entity], list[Relationship]]:
    """Parse extracted JSON into Entity/Relationship objects."""
    entities: list[Entity] = []
    relationships: list[Relationship] = []

    for e in data.get("entities", []):
        ent_type = e.get("type", "LEGAL_CONCEPT").upper()
        if ent_type not in config.ENTITY_TYPES:
            ent_type = "LEGAL_CONCEPT"
        name = e.get("name", "").strip()
        if name:
            entities.append(Entity(
                name=name,
                entity_type=ent_type,
                description=e.get("description", ""),
                source_chunk_ids=[chunk_id],
            ))

    for r in data.get("relationships", []):
        src = r.get("source", "").strip()
        tgt = r.get("target", "").strip()
        if src and tgt:
            relationships.append(Relationship(
                source=src,
                target=tgt,
                rel_type=r.get("type", "RELATED_TO").upper(),
                description=r.get("description", ""),
                weight=float(r.get("weight", 1.0)),
                source_chunk_ids=[chunk_id],
            ))

    return entities, relationships


def extract_from_chunk(chunk: TextChunk, do_gleaning: bool = False) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities and relationships from a single chunk using LLM."""
    prompt = _EXTRACT_PROMPT.format(
        entity_types=", ".join(config.ENTITY_TYPES),
        relationship_types=", ".join(config.RELATIONSHIP_TYPES),
        text=chunk.text[:6000],
    )
    response = llm_utils.chat(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config.MAX_EXTRACTION_TOKENS,
    )
    data = _parse_json_response(response)
    entities, relationships = _parse_extraction(data, chunk.chunk_id)

    if do_gleaning:
        gleaning_prompt = _GLEANING_PROMPT.format(
            text=chunk.text[:4000],
            previous=json.dumps(data, ensure_ascii=False)[:3000],
        )
        gleaning_resp = llm_utils.chat(
            messages=[{"role": "user", "content": gleaning_prompt}],
            max_tokens=config.MAX_EXTRACTION_TOKENS,
        )
        extra = _parse_json_response(gleaning_resp)
        e2, r2 = _parse_extraction(extra, chunk.chunk_id)
        entities.extend(e2)
        relationships.extend(r2)

    return entities, relationships


async def aextract_from_chunk(chunk: TextChunk) -> tuple[list[Entity], list[Relationship]]:
    """Async version of extract_from_chunk (no gleaning for speed)."""
    prompt = _EXTRACT_PROMPT.format(
        entity_types=", ".join(config.ENTITY_TYPES),
        relationship_types=", ".join(config.RELATIONSHIP_TYPES),
        text=chunk.text[:6000],
    )
    response = await llm_utils.achat(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config.MAX_EXTRACTION_TOKENS,
    )
    data = _parse_json_response(response)
    return _parse_extraction(data, chunk.chunk_id)


# ── Checkpoint helpers ──────────────────────────────────────────────

CHECKPOINT_FILE = config.OUTPUT_DIR / "extraction_checkpoint.json"


def _save_checkpoint(done_ids: set[str], entities: list[dict], relationships: list[dict]):
    data = {
        "done_chunk_ids": list(done_ids),
        "entities": entities,
        "relationships": relationships,
    }
    CHECKPOINT_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _load_checkpoint() -> tuple[set[str], list[dict], list[dict]] | None:
    if not CHECKPOINT_FILE.exists():
        return None
    data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    return (
        set(data["done_chunk_ids"]),
        data["entities"],
        data["relationships"],
    )


# ═══════════════════════════════════════════════════════════════════════
# Graph Construction
# ═══════════════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """In-memory knowledge graph with community detection."""

    def __init__(self):
        self.entities: dict[str, Entity] = {}     # name → Entity
        self.relationships: list[Relationship] = []
        self._seen_rel_keys: set[tuple[str, str, str]] = set()
        self.communities: list[Community] = []
        self.graph = nx.Graph()
        self.text_chunks: dict[str, TextChunk] = {}

    # ── Build ───────────────────────────────────────────────────────

    def add_entity(self, entity: Entity):
        key = entity.name
        if key in self.entities:
            existing = self.entities[key]
            existing.source_chunk_ids.extend(entity.source_chunk_ids)
            # Prefer definition descriptions over cross-references
            if entity.is_definition and not existing.is_definition:
                existing.description = entity.description
                existing.is_definition = True
            elif entity.is_definition and existing.is_definition:
                # Both definitions — prefer one with actual legal content
                new_has_def = "\u0e2b\u0e21\u0e32\u0e22\u0e04\u0e27\u0e32\u0e21\u0e27\u0e48\u0e32" in entity.description
                old_has_def = "\u0e2b\u0e21\u0e32\u0e22\u0e04\u0e27\u0e32\u0e21\u0e27\u0e48\u0e32" in existing.description
                old_is_enacting = "\u0e1e\u0e23\u0e30\u0e23\u0e32\u0e0a\u0e1a\u0e31\u0e0d\u0e0d\u0e31\u0e15\u0e34\u0e19\u0e35\u0e49" in existing.description
                new_has_penalty = "\u0e15\u0e49\u0e2d\u0e07\u0e23\u0e30\u0e27\u0e32\u0e07\u0e42\u0e17\u0e29" in entity.description
                old_has_penalty = "\u0e15\u0e49\u0e2d\u0e07\u0e23\u0e30\u0e27\u0e32\u0e07\u0e42\u0e17\u0e29" in existing.description
                if (new_has_def and not old_has_def) or old_is_enacting:
                    existing.description = entity.description
                elif new_has_penalty and not old_has_penalty:
                    existing.description = entity.description
                elif len(entity.description) > len(existing.description) * 1.5:
                    # Fallback: significantly longer definition wins
                    existing.description = entity.description
            elif entity.description and len(entity.description) > len(existing.description) and not existing.is_definition:
                existing.description = entity.description
        else:
            self.entities[key] = entity

    def add_relationship(self, rel: Relationship):
        key = (rel.source, rel.target, rel.rel_type)
        if key in self._seen_rel_keys:
            for existing in self.relationships:
                if (existing.source, existing.target, existing.rel_type) == key:
                    existing.source_chunk_ids.extend(rel.source_chunk_ids)
                    break
            return
        self._seen_rel_keys.add(key)
        self.relationships.append(rel)

    def build_from_chunks(self, chunks: list[TextChunk], use_async: bool = True, use_regex: bool = False):
        """Run full extraction pipeline on all chunks with checkpoint/resume."""
        self.text_chunks = {c.chunk_id: c for c in chunks}

        if use_regex:
            self._regex_extract(chunks)
            self._build_networkx()
            print(f"[GraphBuilder] Entities: {len(self.entities)}, "
                  f"Relationships: {len(self.relationships)}, "
                  f"Nodes: {self.graph.number_of_nodes()}, "
                  f"Edges: {self.graph.number_of_edges()}")
            return

        # Check for existing checkpoint
        checkpoint = _load_checkpoint()
        done_ids: set[str] = set()
        raw_entities: list[dict] = []
        raw_relationships: list[dict] = []

        if checkpoint:
            done_ids, raw_entities, raw_relationships = checkpoint
            print(f"[GraphBuilder] Resuming from checkpoint: {len(done_ids)} chunks done")
            # Restore already-extracted data
            for e_dict in raw_entities:
                self.add_entity(Entity(
                    name=e_dict["name"],
                    entity_type=e_dict["type"],
                    description=e_dict["description"],
                    source_chunk_ids=e_dict["source_chunk_ids"],
                ))
            for r_dict in raw_relationships:
                self.add_relationship(Relationship(
                    source=r_dict["source"],
                    target=r_dict["target"],
                    rel_type=r_dict["type"],
                    description=r_dict["description"],
                    weight=r_dict["weight"],
                    source_chunk_ids=r_dict["source_chunk_ids"],
                ))

        remaining = [c for c in chunks if c.chunk_id not in done_ids]
        print(f"[GraphBuilder] Extracting from {len(remaining)} chunks "
              f"({len(done_ids)} already done) …")

        if not remaining:
            self._build_networkx()
            return

        if use_async:
            asyncio.run(self._async_extract(remaining, done_ids, raw_entities, raw_relationships))
        else:
            self._sync_extract(remaining, done_ids, raw_entities, raw_relationships)

        self._build_networkx()
        print(f"[GraphBuilder] Entities: {len(self.entities)}, "
              f"Relationships: {len(self.relationships)}, "
              f"Nodes: {self.graph.number_of_nodes()}, "
              f"Edges: {self.graph.number_of_edges()}")

    def _regex_extract(self, chunks: list[TextChunk]):
        """Fast regex-based extraction — no LLM calls needed."""
        from graphrag import regex_extractor
        total = len(chunks)
        print(f"[GraphBuilder] Regex extraction from {total} chunks …")
        for i, chunk in enumerate(chunks):
            ents, rels = regex_extractor.extract_from_chunk(chunk)
            for e in ents:
                self.add_entity(Entity(
                    name=e["name"],
                    entity_type=e["type"],
                    description=e["description"],
                    source_chunk_ids=e["source_chunk_ids"],
                    is_definition=e.get("_is_definition", False),
                ))
            for r in rels:
                self.add_relationship(Relationship(
                    source=r["source"],
                    target=r["target"],
                    rel_type=r["type"],
                    description=r["description"],
                    weight=r["weight"],
                    source_chunk_ids=r["source_chunk_ids"],
                ))
            if (i + 1) % 200 == 0 or (i + 1) == total:
                print(f"  [{i+1}/{total}] {len(self.entities)} entities, "
                      f"{len(self.relationships)} rels")

    async def _async_extract(
        self,
        chunks: list[TextChunk],
        done_ids: set[str],
        raw_entities: list[dict],
        raw_relationships: list[dict],
    ):
        """Extract entities/relationships using async concurrency."""
        BATCH_SIZE = 3    # small batches to avoid rate limits
        total = len(chunks)
        processed = 0
        sem = asyncio.Semaphore(3)  # max 3 concurrent API calls

        async def _limited_extract(chunk: TextChunk):
            async with sem:
                return await aextract_from_chunk(chunk)

        for batch_start in range(0, total, BATCH_SIZE):
            batch = chunks[batch_start : batch_start + BATCH_SIZE]
            tasks = [_limited_extract(c) for c in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for chunk, result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"  [!] {chunk.chunk_id}: {result}")
                    continue
                ents, rels = result
                for e in ents:
                    self.add_entity(e)
                    raw_entities.append({
                        "name": e.name, "type": e.entity_type,
                        "description": e.description,
                        "source_chunk_ids": e.source_chunk_ids,
                    })
                for r in rels:
                    self.add_relationship(r)
                    raw_relationships.append({
                        "source": r.source, "target": r.target,
                        "type": r.rel_type, "description": r.description,
                        "weight": r.weight,
                        "source_chunk_ids": r.source_chunk_ids,
                    })
                done_ids.add(chunk.chunk_id)

            processed += len(batch)
            _save_checkpoint(done_ids, raw_entities, raw_relationships)
            print(f"  [{processed}/{total}] checkpoint saved "
                  f"({len(self.entities)} entities, {len(self.relationships)} rels)")

    def _sync_extract(
        self,
        chunks: list[TextChunk],
        done_ids: set[str],
        raw_entities: list[dict],
        raw_relationships: list[dict],
    ):
        """Fallback sync extraction with checkpoint."""
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            try:
                ents, rels = extract_from_chunk(chunk)
                for e in ents:
                    self.add_entity(e)
                    raw_entities.append({
                        "name": e.name, "type": e.entity_type,
                        "description": e.description,
                        "source_chunk_ids": e.source_chunk_ids,
                    })
                for r in rels:
                    self.add_relationship(r)
                    raw_relationships.append({
                        "source": r.source, "target": r.target,
                        "type": r.rel_type, "description": r.description,
                        "weight": r.weight,
                        "source_chunk_ids": r.source_chunk_ids,
                    })
                done_ids.add(chunk.chunk_id)
            except Exception as exc:
                print(f"  [!] {chunk.chunk_id}: {exc}")

            if (i + 1) % 10 == 0:
                _save_checkpoint(done_ids, raw_entities, raw_relationships)
                print(f"  [{i+1}/{total}] checkpoint saved")

        _save_checkpoint(done_ids, raw_entities, raw_relationships)

    def _build_networkx(self):
        """Construct NetworkX graph from extracted data."""
        self.graph.clear()
        for name, ent in self.entities.items():
            self.graph.add_node(name, **{
                "entity_type": ent.entity_type,
                "description": ent.description,
            })
        for rel in self.relationships:
            if rel.source in self.entities and rel.target in self.entities:
                self.graph.add_edge(
                    rel.source, rel.target,
                    rel_type=rel.rel_type,
                    description=rel.description,
                    weight=rel.weight,
                )

    # ── Community Detection ─────────────────────────────────────────

    def detect_communities(self):
        """Leiden clustering using leidenalg + igraph."""
        if self.graph.number_of_nodes() == 0:
            print("[GraphBuilder] No nodes to cluster")
            return

        print("[GraphBuilder] Running Leiden community detection …")
        try:
            import igraph as ig
            import leidenalg

            # Convert NetworkX → igraph
            node_list = list(self.graph.nodes())
            node_idx = {n: i for i, n in enumerate(node_list)}
            ig_graph = ig.Graph()
            ig_graph.add_vertices(len(node_list))
            ig_graph.vs["name"] = node_list
            edges = [(node_idx[u], node_idx[v]) for u, v in self.graph.edges()
                     if u in node_idx and v in node_idx]
            ig_graph.add_edges(edges)

            partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)

            # Build communities from partition
            clusters: dict[int, list[str]] = {}
            for node_i, comm_id in enumerate(partition.membership):
                clusters.setdefault(comm_id, []).append(node_list[node_i])

        except Exception as e:
            print(f"  [!] Leiden failed ({e}), falling back to connected components")
            clusters = {}
            for i, component in enumerate(nx.connected_components(self.graph)):
                clusters[i] = list(component)

        self.communities = []
        for cluster_id, members in clusters.items():
            self.communities.append(Community(
                community_id=cluster_id,
                level=0,
                entity_ids=members,
            ))

        print(f"[GraphBuilder] Communities: {len(self.communities)}")

    # ── Community Summarisation ─────────────────────────────────────

    def summarize_communities(self):
        """Generate rule-based summaries for each community (no LLM needed)."""
        print(f"[GraphBuilder] Summarizing {len(self.communities)} communities …")

        for comm in self.communities:
            # Collect member entities by type
            by_type: dict[str, list[Entity]] = {}
            for eid in comm.entity_ids:
                ent = self.entities.get(eid)
                if ent:
                    by_type.setdefault(ent.entity_type, []).append(ent)

            # Build title from most prominent type + members
            sections = [e.name for e in by_type.get("SECTION", [])]
            offenses = [e.name for e in by_type.get("OFFENSE", [])]
            laws = [e.name for e in by_type.get("LAW", [])]
            concepts = [e.name for e in by_type.get("LEGAL_CONCEPT", [])]
            penalties = [e.name for e in by_type.get("PENALTY", [])]

            # Title
            if offenses:
                comm.title = offenses[0] if len(offenses) == 1 else f"กลุ่มความผิด: {', '.join(offenses[:3])}"
            elif sections:
                comm.title = f"กลุ่มมาตรา {sections[0]}–{sections[-1]}" if len(sections) > 1 else sections[0]
            elif concepts:
                comm.title = concepts[0]
            else:
                comm.title = f"กลุ่มที่ {comm.community_id}"

            # Summary
            parts: list[str] = []
            if laws:
                parts.append(f"กฎหมาย: {', '.join(laws[:2])}")
            if sections:
                parts.append(f"มาตรา: {', '.join(sections[:5])}" + (f" (รวม {len(sections)} มาตรา)" if len(sections) > 5 else ""))
            if offenses:
                parts.append(f"ความผิด: {', '.join(offenses[:3])}")
            if penalties:
                parts.append(f"บทลงโทษ: {', '.join(penalties[:3])}")
            if concepts:
                parts.append(f"แนวคิด: {', '.join(concepts[:3])}")
            comm.summary = "; ".join(parts) if parts else "ไม่มีข้อมูล"

            # Importance — more entities & relationships = more important
            member_set = set(comm.entity_ids)
            rel_count = sum(1 for r in self.relationships
                          if r.source in member_set and r.target in member_set)
            comm.importance = min(1.0, (len(comm.entity_ids) + rel_count) / 50)

            # Findings
            findings: list[str] = []
            for e in by_type.get("SECTION", [])[:3]:
                findings.append(f"{e.name}: {e.description[:100]}")
            for e in by_type.get("OFFENSE", [])[:2]:
                findings.append(f"{e.name}: {e.description[:100]}")
            comm.findings = findings

        print(f"[GraphBuilder] Community summaries done")

    # ── Embeddings ──────────────────────────────────────────────────

    def compute_embeddings(self):
        """Embed all entities using TF-IDF (no API needed).
        Falls back to OpenAI embeddings if sklearn is not available."""
        names = list(self.entities.keys())
        if not names:
            return
        print(f"[GraphBuilder] Embedding {len(names)} entities …")
        texts = [
            f"{ent.entity_type}: {ent.name} - {ent.description}"
            for ent in self.entities.values()
        ]

        try:
            # Try local TF-IDF embeddings (no API needed)
            import pickle
            from sklearn.feature_extraction.text import TfidfVectorizer
            try:
                from pythainlp.tokenize import word_tokenize
                tokenized = [" ".join(word_tokenize(t, engine="newmm")) for t in texts]
            except ImportError:
                tokenized = texts
            vectorizer = TfidfVectorizer(max_features=512)
            matrix = vectorizer.fit_transform(tokenized).toarray()
            for name, vec in zip(names, matrix):
                self.entities[name].embedding = vec.tolist()
            # Save vectorizer for query-time use
            vec_path = config.PARQUET_DIR / "tfidf_vectorizer.pkl"
            vec_path.parent.mkdir(parents=True, exist_ok=True)
            with open(vec_path, "wb") as f:
                pickle.dump(vectorizer, f)
            print(f"[GraphBuilder] TF-IDF embeddings done (dim={matrix.shape[1]})")
        except ImportError:
            # Fallback to OpenAI embeddings
            vectors = llm_utils.embed(texts)
            for name, vec in zip(names, vectors):
                self.entities[name].embedding = vec.tolist()
            print("[GraphBuilder] OpenAI embeddings done")

    # ── Persistence (Parquet) ───────────────────────────────────────

    def save_parquet(self, output_dir: Path | None = None):
        output_dir = output_dir or config.PARQUET_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Entities
        ent_records = []
        for e in self.entities.values():
            ent_records.append({
                "id": e.id,
                "name": e.name,
                "type": e.entity_type,
                "description": e.description,
                "source_chunk_ids": json.dumps(e.source_chunk_ids, ensure_ascii=False),
            })
        pd.DataFrame(ent_records).to_parquet(output_dir / "entities.parquet")

        # Relationships
        rel_records = []
        for r in self.relationships:
            rel_records.append({
                "source": r.source,
                "target": r.target,
                "type": r.rel_type,
                "description": r.description,
                "weight": r.weight,
                "source_chunk_ids": json.dumps(r.source_chunk_ids, ensure_ascii=False),
            })
        pd.DataFrame(rel_records).to_parquet(output_dir / "relationships.parquet")

        # Communities
        comm_records = []
        for c in self.communities:
            comm_records.append({
                "community_id": c.community_id,
                "level": c.level,
                "entity_ids": json.dumps(c.entity_ids, ensure_ascii=False),
                "title": c.title,
                "summary": c.summary,
                "importance": c.importance,
                "findings": json.dumps(c.findings, ensure_ascii=False),
            })
        pd.DataFrame(comm_records).to_parquet(output_dir / "communities.parquet")

        # Text units
        chunk_records = []
        for c in self.text_chunks.values():
            chunk_records.append({
                "chunk_id": c.chunk_id,
                "law_name": c.law_name,
                "chapter": c.chapter,
                "sections": json.dumps(c.sections, ensure_ascii=False),
                "text": c.text,
                "token_count": c.token_count,
            })
        pd.DataFrame(chunk_records).to_parquet(output_dir / "text_units.parquet")

        print(f"[GraphBuilder] Saved parquet files -> {output_dir}")

    def load_parquet(self, input_dir: Path | None = None):
        """Restore graph from saved parquet files."""
        input_dir = input_dir or config.PARQUET_DIR

        # Entities
        df = pd.read_parquet(input_dir / "entities.parquet")
        for _, row in df.iterrows():
            self.entities[row["name"]] = Entity(
                name=row["name"],
                entity_type=row["type"],
                description=row["description"],
                source_chunk_ids=json.loads(row["source_chunk_ids"]),
            )

        # Relationships
        df = pd.read_parquet(input_dir / "relationships.parquet")
        for _, row in df.iterrows():
            self.relationships.append(Relationship(
                source=row["source"],
                target=row["target"],
                rel_type=row["type"],
                description=row["description"],
                weight=row["weight"],
                source_chunk_ids=json.loads(row["source_chunk_ids"]),
            ))

        # Communities
        df = pd.read_parquet(input_dir / "communities.parquet")
        for _, row in df.iterrows():
            self.communities.append(Community(
                community_id=row["community_id"],
                level=row["level"],
                entity_ids=json.loads(row["entity_ids"]),
                title=row["title"],
                summary=row["summary"],
                importance=row["importance"],
                findings=json.loads(row["findings"]),
            ))

        # Text units
        df = pd.read_parquet(input_dir / "text_units.parquet")
        for _, row in df.iterrows():
            self.text_chunks[row["chunk_id"]] = TextChunk(
                chunk_id=row["chunk_id"],
                law_name=row["law_name"],
                chapter=row["chapter"],
                sections=json.loads(row["sections"]),
                text=row["text"],
                token_count=row["token_count"],
            )

        self._build_networkx()
        print(f"[GraphBuilder] Loaded from parquet: {len(self.entities)} entities, "
              f"{len(self.relationships)} relationships, "
              f"{len(self.communities)} communities")

    # ── Serialise graph to JSON (for PyVis) ─────────────────────────

    def to_json(self) -> dict:
        nodes = []
        for name, ent in self.entities.items():
            # Strip law prefix for display label: "[อาญา] มาตรา 3" → "มาตรา 3"
            label = name
            if label.startswith("[") and "] " in label:
                label = label.split("] ", 1)[1]
            nodes.append({
                "id": name,
                "label": label,
                "type": ent.entity_type,
                "description": ent.description,
            })
        edges = []
        for rel in self.relationships:
            if rel.source in self.entities and rel.target in self.entities:
                edges.append({
                    "source": rel.source,
                    "target": rel.target,
                    "type": rel.rel_type,
                    "description": rel.description,
                    "weight": rel.weight,
                })
        return {"nodes": nodes, "edges": edges}

    def save_graph_json(self, path: Path | None = None):
        path = path or config.GRAPH_OUTPUT_DIR / "knowledge_graph.json"
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[GraphBuilder] Graph JSON -> {path}")
