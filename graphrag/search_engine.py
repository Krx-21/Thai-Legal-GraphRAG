"""Dual-level Search Engine

- Local Search: multi-strategy retrieval (TF-IDF + BM25 + Dense) with RRF fusion
- Global Search: community-report summarisation (map-reduce)
- Hybrid Search: combined local + global
"""

from __future__ import annotations

import pickle
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from graphrag import llm_utils
from graphrag.graph_builder import KnowledgeGraph, Entity, Community


@dataclass
class SearchResult:
    mode: str                                 # "local" | "global" | "hybrid"
    context: str = ""                         # assembled context for LLM
    entities: list[dict] = field(default_factory=list)
    community_reports: list[dict] = field(default_factory=list)
    source_chunks: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Multi-Strategy Retrieval Index (lazily initialised)
# ═══════════════════════════════════════════════════════════════════════

class _RetrievalIndex:
    """Singleton index that holds TF-IDF, BM25, and Dense structures."""

    _instance: "_RetrievalIndex | None" = None
    _lock = threading.Lock()

    def __init__(self):
        self._ready = False

    @classmethod
    def get(cls, kg: KnowledgeGraph) -> "_RetrievalIndex":
        # Double-checked locking: cheap path when ready, locked path on first build.
        if cls._instance is not None and cls._instance._ready:
            return cls._instance
        with cls._lock:
            if cls._instance is None or not cls._instance._ready:
                inst = cls()
                inst._build(kg)
                cls._instance = inst
            return cls._instance

    # ── build ────────────────────────────────────────────────────────

    def _build(self, kg: KnowledgeGraph):
        self.names: list[str] = []
        self.item_types: list[str] = []   # 'entity' | 'chunk'
        self.texts: list[str] = []

        # Index entities
        for name, ent in kg.entities.items():
            self.names.append(name)
            self.item_types.append("entity")
            self.texts.append(
                f"{ent.entity_type}: {name} - {ent.description}"
            )

        # Index text chunks (full section text for better recall)
        self._chunk_section_map: dict[str, list[str]] = {}  # chunk_id → entity names
        # Build reverse map: chunk_id → entities sourced from it
        for ename, ent in kg.entities.items():
            for cid in ent.source_chunk_ids:
                self._chunk_section_map.setdefault(cid, []).append(ename)
        for chunk_id, chunk in kg.text_chunks.items():
            self.names.append(chunk_id)
            self.item_types.append("chunk")
            self.texts.append(chunk.text[:800])

        self.n = len(self.names)

        # Tokenizer
        try:
            from pythainlp.tokenize import word_tokenize
            self._tok = lambda t: word_tokenize(t, engine="newmm")
        except ImportError:
            self._tok = lambda t: t.split()

        self.tokenized = [self._tok(t) for t in self.texts]

        self._build_tfidf()
        self._build_bm25()
        self._build_dense()
        self._ready = True

    def _build_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        joined = [" ".join(toks) for toks in self.tokenized]
        self.tfidf_vec = TfidfVectorizer(max_features=512)
        self.tfidf_mat = self.tfidf_vec.fit_transform(joined).toarray().astype(np.float32)
        # Save vectorizer for backward compat
        p = config.PARQUET_DIR / "tfidf_vectorizer.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self.tfidf_vec, f)

    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.tokenized)

    def _build_dense(self):
        from sentence_transformers import SentenceTransformer
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.dense_model = SentenceTransformer(model_name)

        # Disk cache: dense_mat is a deterministic function of (model, texts).
        # Hash the joined corpus + model name to derive a cache key. Rebuild
        # whenever texts or model change.
        import hashlib
        h = hashlib.sha1()
        h.update(model_name.encode("utf-8"))
        h.update(b"\x1f")
        for t in self.texts:
            h.update(t.encode("utf-8", errors="ignore"))
            h.update(b"\x1e")
        cache_key = h.hexdigest()[:16]
        cache_path = config.PARQUET_DIR / f"dense_mat_{cache_key}.npy"

        if cache_path.exists():
            try:
                arr = np.load(cache_path)
                if arr.shape[0] == self.n:
                    self.dense_mat = arr.astype(np.float32)
                    return
            except Exception:
                pass  # fall through to recompute

        self.dense_mat = self.dense_model.encode(
            self.texts, show_progress_bar=True, batch_size=64,
            normalize_embeddings=True,
        ).astype(np.float32)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self.dense_mat)
        except Exception:
            pass  # cache miss is non-fatal

    # ── query helpers ────────────────────────────────────────────────

    def _q_tfidf(self, query: str, k: int) -> list[tuple[int, float]]:
        q = " ".join(self._tok(query))
        v = self.tfidf_vec.transform([q]).toarray().astype(np.float32)
        sims = cosine_similarity(v, self.tfidf_mat)[0]
        idx = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in idx]

    def _q_bm25(self, query: str, k: int) -> list[tuple[int, float]]:
        toks = self._tok(query)
        scores = self.bm25.get_scores(toks)
        idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in idx]

    def _q_dense(self, query: str, k: int) -> list[tuple[int, float]]:
        v = self.dense_model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        sims = cosine_similarity(v, self.dense_mat)[0]
        idx = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in idx]

    # ── RRF fusion ───────────────────────────────────────────────────

    @staticmethod
    def _rrf(lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion over multiple ranked lists of (idx, score)."""
        scores: dict[int, float] = {}
        for ranked in lists:
            for rank, (idx, _) in enumerate(ranked, 1):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ── public query ─────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float, str]]:
        """RRF fusion of TF-IDF + BM25 + Dense.  Returns [(name, rrf_score, item_type)]."""
        fetch = top_k * 10  # fetch more for fusion
        tfidf_r = self._q_tfidf(query, fetch)
        bm25_r = self._q_bm25(query, fetch)
        dense_r = self._q_dense(query, fetch)
        fused = self._rrf([tfidf_r, bm25_r, dense_r])
        return [(self.names[i], s, self.item_types[i]) for i, s in fused[:top_k * 5]]


# ═══════════════════════════════════════════════════════════════════════
# Legal Query Expansion
# ═══════════════════════════════════════════════════════════════════════

# Mapping: Thai legal keyword → (law_prefix, section_numbers[])
# Each key is a single Thai legal *term* (a concept that exists in legal dictionaries
# or statute headings), NOT a question phrase. Sections listed are those where the
# term is defined or principally regulated. The list is intentionally short:
# disambiguation between adjacent sections is left to BM25 + Dense retrieval.
_LEGAL_QUERY_BOOST: dict[str, list[tuple[str, str]]] = {
    # ── Criminal code — general principles ──
    "ป้องกัน": [("[อาญา]", "68"), ("[อาญา]", "69")],
    "ป้องกันตัว": [("[อาญา]", "68"), ("[อาญา]", "69")],
    "จำเป็น": [("[อาญา]", "67")],
    "เจตนา": [("[อาญา]", "59"), ("[อาญา]", "60"), ("[อาญา]", "61")],
    "เจตนาพลาด": [("[อาญา]", "60")],
    "ประมาท": [("[อาญา]", "59"), ("[อาญา]", "291"), ("[อาญา]", "300")],
    "พยายาม": [("[อาญา]", "80"), ("[อาญา]", "81"), ("[อาญา]", "82")],
    "ตัวการ": [("[อาญา]", "83"), ("[อาญา]", "84")],
    "ผู้สนับสนุน": [("[อาญา]", "86")],
    "บันดาลโทสะ": [("[อาญา]", "72")],
    "กรรมเดียว": [("[อาญา]", "90"), ("[อาญา]", "91")],
    "ริบทรัพย์": [("[อาญา]", "33"), ("[อาญา]", "34"), ("[อาญา]", "35"), ("[อาญา]", "36")],

    # ── Criminal code — specific offences ──
    "ลักทรัพย์": [("[อาญา]", "334"), ("[อาญา]", "335"), ("[อาญา]", "336")],
    "วิ่งราวทรัพย์": [("[อาญา]", "336")],
    "ชิงทรัพย์": [("[อาญา]", "339")],
    "ปล้นทรัพย์": [("[อาญา]", "340")],
    "กรรโชก": [("[อาญา]", "337"), ("[อาญา]", "338")],
    "รีดเอาทรัพย์": [("[อาญา]", "338")],
    "ฉ้อโกง": [("[อาญา]", "341"), ("[อาญา]", "342"), ("[อาญา]", "343"), ("[อาญา]", "344")],
    "ยักยอก": [("[อาญา]", "352"), ("[อาญา]", "353"), ("[อาญา]", "354")],
    "รับของโจร": [("[อาญา]", "357")],
    "ทำให้เสียทรัพย์": [("[อาญา]", "358"), ("[อาญา]", "359"), ("[อาญา]", "360")],
    "บุกรุก": [("[อาญา]", "362"), ("[อาญา]", "363"), ("[อาญา]", "364"), ("[อาญา]", "365")],
    "ทำร้ายร่างกาย": [("[อาญา]", "295"), ("[อาญา]", "296"), ("[อาญา]", "297"), ("[อาญา]", "298")],
    "ฆ่าผู้อื่น": [("[อาญา]", "288"), ("[อาญา]", "289"), ("[อาญา]", "290")],
    "ข่มขืน": [("[อาญา]", "276"), ("[อาญา]", "277")],
    "กระทำชำเรา": [("[อาญา]", "276"), ("[อาญา]", "277")],
    "อนาจาร": [("[อาญา]", "278"), ("[อาญา]", "279")],
    "ข่มขืนใจ": [("[อาญา]", "309")],
    "หมิ่นประมาท": [("[อาญา]", "326"), ("[อาญา]", "327"), ("[อาญา]", "328")],
    "หมิ่นพระบรมเดชานุภาพ": [("[อาญา]", "112")],
    "ปลอมเอกสาร": [("[อาญา]", "264"), ("[อาญา]", "265"), ("[อาญา]", "266"), ("[อาญา]", "267"), ("[อาญา]", "268")],
    "วางเพลิง": [("[อาญา]", "217"), ("[อาญา]", "218"), ("[อาญา]", "219"), ("[อาญา]", "220")],
    "แจ้งความเท็จ": [("[อาญา]", "137")],
    "สินบน": [("[อาญา]", "149"), ("[อาญา]", "150"), ("[อาญา]", "151")],
    "เบียดบัง": [("[อาญา]", "147"), ("[อาญา]", "148")],
    "ปฏิบัติหน้าที่โดยมิชอบ": [("[อาญา]", "157"), ("[อาญา]", "158")],

    # ── Civil code — persons & capacity ──
    "สภาพบุคคล": [("[แพ่ง]", "15"), ("[แพ่ง]", "16")],
    "บรรลุนิติภาวะ": [("[แพ่ง]", "19"), ("[แพ่ง]", "20")],
    "ผู้เยาว์": [("[แพ่ง]", "21"), ("[แพ่ง]", "22"), ("[แพ่ง]", "23"), ("[แพ่ง]", "24")],
    "คนไร้ความสามารถ": [("[แพ่ง]", "28"), ("[แพ่ง]", "29")],
    "วิกลจริต": [("[แพ่ง]", "29"), ("[แพ่ง]", "30")],
    "คนเสมือนไร้ความสามารถ": [("[แพ่ง]", "32"), ("[แพ่ง]", "33"), ("[แพ่ง]", "34")],
    "ภูมิลำเนา": [("[แพ่ง]", "37"), ("[แพ่ง]", "38"), ("[แพ่ง]", "39"), ("[แพ่ง]", "40"), ("[แพ่ง]", "41")],
    "สาบสูญ": [("[แพ่ง]", "62"), ("[แพ่ง]", "63"), ("[แพ่ง]", "64"), ("[แพ่ง]", "65")],

    # ── Civil code — juristic acts ──
    "นิติกรรม": [("[แพ่ง]", "149"), ("[แพ่ง]", "150"), ("[แพ่ง]", "151"), ("[แพ่ง]", "152")],
    "เจตนาลวง": [("[แพ่ง]", "154"), ("[แพ่ง]", "155")],
    "สำคัญผิด": [("[แพ่ง]", "156"), ("[แพ่ง]", "157"), ("[แพ่ง]", "158")],
    "กลฉ้อฉล": [("[แพ่ง]", "159"), ("[แพ่ง]", "160")],
    "ข่มขู่": [("[แพ่ง]", "164"), ("[แพ่ง]", "165")],

    # ── Civil code — limitation periods ──
    "อายุความ": [
        ("[แพ่ง]", "193/30"),
        ("[แพ่ง]", "193/9"), ("[แพ่ง]", "193/10"), ("[แพ่ง]", "193/12"),
    ],

    # ── Civil code — torts & obligations ──
    "ละเมิด": [("[แพ่ง]", "420"), ("[แพ่ง]", "421"), ("[แพ่ง]", "422"), ("[แพ่ง]", "423"), ("[แพ่ง]", "424")],
    "ดอกเบี้ย": [("[แพ่ง]", "7"), ("[แพ่ง]", "224"), ("[แพ่ง]", "654")],

    # ── Civil code — contracts ──
    "ซื้อขาย": [("[แพ่ง]", "453"), ("[แพ่ง]", "454"), ("[แพ่ง]", "455"), ("[แพ่ง]", "456")],
    "แลกเปลี่ยน": [("[แพ่ง]", "518"), ("[แพ่ง]", "519")],
    "เช่าทรัพย์": [("[แพ่ง]", "537"), ("[แพ่ง]", "538"), ("[แพ่ง]", "539"), ("[แพ่ง]", "540")],
    "เช่าซื้อ": [("[แพ่ง]", "572"), ("[แพ่ง]", "573"), ("[แพ่ง]", "574")],
    "ค้ำประกัน": [("[แพ่ง]", "680"), ("[แพ่ง]", "681"), ("[แพ่ง]", "685"), ("[แพ่ง]", "686")],
    "จำนอง": [("[แพ่ง]", "702"), ("[แพ่ง]", "703"), ("[แพ่ง]", "704")],
    "จำนำ": [("[แพ่ง]", "747"), ("[แพ่ง]", "748"), ("[แพ่ง]", "749")],
    "ตัวแทน": [("[แพ่ง]", "797"), ("[แพ่ง]", "798"), ("[แพ่ง]", "799")],

    # ── Civil code — family & succession ──
    "หมั้น": [("[แพ่ง]", "1435"), ("[แพ่ง]", "1437")],
    "สมรส": [("[แพ่ง]", "1448"), ("[แพ่ง]", "1449"), ("[แพ่ง]", "1450"), ("[แพ่ง]", "1451"), ("[แพ่ง]", "1452")],
    "หย่า": [("[แพ่ง]", "1516"), ("[แพ่ง]", "1517"), ("[แพ่ง]", "1518")],
    "มรดก": [("[แพ่ง]", "1599"), ("[แพ่ง]", "1600"), ("[แพ่ง]", "1601")],

    # ── Constitution — fundamentals ──
    "อำนาจอธิปไตย": [("[รธน]", "3")],
    "กฎหมายสูงสุด": [("[รธน]", "5")],
    "สิทธิและเสรีภาพ": [("[รธน]", "25"), ("[รธน]", "26"), ("[รธน]", "27"), ("[รธน]", "28")],
    "เสรีภาพ": [("[รธน]", "26"), ("[รธน]", "34")],
    "หน้าที่ของปวงชน": [("[รธน]", "50")],
    "หน้าที่ของรัฐ": [("[รธน]", "51"), ("[รธน]", "52"), ("[รธน]", "53"), ("[รธน]", "54"), ("[รธน]", "55"), ("[รธน]", "56")],

    # ── Constitution — institutions ──
    "รัฐสภา": [("[รธน]", "79"), ("[รธน]", "80")],
    "วุฒิสภา": [("[รธน]", "107"), ("[รธน]", "108"), ("[รธน]", "109")],
    "คณะรัฐมนตรี": [("[รธน]", "158"), ("[รธน]", "159"), ("[รธน]", "160"), ("[รธน]", "161"), ("[รธน]", "162")],
    "ศาลรัฐธรรมนูญ": [("[รธน]", "200"), ("[รธน]", "210"), ("[รธน]", "211"), ("[รธน]", "212"), ("[รธน]", "213")],
    "ปกครองส่วนท้องถิ่น": [("[รธน]", "249"), ("[รธน]", "250"), ("[รธน]", "251"), ("[รธน]", "252")],
    "แก้ไขรัฐธรรมนูญ": [("[รธน]", "255"), ("[รธน]", "256")],
}


def _fuzzy_match_keyword(query: str, keyword: str, max_dist: int = 1) -> bool:
    """Check if keyword appears in query, allowing up to max_dist missing/extra chars.

    Instead of full edit distance, uses a sliding-window approach optimized
    for the common Thai typo case where 1-2 characters are dropped.
    """
    if keyword in query:
        return True
    klen = len(keyword)
    # Only allow fuzzy matching for keywords with 5+ chars
    # Short keywords (3-4 chars like "หย่า") produce too many false positives
    if klen < 5:
        return False
    # Try matching with 1 char deleted from keyword (user dropped a char)
    for i in range(klen):
        shortened = keyword[:i] + keyword[i+1:]
        if shortened in query:
            return True
    # Try matching with 1 char deleted from each query window
    for start in range(len(query) - klen + 2):
        window = query[start:start + klen + 1]
        if len(window) >= klen:
            for i in range(len(window)):
                reduced = window[:i] + window[i+1:]
                if reduced == keyword:
                    return True
    return False


# ═══════════════════════════════════════════════════════════════════════
# Local Search
# ═══════════════════════════════════════════════════════════════════════

def _entity_embedding_matrix(kg: KnowledgeGraph) -> tuple[list[str], np.ndarray]:
    """Return (names, matrix) for entities that have embeddings."""
    names: list[str] = []
    vecs: list[list[float]] = []
    for name, ent in kg.entities.items():
        if ent.embedding is not None:
            names.append(name)
            vecs.append(ent.embedding)
    if not vecs:
        return [], np.empty((0, 0))
    return names, np.array(vecs, dtype=np.float32)


# Boost score weights for legal keyword expansion.
# Higher means an expansion-injected section dominates RRF-ranked entities.
_BOOST_BASE = 1.5
_BOOST_RANK_STEP = 0.1


def local_search(
    query: str,
    kg: KnowledgeGraph,
    top_k: int = config.LOCAL_SEARCH_TOP_K,
) -> SearchResult:
    """Multi-strategy entity retrieval (TF-IDF+BM25+Dense RRF) + graph traversal."""

    # 1. RRF fusion search (entities + text chunks)
    idx = _RetrievalIndex.get(kg)
    rrf_results = idx.search(query, top_k=top_k)

    matched_entities: list[tuple[str, float]] = []
    already: set[str] = set()

    # Separate entity results from chunk results
    for name, score, item_type in rrf_results:
        if item_type == "entity":
            if name not in already:
                matched_entities.append((name, score))
                already.add(name)
        elif item_type == "chunk":
            # Resolve chunk → entities sourced from this chunk
            chunk_entities = idx._chunk_section_map.get(name, [])
            for ename in chunk_entities:
                if ename not in already:
                    matched_entities.append((ename, score * 0.95))
                    already.add(ename)

    # 2. Legal query expansion – boost sections for known legal terms
    _LAW_PREFIXES = ["[อาญา]", "[แพ่ง]", "[รธน]"]
    # Sort expansion keys by length descending (match longest first)
    matched_keywords: set[str] = set()
    for keyword in sorted(_LEGAL_QUERY_BOOST, key=len, reverse=True):
        if _fuzzy_match_keyword(query, keyword):
            matched_keywords.add(keyword)

    # Disambiguation: longer matched keywords suppress shorter substrings
    # e.g. "ข่มขืนใจ" suppresses "ข่มขืน", "เจตนาลวง" suppresses "เจตนา"
    to_remove: set[str] = set()
    sorted_matched = sorted(matched_keywords, key=len, reverse=True)
    for i, longer in enumerate(sorted_matched):
        for shorter in sorted_matched[i + 1:]:
            if shorter in longer:
                to_remove.add(shorter)
    matched_keywords -= to_remove

    for keyword in matched_keywords:
        entries = _LEGAL_QUERY_BOOST[keyword]
        for idx_exp, (pfx, sec_num) in enumerate(entries):
            key = f"{pfx} มาตรา {sec_num}"
            if key not in kg.entities:
                continue
            # First entry in list = most relevant → highest boost
            boost = _BOOST_BASE + (len(entries) - idx_exp) * _BOOST_RANK_STEP
            if key in already:
                # Boost existing entry's score
                for i, (name, score) in enumerate(matched_entities):
                    if name == key:
                        matched_entities[i] = (name, max(score, boost))
                        break
            else:
                matched_entities.insert(0, (key, boost))
                already.add(key)

    # 3. Concept-based graph boost – use INVOLVES_CONCEPT edges
    for ename, ent in kg.entities.items():
        if ent.entity_type == "LEGAL_CONCEPT" and ent.name in query:
            # Find all sections linked to this concept
            if kg.graph.has_node(ename):
                for neighbor in kg.graph.neighbors(ename):
                    n_ent = kg.entities.get(neighbor)
                    if n_ent and n_ent.entity_type == "SECTION":
                        if neighbor in already:
                            for i, (name, score) in enumerate(matched_entities):
                                if name == neighbor:
                                    matched_entities[i] = (name, max(score, 1.2))
                                    break
                        else:
                            matched_entities.insert(0, (neighbor, 1.2))
                            already.add(neighbor)

    # 4. Keyword fallback – also match มาตรา numbers from the query
    section_nums = re.findall(r"มาตรา\s*(\d+(?:/\d+)?)", query)
    for sn in section_nums:
        bare = f"มาตรา {sn}"
        for pfx in _LAW_PREFIXES:
            key = f"{pfx} {bare}"
            if key in kg.entities and key not in already:
                matched_entities.insert(0, (key, 1.0))
                already.add(key)
        if bare in kg.entities and bare not in already:
            matched_entities.insert(0, (bare, 1.0))
            already.add(bare)

    # 5. Graph traversal – gather neighbors (limit to top entities to avoid explosion)
    entity_details: list[dict] = []
    visited_chunks: set[str] = set()
    traversal_limit = top_k * 3  # only graph-traverse top matches
    for idx_ent, (ename, score) in enumerate(matched_entities):
        ent = kg.entities.get(ename)
        if not ent:
            continue
        entity_details.append({
            "name": ent.name,
            "type": ent.entity_type,
            "description": ent.description,
            "score": score,
            "source_chunk_ids": ent.source_chunk_ids,
        })
        visited_chunks.update(ent.source_chunk_ids)

        # 1-hop neighbors (only for top entities)
        if idx_ent < traversal_limit:
            for neighbor in kg.graph.neighbors(ename):
                n_ent = kg.entities.get(neighbor)
                if n_ent:
                    entity_details.append({
                        "name": n_ent.name,
                        "type": n_ent.entity_type,
                        "description": n_ent.description,
                        "score": score * 0.8,
                    })
                    visited_chunks.update(n_ent.source_chunk_ids)

            # Edge descriptions
            for u, v, data in kg.graph.edges(ename, data=True):
                entity_details.append({
                    "name": f"({u}) -> ({v})",
                    "type": data.get("rel_type", "RELATED_TO"),
                    "description": data.get("description", ""),
                    "score": score * 0.7,
                })

    # 6. Assemble source text
    source_texts: list[str] = []
    for cid in visited_chunks:
        chunk = kg.text_chunks.get(cid)
        if chunk:
            source_texts.append(chunk.text)

    # 7. Build context – prioritize SECTION entities by score
    entity_details.sort(key=lambda x: x["score"], reverse=True)
    context_parts = ["== Entity Context =="]
    seen = set()
    for ed in entity_details:
        key = ed["name"]
        if key in seen:
            continue
        seen.add(key)
        context_parts.append(f"[{ed['type']}] {ed['name']}: {ed['description']}")

    context_parts.append("\n== Source Text ==")
    for st in source_texts[:5]:
        context_parts.append(st[:2000])

    return SearchResult(
        mode="local",
        context="\n".join(context_parts),
        entities=entity_details,
        source_chunks=list(visited_chunks),
    )


# ═══════════════════════════════════════════════════════════════════════
# Global Search
# ═══════════════════════════════════════════════════════════════════════

def global_search(
    query: str,
    kg: KnowledgeGraph,
    top_communities: int = config.GLOBAL_SEARCH_TOP_COMMUNITIES,
) -> SearchResult:
    """Community-report map-reduce search."""
    if not kg.communities:
        return SearchResult(mode="global", context="ไม่พบ community data")

    # 1. Rank communities by relevance (simple keyword overlap + importance)
    q_lower = query.lower()
    scored: list[tuple[Community, float]] = []
    for comm in kg.communities:
        text_blob = f"{comm.title} {comm.summary} {' '.join(comm.findings)}".lower()
        overlap = sum(1 for w in q_lower.split() if w in text_blob)
        score = overlap + comm.importance
        scored.append((comm, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_communities]

    # 2. Map phase – extract relevant community reports (no LLM needed)
    reports: list[dict] = []
    context_parts: list[str] = ["== Community Reports =="]
    for comm, score in top:
        if not comm.summary:
            continue
        reports.append({"title": comm.title, "summary": comm.summary, "score": score, "findings": comm.findings})
        context_parts.append(f"\nหัวข้อ: {comm.title}\nสรุป: {comm.summary}")
        if comm.findings:
            context_parts.append(f"ข้อค้นพบ: {'; '.join(comm.findings)}")

    context = "\n".join(context_parts) if len(context_parts) > 1 else "ไม่พบข้อมูลที่เกี่ยวข้องจาก community reports"

    return SearchResult(
        mode="global",
        context=context,
        community_reports=reports,
    )


# ═══════════════════════════════════════════════════════════════════════
# Hybrid Search
# ═══════════════════════════════════════════════════════════════════════

def hybrid_search(
    query: str,
    kg: KnowledgeGraph,
) -> SearchResult:
    """Run local + global in parallel-style and combine."""
    local_res = local_search(query, kg)
    global_res = global_search(query, kg)

    combined_context = (
        "【ผลการค้นหาเฉพาะเจาะจง (Local Search)】\n"
        f"{local_res.context}\n\n"
        "【ภาพรวม (Global Search)】\n"
        f"{global_res.context}"
    )

    return SearchResult(
        mode="hybrid",
        context=combined_context,
        entities=local_res.entities,
        community_reports=global_res.community_reports,
        source_chunks=local_res.source_chunks,
    )
