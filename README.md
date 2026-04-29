# Thai Legal GraphRAG

[![Python](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-16%2F16%20passing-brightgreen.svg)](tests/)

ระบบถาม-ตอบกฎหมายไทยโดยใช้ **Graph-based Retrieval-Augmented Generation (GraphRAG)** พร้อม Knowledge Graph บน Neo4j

## Architecture

```
Raw Thai Law Texts (.txt)
        │
        ▼
Preprocessing (PyThaiNLP: normalize, tokenize, chunk)
        │
        ▼
Regex Entity/Relationship Extraction
        │
        ▼
Knowledge Graph (NetworkX) ──► Neo4j Export
        │
        ├─ Leiden Community Detection ──► Community Reports
        │
        └─ TF-IDF Embeddings (512-dim)
                │
                ▼
Retrieval Index (TF-IDF + BM25 + Dense RRF Fusion)
        │
        ▼
User Query ──► Search (Local / Global / Hybrid)
        │
        ▼
Gemini LLM Generation (or Heuristic Fallback) ──► Answer with Citations
```

### Core Components

| Module | Description |
|---|---|
| `graphrag/preprocessor.py` | Thai text normalization, section extraction, tokenization, chunking |
| `graphrag/regex_extractor.py` | Regex-based entity/relationship extraction with law-prefixed section IDs |
| `graphrag/graph_builder.py` | Knowledge graph construction, Leiden community detection, TF-IDF embeddings |
| `graphrag/search_engine.py` | Multi-strategy retrieval: Local (RRF fusion + graph traversal + legal query expansion), Global (community reports), Hybrid |
| `graphrag/query_engine.py` | Answer generation with section citations (Gemini API with heuristic fallback) |
| `scripts/neo4j_exporter.py` | Export knowledge graph to Neo4j |
| `eval/evaluation.py` | RAGAS-style metrics: Hit Rate, Context Precision/Recall, Faithfulness, Answer Relevancy, Citation F1 |
| `app.py` | Streamlit web interface with PyVis graph visualization |
| `main.py` | CLI entry point |
| `config.py` | Configuration (API keys, models, chunking params, entity/relationship types) |
| `graphrag/llm_utils.py` | Gemini API wrapper with retry logic and rate limiting |

### Legal Ontology

**Entity Types (7):** LAW, SECTION, OFFENSE, PENALTY, LEGAL_CONCEPT, ORGANIZATION, PERSON_TYPE

**Relationship Types (8):** CONTAINS, DEFINES, PRESCRIBES_PENALTY, REFERENCES, RELATED_TO, AGGRAVATED_FORM_OF, EXCEPTION_TO, APPLIES_TO

### Knowledge Graph Stats

| Metric | Value |
|---|---|
| Entities | 2,734 |
| Relationships | 10,124 |
| Communities | 10 |
| Text Chunks | 1,246 |
| Sections | 2,551 (อาญา 448 / แพ่ง 1,852 / รธน 251) |

Section IDs ใช้ prefix ป้องกันชื่อซ้ำข้ามกฎหมาย:
`[อาญา] มาตรา 334`, `[แพ่ง] มาตรา 420`, `[รธน] มาตรา 44`

## Setup

### 1. Install Dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# แก้ไข .env ใส่ Gemini API key และ Neo4j credentials
```

ตัวแปรที่จำเป็น:
```
GEMINI_API_KEY=your-gemini-api-key
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=text-embedding-004
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

### 3. (Optional) Start Neo4j

```bash
docker run -d --name neo4j-legal-graphrag \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  neo4j:5
```

## Usage

### Index (Build Knowledge Graph)

```bash
python main.py index
```

Pipeline:
1. Preprocess law texts ใน `Code of Laws/` (127 files -> 1,246 chunks)
2. Extract entities & relationships ด้วย regex patterns
3. Community detection (Leiden algorithm)
4. สร้าง community summaries (rule-based)
5. คำนวณ TF-IDF embeddings (512-dim)
6. Save เป็น Parquet + JSON

> **Note:** Indexing ใช้ regex extraction + TF-IDF ไม่ต้องใช้ API key

### Query

```bash
python main.py query "การลักทรัพย์มีโทษอะไร?" --mode hybrid
python main.py query "สรุปความผิดเกี่ยวกับทรัพย์" --mode global
python main.py query "มาตรา 334 คืออะไร?" --mode local
```

### Evaluate

```bash
# eval บนชุด original (87 คำถามที่ใช้ tune)
python main.py evaluate --qa-file output/qa/qa_dataset_original.json --modes local

# eval บนชุด held-out (25 คำถามใหม่ที่ไม่เคย tune)
python main.py evaluate --qa-file output/qa/qa_holdout.json --modes local

# k-fold cross-validation (mean ± std ข้าม 5 folds)
python -m scripts.kfold_eval --qa-file output/qa/qa_all.json --k 5 --mode local

# regression tests (11 unit + 5 retrieval snapshot)
python -m pytest tests/
```

### Export to Neo4j

```bash
python main.py export-neo4j
```

### Web Interface

```bash
streamlit run app.py
```

## Search Modes

| Mode | เหมาะกับ | กลไก |
|---|---|---|
| **Local** | คำถามเจาะจง (มาตรา/ความผิด) | RRF fusion (TF-IDF + BM25 + Dense) + legal query expansion + graph traversal |
| **Global** | คำถามภาพรวม/สรุป | จัดอันดับ community reports ตาม keyword overlap + importance |
| **Hybrid** | คำถามทั่วไป | รวมผล Local + Global |

### Local Search Pipeline

1. **RRF Fusion** — ค้นหาจาก 3 retriever (TF-IDF, BM25, Dense embedding) แล้วรวมคะแนนด้วย Reciprocal Rank Fusion
2. **Legal Query Expansion** — ขยายคำค้นด้วย dictionary ที่ map คำสำคัญ -> มาตราที่เกี่ยวข้อง (เช่น "ฉ้อโกง" -> ม.341-344)
3. **Concept-based Graph Boost** — ค้น LEGAL_CONCEPT entity แล้วดึง SECTION ที่เชื่อมอยู่
4. **Graph Traversal** — ไล่ 1-hop neighbors ของ entity ที่ได้คะแนนสูงสุด

## Data

ไฟล์กฎหมายอยู่ใน `Code of Laws/`:
- **ประมวลกฎหมายอาญา** — Criminal Code (448 มาตรา)
- **ประมวลกฎหมายแพ่งและพาณิชย์** — Civil and Commercial Code (1,852 มาตรา)
- **รัฐธรรมนูญแห่งราชอาณาจักรไทย พ.ศ. 2560** — Constitution of Thailand 2017 (251 มาตรา)

## Evaluation

### Metrics

| หมวด | Metric | คำอธิบาย |
|---|---|---|
| Retrieval | Hit Rate | มี ground truth section อยู่ใน context หรือไม่ |
| Retrieval | Context Precision | สัดส่วน context ที่เกี่ยวข้อง |
| Retrieval | Context Recall | ครอบคลุม ground truth sections ครบหรือไม่ |
| Generation | Faithfulness | คำตอบสอดคล้องกับ context (LLM-judge / keyword overlap fallback) |
| Generation | Answer Relevancy | คำตอบตรงคำถาม (LLM-judge / keyword overlap fallback) |
| Generation | Citation F1 | มาตราที่อ้างอิงตรงกับ ground truth |

### Results (Local Search)

**Honest evaluation** — หลัง audit เพื่อกำจัด answer-key memorization:

| Metric | Original (87Q) | Holdout (25Q) | K-fold (5×22, mean ± std) |
|---|---|---|---|
| Hit Rate | 0.954 | 0.880 | 0.937 ± 0.054 |
| Faithfulness | 0.762 | 0.745 | 0.758 ± 0.033 |
| Answer Relevancy | 0.495 | 0.498 | 0.496 ± 0.022 |
| Citation F1 | 0.522 | 0.565 | 0.532 ± 0.054 |

Gap ระหว่าง original/holdout เล็ก (Hit -0.07, F1 +0.04) → ระบบ generalize
ไม่ overfit ดูรายละเอียดที่ [Doc/Evaluation_Report.md](Doc/Evaluation_Report.md)

## Project Structure

```
Thai-Legal-GraphRAG/
├── app.py                  # Streamlit UI
├── main.py                 # CLI entry point
├── config.py               # Configuration
├── requirements.txt
├── .env.example
├── graphrag/
│   ├── preprocessor.py     # Text preprocessing & chunking
│   ├── regex_extractor.py  # Entity/relationship extraction
│   ├── graph_builder.py    # KG construction & community detection
│   ├── search_engine.py    # Local/Global/Hybrid search
│   ├── query_engine.py     # Answer generation
│   └── llm_utils.py        # Gemini API wrapper
├── eval/
│   └── evaluation.py       # RAGAS-style evaluation
├── scripts/
│   ├── kfold_eval.py       # K-fold cross-validation
│   ├── neo4j_exporter.py   # Neo4j export
│   └── dev/                # (gitignored) debug & tuning scratch
├── tests/
│   ├── test_units.py                # regex / F1 / k-fold splitter
│   └── test_retrieval_snapshot.py   # frozen retrieval canaries
├── output/
│   ├── parquet/            # Indexed graph (entities, relationships, communities)
│   ├── graph/              # knowledge_graph.json (NetworkX dump)
│   ├── qa/                 # QA datasets (original / holdout / all)
│   └── results/            # eval / k-fold output
├── Doc/
│   └── Evaluation_Report.md
└── Code of Laws/           # Raw Thai law text files
```
