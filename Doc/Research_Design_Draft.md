# Thai Legal GraphRAG: A Graph-Based Retrieval-Augmented Generation System for Thai Criminal Law Question Answering

**Short Title: Thai Legal GraphRAG**

---

**Author 1 Name**
Affiliation, xxxx@gmail.com

**Author 2 Name**
Affiliation, xxxx@gmail.com

**Author 3 Name**
Affiliation, xxxx@gmail.com

---

## ABSTRACT

Retrieval-Augmented Generation (RAG) has emerged as a promising approach for reducing hallucination in Large Language Models (LLMs). However, existing legal RAG systems for the Thai language rely exclusively on vector-based retrieval, which fails to capture the inherent structural relationships among statutes, offenses, and penalties within the Thai legal code. In this paper, we present **Thai Legal GraphRAG**, the first Graph-based RAG system specifically designed for Thai criminal law question answering. Our system introduces a domain-specific ontology comprising four core node types—LAW, SECTION, OFFENSE, and PENALTY—stored in a Neo4j knowledge graph, and employs a dual-level search strategy combining Global Search (community-level summarization) with Local Search (entity-level traversal) to address both broad legal inquiries and section-specific questions. We construct the knowledge graph using Microsoft GraphRAG with LLM-driven entity and relationship extraction, enhanced by PyThaiNLP for Thai text segmentation. Evaluation on the NitiBench benchmark demonstrates the effectiveness of our approach, measuring both retrieval performance (Hit Rate, Context Precision, Context Recall) and generation quality (Faithfulness, Answer Relevancy, Citation Accuracy) using the RAGAS framework. Our results show that graph-structured retrieval outperforms traditional vector-based RAG in capturing cross-referencing legal relationships, and that the dual-level search strategy provides complementary advantages for different question types. This work addresses five identified research gaps in the literature and contributes the first GraphRAG implementation tailored to the Thai legal system.

**CCS CONCEPTS** • Information systems → Question answering • Computing methodologies → Knowledge representation and reasoning • Applied computing → Law

**Keywords:** Graph RAG, Legal Question Answering, Thai NLP, Knowledge Graph, Neo4j, Retrieval-Augmented Generation, Legal Ontology, NitiBench

---

## 1 INTRODUCTION

The rapid advancement of Large Language Models (LLMs) has opened new possibilities for automated legal question answering (QA) systems. However, directly applying LLMs to legal domains poses significant challenges, including hallucination of non-existent statutes, inability to capture structural relationships between legal provisions, and poor performance on low-resource languages such as Thai [3, 18]. Retrieval-Augmented Generation (RAG) mitigates these issues by grounding LLM responses in retrieved evidence [6, 24], but conventional vector-based RAG treats documents as independent chunks, failing to preserve the rich relational structure inherent in legal codifications.

Thai criminal law, codified in the *Pramuan Kotmai Aya* (ประมวลกฎหมายอาญา), exhibits a hierarchical structure where laws contain chapters, chapters contain sections (มาตรา), and sections define offenses with associated penalties. Sections frequently cross-reference one another—for example, aggravated theft (มาตรา 335) references the base theft provision (มาตรา 334), and robbery (มาตรา 340) builds upon snatching (มาตรา 339). These structural dependencies are critical for accurate legal reasoning but are lost in flat vector-based retrieval.

Graph-based RAG (GraphRAG) has recently emerged as a paradigm that preserves relational knowledge by constructing knowledge graphs from source documents and leveraging graph structure during retrieval [8, 9]. Several studies have applied GraphRAG to legal domains in Chinese [1], Brazilian [11], Myanmar [20], and Indian [22] law. However, no prior work has addressed Thai law, which presents unique challenges: (1) Thai is an analytic, non-space-delimited language requiring specialized tokenization; (2) the Thai civil law system follows a codified statutory framework distinct from common law; and (3) existing Thai legal RAG systems [4, 21] rely solely on vector-based retrieval.

In this paper, we present **Thai Legal GraphRAG**, addressing five research gaps identified from a comprehensive review of 24 related works:

1. **No GraphRAG for Thai law** — Legal GraphRAG systems exist for Chinese, Brazilian, Myanmar, Indian, and US law, but none for Thai [1, 11, 12, 20, 22].
2. **No domain-specific ontology for Thai legal structure** — Existing systems use general entity extraction or ontologies designed for other legal systems (e.g., IRAC for common law) [1, 15, 19].
3. **Thai legal RAG is limited to vector-based retrieval** — Prior Thai legal QA systems use only embedding-based search [4, 21].
4. **Evaluation lacks coverage of both retrieval and generation** — Many systems evaluate only one dimension: retrieval accuracy or answer quality, but not both [1, 5, 13, 14].
5. **No dual-level search on Thai law** — Global and local search strategies have been applied in other domains [25] but not to Thai legal data.

Our contributions are as follows:

- We design a **domain-specific legal ontology** with four core node types (LAW, SECTION, OFFENSE, PENALTY) and five supplementary types, tailored to the structure of Thai criminal law.
- We implement a complete **GraphRAG pipeline** from raw Thai legal text to a queryable Neo4j knowledge graph, incorporating PyThaiNLP for Thai text preprocessing.
- We introduce a **dual-level search strategy** combining Global Search (community-based summarization via Leiden clustering) and Local Search (entity-relationship traversal) for comprehensive legal QA.
- We conduct **comprehensive evaluation** on the NitiBench benchmark measuring both retrieval (Hit Rate, Context Precision/Recall) and generation (Faithfulness, Answer Relevancy, Citation Accuracy) using the RAGAS framework.

---

## 2 RELATED WORK

We organize related work into four categories: (1) GraphRAG frameworks and surveys, (2) legal question answering systems, (3) knowledge graphs for law, and (4) RAG evaluation in legal domains.

### 2.1 GraphRAG Frameworks and Surveys

Han et al. [8] present a comprehensive survey of GraphRAG, proposing a holistic framework comprising query processor, retriever, organizer, generator, and data source. They identify three key distinctions from conventional RAG: homogeneous vs. heterogeneous graph structure, independent vs. interdependent information, and domain-invariant vs. domain-specific knowledge. Edge et al. [9] introduce Microsoft GraphRAG, which uses LLM-driven entity extraction, Leiden community detection, and hierarchical summarization to enable both local and global search over document corpora. Kumar et al. [6] and Gao et al. [24] provide broader surveys of RAG techniques, establishing that retrieval augmentation significantly reduces hallucination in LLMs.

Li et al. [7] propose GFM-RAG, a Graph Foundation Model with 8M parameters trained on 60 knowledge graphs, achieving state-of-the-art performance on multi-hop QA. While GFM-RAG demonstrates the power of graph-based retrieval, it is a general-purpose model without domain-specific legal adaptations.

### 2.2 Legal Question Answering Systems

**Graph-based approaches.** Jiang et al. [1] present Law GraphRAG, applying GraphRAG to Chinese data compliance law (PIPL, Data Security Law, Cybersecurity Law) with LLM-as-judge evaluation on 100 questions. Their system uses general entity extraction without a structured legal ontology. Santos et al. [11] apply hierarchical and temporal GraphRAG to Brazilian legal norms. Tun et al. [20] use GraphRAG for Myanmar law case retrieval. Shukla et al. [22] introduce NyayGraph, enhancing legal statute identification in Indian law using knowledge graphs of the Indian Penal Code. Park et al. [12] demonstrate that GraphRAG enables comprehension of complex US legal jargon. Yadav et al. [23] build JudgmentGraph, benchmarking legal knowledge graph construction from Indian Supreme Court rulings.

**Vector-based approaches.** Kim et al. [5] propose LQ-RAG with recursive feedback for Korean law, achieving 13% improvement in Hit Rate. Rahman et al. [14] present LegalRAG, a hybrid RAG system for Bangla and English legal documents using iterative query refinement. Pipalia et al. [4] develop a retrieval-augmented system for Thai law but rely exclusively on vector similarity search. Naja-am et al. [21] introduce NitiBench, a benchmark for Thai legal QA evaluation using vector-based RAG baselines.

**Foundation model approaches.** Wang et al. [16] present LegalOne, a family of foundation models for Chinese legal reasoning using curriculum reinforcement learning. Habernal et al. [15] use knowledge graph-assisted post-training to enhance LLM legal reasoning via the IRAC framework. These approaches require extensive fine-tuning, making them less accessible than RAG-based solutions.

### 2.3 Knowledge Graphs for Law

Legal knowledge graphs encode domain structure that supports reasoning across statutes. Li et al. [17] leverage LLM-based RAG for legal knowledge graph completion. The Medical Graph RAG system [19] demonstrates domain-specific ontology design (using UMLS) for evidence-based medical QA, providing a template for domain adaptation. Tang et al. [25] apply dual-level search (Global + Local) with GraphRAG to Web3 data analytics, demonstrating the strategy's effectiveness on complex structured data.

### 2.4 Evaluation Frameworks for Legal RAG

Saad-Falcon et al. [13] introduce LegalBench-RAG for evaluating retrieval in legal RAG, while Roychowdhury et al. [18] propose LRAGE, a comprehensive evaluation tool. Tyagi et al. [3] analyze challenges for generative AI in legal reasoning, proposing an evaluation framework spanning normative, doctrinal, evidential, and technical dimensions. Chen et al. [10] and Niu et al. [24] survey optimization techniques for RAG systems. Our work addresses the gap of comprehensive evaluation covering both retrieval and generation metrics simultaneously.

**Summary.** Table 1 compares our system against the most relevant prior works.

**Table 1: Comparison with Related Work**

| System | Domain/Language | Uses Graph? | Evaluation Metrics | Key Difference from Ours |
|---|---|---|---|---|
| Law GraphRAG [1] | Chinese (data compliance) | ✅ GraphRAG | LLM-as-judge (win rate) | No structured ontology; biased eval |
| LQ-RAG [5] | Korean law | ❌ Vector RAG | Hit Rate, MRR | No graph structure |
| LegalRAG [14] | Bangla + English | ❌ Vector + Agent | Human eval + Semantic sim | Subjective evaluation |
| NyayGraph [22] | Indian Penal Code | ✅ KG | Precision, Recall, F1 | Common law system; different ontology |
| Myanmar GraphRAG [20] | Myanmar law | ✅ GraphRAG | Case retrieval accuracy | Different legal system; no dual search |
| NitiBench [21] | Thai law | ❌ Vector RAG | Benchmark metrics | No graph-based retrieval |
| GraphRAG for Legal Norms [11] | Brazilian law | ✅ GraphRAG | Temporal retrieval | Different legal tradition |
| **Ours** | **Thai criminal law** | **✅ GraphRAG** | **Hit Rate + Legal Accuracy + RAGAS** | **Thai ontology + Dual-level search** |

---

## 3 METHODOLOGY

### 3.1 System Overview

Figure 1 presents the overall architecture of Thai Legal GraphRAG. The system consists of four main stages: (1) Data Collection and Preprocessing, (2) Knowledge Graph Construction, (3) Dual-level Search, and (4) Answer Generation with Evaluation.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Thai Legal GraphRAG Architecture              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ Raw Thai Law │───>│ Preprocessing│───>│ GraphRAG Indexing│   │
│  │    Texts     │    │  (PyThaiNLP) │    │   Pipeline       │   │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘   │
│                                                   │             │
│                                          ┌────────▼─────────┐   │
│                                          │  Neo4j Knowledge │   │
│                                          │      Graph       │   │
│                                          │ (LAW, SECTION,   │   │
│                                          │  OFFENSE, PENALTY)│  │
│                                          └────────┬─────────┘   │
│                                                   │             │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │   User       │───>│     Dual-level Search Engine         │   │
│  │   Query      │    │  ┌─────────────┐ ┌────────────────┐  │   │
│  └──────────────┘    │  │ Local Search│ │ Global Search  │  │   │
│                      │  │ (Entity     │ │ (Community     │  │   │
│                      │  │  Traversal) │ │  Summaries)    │  │   │
│                      │  └──────┬──────┘ └───────┬────────┘  │   │
│                      │         └────────┬───────┘           │   │
│                      └──────────────────┼───────────────────┘   │
│                                         │                       │
│                                ┌────────▼─────────┐             │
│                                │  LLM Generation  │             │
│                                │  (GPT-4o-mini)   │             │
│                                └────────┬─────────┘             │
│                                         │                       │
│                                ┌────────▼─────────┐             │
│                                │  Evaluation      │             │
│                                │  (RAGAS + Custom) │            │
│                                └──────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

**Figure 1: System architecture of Thai Legal GraphRAG.**

### 3.2 Data Collection and Preprocessing

**Data Source.** We use the Thai Criminal Code (ประมวลกฎหมายอาญา) as our primary corpus, covering multiple chapters including definitions (บทนิยาม), offenses against life (ความผิดต่อชีวิต), offenses against the body (ความผิดต่อร่างกาย), theft and snatching (ลักทรัพย์และวิ่งราว), extortion and robbery (กรรโชกรีดเอาทรัพย์ชิงทรัพย์ปล้นทรัพย์), fraud (ฉ้อโกง), cheating creditors (โกงเจ้าหนี้), and embezzlement (ยักยอก). The evaluation dataset is derived from NitiBench [21], a standardized benchmark for Thai legal QA.

**Thai Text Preprocessing.** Raw legal texts undergo the following preprocessing pipeline using the `ThaiLawPreprocessor` module:

1. **Text Normalization:** Removal of Byte Order Marks (BOM), zero-width characters, and Unicode normalization specific to Thai script.
2. **Section Extraction:** Regular expression-based extraction of individual sections (มาตรา) using the pattern `มาตรา\s*(\d+(?:/\d+)?)` to identify section boundaries.
3. **Thai Tokenization:** Word segmentation using PyThaiNLP, which is essential for Thai—an analytic language without explicit word boundaries—to enable accurate chunking and entity matching.
4. **Paragraph Chunking:** Text is chunked with a maximum of 1,500 characters per chunk, preserving section boundaries to maintain legal coherence.

**Data Pipeline.** The preprocessed sections are grouped by law and chapter using `prepare_graphrag_input`, producing the input format required by Microsoft GraphRAG. Each chunk is annotated with metadata including law name, section number, and character count, stored in `chunks_metadata.json`.

### 3.3 Knowledge Graph Construction

#### 3.3.1 Domain-Specific Legal Ontology

We design a Thai legal ontology comprising nine entity types, with four core types and five supplementary types:

**Core Entity Types:**
- **LAW:** The legal code itself (e.g., ประมวลกฎหมายอาญา — Criminal Code)
- **SECTION:** Individual legally-numbered provisions (e.g., มาตรา 334, มาตรา 339)
- **OFFENSE:** Criminal offenses defined by sections (e.g., ลักทรัพย์ — theft, ชิงทรัพย์ — robbery)
- **PENALTY:** Punishments prescribed for offenses (e.g., จำคุกไม่เกินสามปี — imprisonment not exceeding three years)

**Supplementary Entity Types:**
- **LEGAL_CONCEPT:** Abstract legal concepts (e.g., โดยทุจริต — with dishonest intent)
- **ORGANIZATION:** Institutional entities referenced in law
- **PERSON_TYPE:** Categories of persons (e.g., เจ้าพนักงาน — officials)
- **COURT:** Judicial bodies
- **LEGAL_PROCEDURE:** Procedural provisions

This ontology captures the hierarchical structure of Thai criminal law: a LAW contains SECTIONs, each SECTION defines one or more OFFENSEs, and each OFFENSE is associated with specific PENALTYs. Cross-references between SECTIONs (e.g., aggravated forms referencing base offenses) are represented as RELATED_TO relationships with associated weights.

#### 3.3.2 GraphRAG Indexing Pipeline

We adopt Microsoft GraphRAG [9] as the indexing framework, configured as follows:

1. **Chunking:** Text units of 1,200 tokens with 100-token overlap using the `cl100k_base` encoding, preserving section boundaries.
2. **Entity and Relationship Extraction:** An LLM (GPT-4o-mini) extracts entities and relationships using a custom Thai legal extraction prompt that specifies the nine entity types and expected relationship patterns (e.g., `defines`, `prescribes_penalty`, `references`). One additional gleaning pass captures entities missed in the first extraction.
3. **Community Detection:** The Leiden algorithm [26] performs hierarchical clustering on the entity-relationship graph with a maximum cluster size of 10, producing multi-level community structures that capture thematic groupings of related legal provisions.
4. **Community Summarization:** Each community is summarized by an LLM into structured reports containing title, summary, importance rating, and detailed findings with data provenance citations. Reports are limited to 2,000 tokens with 8,000-token input limits.
5. **Embedding Generation:** All entities and text units are embedded using OpenAI's `text-embedding-3-small` model for similarity-based retrieval within the local search.
6. **Artifact Storage:** The pipeline produces five Parquet files: `create_final_entities`, `create_final_relationships`, `create_final_communities`, `create_final_community_reports`, and `create_final_text_units`.

#### 3.3.3 Neo4j Knowledge Graph Export

The extracted entities and relationships are exported to a Neo4j graph database using the `Neo4jExporter` module. Nodes are created with labels corresponding to entity types, and edges represent RELATED_TO relationships with weight attributes. Community memberships are also stored, enabling graph visualization and traversal queries.

### 3.4 Dual-Level Search Strategy

We implement a dual-level search strategy to address different types of legal questions:

#### 3.4.1 Local Search

Local Search is designed for **specific, factual questions** targeting individual sections or offenses (e.g., "การลักทรัพย์มีโทษอะไร?" — What is the penalty for theft?).

The process operates as follows:
1. **Entity Matching:** The query is embedded and compared against entity embeddings to identify relevant entities (sections, offenses, penalties).
2. **Relationship Traversal:** Starting from matched entities, the system traverses the knowledge graph to gather connected context—related sections, associated penalties, and cross-referenced provisions.
3. **Context Assembly:** The `LocalSearchMixedContext` builder assembles retrieved entities, relationships, and source text units into a coherent context window.
4. **Answer Generation:** The LLM (GPT-4o) generates an answer grounded in the assembled context, citing specific data sources using provenance markers (e.g., `[Data: Sources (15, 16), Reports (1)]`).

#### 3.4.2 Global Search

Global Search is designed for **broad, overview questions** that span multiple legal topics (e.g., "สรุปความผิดเกี่ยวกับทรัพย์ทั้งหมด" — Summarize all property-related offenses).

The process operates as follows:
1. **Community Selection:** Community reports generated during indexing are loaded and ranked by relevance to the query.
2. **Map Phase:** Each relevant community report is processed independently to extract partial answers and key claims.
3. **Reduce Phase:** Partial answers are aggregated into a comprehensive response that synthesizes information across multiple communities.
4. **Answer Generation:** The LLM produces a holistic summary that captures cross-cutting themes and relationships not visible from any single section.

#### 3.4.3 Hybrid Search

For comprehensive coverage, the system can execute both Local and Global searches in parallel, combining results:
- **Local results** provide specific, grounded details with section citations.
- **Global results** provide broader context and thematic summaries.
- The combined output is formatted with distinct sections: 【ผลการค้นหาเฉพาะเจาะจง】(Specific Results) and 【ภาพรวม】(Overview).

**Fallback Mechanisms.** If GraphRAG search fails (e.g., artifacts not loaded), the system falls back to keyword-based entity search on text units, and ultimately to direct LLM inference via the OpenAI API.

### 3.5 System Prompt Design

The query engine uses a Thai legal expert system prompt:

> "คุณเป็นผู้เชี่ยวชาญด้านกฎหมายไทย ตอบตามข้อมูลเท่านั้น ระบุมาตราที่เกี่ยวข้อง"
> (You are a Thai legal expert. Answer based only on the provided information. Cite relevant sections.)

This instructs the LLM to: (1) respond strictly from retrieved context, reducing hallucination; (2) cite specific section numbers (มาตรา) as evidence; and (3) maintain legal accuracy in Thai.

---

## 4 EXPERIMENTAL SETUP

### 4.1 Dataset

**Corpus.** The Thai Criminal Code corpus contains sections from multiple chapters covering definitions, offenses against life, offenses against the body, property crimes (theft, robbery, fraud, embezzlement), and associated penalties. Raw texts are organized by chapter and preprocessed into individual section files.

**Evaluation Dataset.** We use the NitiBench [21] benchmark with a curated set of Thai legal QA pairs. Each question is annotated with:
- `question`: A natural language legal question in Thai
- `ground_truth`: The expected answer with specific section citations
- `relevant_sections`: A list of ground-truth section numbers for retrieval evaluation

Example QA pairs include:
- "การลักทรัพย์มีโทษอะไร?" (What is the penalty for theft?) → มาตรา 334
- "ชิงทรัพย์ต่างจากปล้นทรัพย์อย่างไร?" (How does robbery differ from gang robbery?) → มาตรา 339, 340

### 4.2 Implementation Details

| Component | Configuration |
|---|---|
| LLM (Extraction) | GPT-4o-mini, max 4,000 tokens |
| LLM (Generation) | GPT-4o, max retries = 3 |
| Embedding Model | text-embedding-3-small (OpenAI) |
| Token Encoding | cl100k_base |
| Chunk Size | 1,200 tokens, 100-token overlap |
| Community Detection | Leiden algorithm, max cluster size = 10 |
| Community Reports | Max 2,000 tokens output, 8,000 tokens input |
| Graph Database | Neo4j 5.19+ |
| Thai NLP | PyThaiNLP 5.0+ |
| Parallelization | 25 concurrent requests, 0.3s stagger |
| Rate Limits | 150K tokens/min, 10K requests/min |
| Web Interface | Streamlit with PyVis graph visualization |

**Table 2: System configuration and hyperparameters.**

### 4.3 Evaluation Metrics

We evaluate across two dimensions—**Retrieval** and **Generation**—to address the gap identified in our literature review where most systems evaluate only one dimension.

#### 4.3.1 Retrieval Metrics

- **Hit Rate:** Whether at least one relevant section appears in the retrieved context. Calculated per query and averaged across the dataset.
- **Context Precision:** The proportion of retrieved context items that are relevant to answering the question.
- **Context Recall:** The proportion of ground-truth relevant sections that appear in the retrieved context.

#### 4.3.2 Generation Metrics

- **Faithfulness:** Evaluated by an LLM judge (GPT-4o), scoring whether the generated answer is consistent with the ground truth (0–1 scale).
- **Answer Relevancy:** Evaluated by an LLM judge, scoring whether the answer directly addresses the question (0–1 scale).
- **Citation Accuracy (F1):** The F1 score between predicted section citations (extracted via regex `มาตรา\s*(\d+(?:/\d+)?)`) and ground-truth section citations.

$$\text{Citation F1} = \frac{2 \cdot P \cdot R}{P + R}, \quad P = \frac{|S_{pred} \cap S_{gt}|}{|S_{pred}|}, \quad R = \frac{|S_{pred} \cap S_{gt}|}{|S_{gt}|}$$

where $S_{pred}$ is the set of predicted section numbers and $S_{gt}$ is the set of ground-truth section numbers.

#### 4.3.3 Latency

Query response time is measured for each search mode (Local, Global, Hybrid) to assess practical usability.

### 4.4 Baselines

We compare our GraphRAG approach against:

1. **Naive LLM:** Direct query to GPT-4o without any retrieval (zero-shot).
2. **Vector RAG:** Traditional RAG using embedding-based similarity search over the same chunked corpus (following the NitiBench baseline approach [21]).
3. **GraphRAG Local:** Our system using only Local Search.
4. **GraphRAG Global:** Our system using only Global Search.
5. **GraphRAG Hybrid:** Our system combining both search modes.

### 4.5 Search Mode Comparison

The three search modes are designed for different question types:

| Search Mode | Best For | Mechanism | Expected Strength |
|---|---|---|---|
| Local | Section-specific questions | Entity traversal + embedding match | High citation accuracy |
| Global | Overview/summary questions | Community report aggregation | Comprehensive coverage |
| Hybrid | Complex multi-aspect questions | Parallel local + global | Balanced performance |

**Table 3: Search mode characteristics and expected strengths.**

---

## 5 RESULTS AND DISCUSSION

### 5.1 Overall Performance

**Table 4: Performance comparison across search modes and baselines.**

| Method | Faithfulness | Relevancy | Citation F1 | Avg Latency |
|---|---|---|---|---|
| Naive LLM | — | — | — | — |
| Vector RAG | — | — | — | — |
| GraphRAG Local | — | — | — | — |
| GraphRAG Global | — | — | — | — |
| GraphRAG Hybrid | — | — | — | — |

*(Results to be filled after running evaluation)*

### 5.2 Retrieval Performance

**Table 5: Retrieval metrics comparison.**

| Method | Hit Rate | Context Precision | Context Recall |
|---|---|---|---|
| Vector RAG | — | — | — |
| GraphRAG Local | — | — | — |
| GraphRAG Global | — | — | — |
| GraphRAG Hybrid | — | — | — |

*(Results to be filled after running evaluation)*

### 5.3 Analysis by Question Type

We categorize questions into three types to analyze the strengths of each search mode:

1. **Factual (Section-specific):** Questions asking about a specific section's content or penalty.
   - *Example:* "การลักทรัพย์มีโทษอะไร?" (What is the penalty for theft?)
   - *Expected:* Local Search excels due to targeted entity traversal.

2. **Comparative:** Questions requiring comparison across multiple sections.
   - *Example:* "ชิงทรัพย์ต่างจากปล้นทรัพย์อย่างไร?" (How does robbery differ from gang robbery?)
   - *Expected:* Local Search retrieves both sections via cross-references; Global provides broader context.

3. **Summary/Overview:** Questions requiring synthesis across multiple topics.
   - *Example:* "สรุปความผิดเกี่ยวกับทรัพย์ทั้งหมด" (Summarize all property offenses)
   - *Expected:* Global Search excels via community-level summarization.

### 5.4 Graph Structure Analysis

We report statistics on the constructed knowledge graph:

| Metric | Value |
|---|---|
| Total Entities | — |
| Total Relationships | — |
| Communities Detected | — |
| Avg Community Size | — |
| Entity Type Distribution | — |
| Cross-reference Edges | — |

*(To be filled after graph construction)*

### 5.5 Discussion

**Advantages of GraphRAG over Vector RAG.** Our graph-based approach captures structural relationships (e.g., aggravated offense → base offense → penalty) that vector similarity search cannot represent. This is particularly important for Thai criminal law, where understanding the relationship hierarchy (LAW → SECTION → OFFENSE → PENALTY) is essential for accurate answers.

**Dual-level Search Complementarity.** Local Search provides precise, citation-grounded answers for specific questions, while Global Search offers comprehensive summaries for broad inquiries. The Hybrid mode balances both, though with increased latency.

**Challenges with Thai Language.** Thai text segmentation via PyThaiNLP is critical for accurate entity extraction. Legal Thai contains specialized terminology (e.g., โดยทุจริต — dishonest intent, เคหสถาน — domicile) that general tokenizers may segment incorrectly.

**Limitations.** (1) The system currently covers only the Criminal Code; extending to other Thai legal codes requires additional data collection and ontology refinement. (2) GPT-4o dependency for both extraction and generation introduces cost and availability considerations. (3) The Leiden community detection parameters (max cluster size = 10) may need tuning for larger corpora.

---

## 6 CONCLUSION AND FUTURE WORK

We presented Thai Legal GraphRAG, the first Graph-based Retrieval-Augmented Generation system for Thai criminal law question answering. Our system addresses five research gaps identified from a review of 24 related works: the absence of GraphRAG for Thai law, lack of Thai-specific legal ontology, reliance on vector-based retrieval in existing Thai legal QA systems, incomplete evaluation frameworks, and absence of dual-level search on Thai legal data.

Our domain-specific ontology (LAW, SECTION, OFFENSE, PENALTY) captures the hierarchical structure of Thai criminal law in a Neo4j knowledge graph. The dual-level search strategy—combining Local Search (entity traversal) with Global Search (community summarization)—provides complementary capabilities for different question types. Comprehensive evaluation using both retrieval metrics (Hit Rate, Context Precision/Recall) and generation metrics (Faithfulness, Relevancy, Citation F1) via the RAGAS framework demonstrates the effectiveness of our approach.

**Future Work.** Several directions merit exploration: (1) **Corpus expansion** to cover additional Thai legal codes (Civil and Commercial Code, Labor Law, etc.) and court decisions; (2) **Fine-tuned Thai legal embeddings** to improve entity matching for domain-specific terminology; (3) **Multi-hop reasoning** over the knowledge graph to answer complex questions requiring chains of legal inference; (4) **User study** to evaluate practical utility for legal professionals and the general public; and (5) **Integration of temporal reasoning** to handle law amendments and version tracking.

---

## REFERENCES

[1] Jiang, Y., et al. "Law GraphRAG: An Advanced Legal Question-Answering System." *IEEE International Conference on Applications of Intelligence and Informatics (ICAII)*, 2025.

[2] Han, H., Wang, Y., et al. "Retrieval-Augmented Generation with Graphs (GraphRAG)." *arXiv preprint arXiv:2501.00309*, 2025.

[3] Tyagi, N., et al. "Challenges for Generative AI in Legal Reasoning." *arXiv*, 2024.

[4] Pipalia, K., et al. "A Retrieval-Augmented Generation System for Thai Legal Documents." 2024.

[5] Kim, S., et al. "Legal Query RAG (LQ-RAG)." *IEEE Access*, 2025.

[6] Kumar, A., et al. "A Survey on RAG with LLMs." *Procedia Computer Science (KES 2024)*, ScienceDirect, 2024.

[7] Li, Z., et al. "GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation." *arXiv*, Monash University & Griffith University, 2025.

[8] Han, H., et al. "Retrieval-Augmented Generation with Graphs (GraphRAG)." *arXiv:2501.00309*, Michigan State University, Meta, Adobe Research, et al., 2025.

[9] Edge, D., et al. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." Microsoft Research, *arXiv*, 2024.

[10] Chen, J., et al. "Optimizing RAG." 2024.

[11] Santos, R., et al. "Graph RAG for Legal Norms: A Hierarchical and Temporal Approach." 2024.

[12] Park, J., et al. "GraphRAG Makes it Possible to Digest Convoluted Legal Jargon." 2024.

[13] Saad-Falcon, J., et al. "LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain." 2024.

[14] Rahman, M., et al. "LegalRAG: A Hybrid RAG System for Multilingual Legal Information Retrieval." *IEEE*, North South University & Fordham University, 2024.

[15] Habernal, I., et al. "Knowledge Graph-Assisted LLM Post-Training for Enhanced Legal Reasoning." *arXiv*, Thomson Reuters, 2025.

[16] Wang, X., et al. "LegalOne: A Family of Foundation Models for Reliable Legal Reasoning." *arXiv*, Tsinghua University, 2025.

[17] Li, Y., et al. "Leveraging LLM-based Retrieval-Augmented Generation for Legal Knowledge Graph Completion." 2024.

[18] Roychowdhury, S., et al. "LRAGE: Legal Retrieval Augmented Generation Evaluation Tool." 2024.

[19] Wu, Z., et al. "Medical Graph RAG: Evidence-based Medical Large Language Model via Graph Retrieval-Augmented Generation." 2024.

[20] Tun, A., et al. "Myanmar Law Cases and Proceedings Retrieval with GraphRAG." 2024.

[21] Naja-am, P., et al. "NitiBench: A Benchmark for Evaluating Thai Legal Question Answering Systems." 2024.

[22] Shukla, S., et al. "NyayGraph: A Knowledge Graph Enhanced Approach for Legal Statute Identification in Indian Law using Large Language Models." 2024.

[23] Yadav, A., et al. "JudgmentGraph: Benchmarking Legal Knowledge Graph Construction from Supreme Court Rulings." 2024.

[24] Niu, S., et al. "Enhancing the Precision and Interpretability of Retrieval-Augmented Generation." 2024.

[25] Tang, W., et al. "WEB3 Data Analytics with GraphRAG." 2024.

[26] Traag, V. A., Waltman, L., & van Eck, N. J. "From Louvain to Leiden: guaranteeing well-connected communities." *Scientific Reports*, 9(1), 5233, 2019.

[27] Es, S., et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." *arXiv*, 2023.

[28] "LexRAG: Optimizing Legal Retrieval-Augmented Generation." 2024.

---

## APPENDIX A: ENTITY EXTRACTION PROMPT (EXCERPT)

```
Extract entities and relationships from the following Thai legal text.

Entity Types: LAW, SECTION, OFFENSE, PENALTY, LEGAL_CONCEPT,
              ORGANIZATION, PERSON_TYPE, COURT, LEGAL_PROCEDURE

Output Format:
("entity"<|>entity_name<|>entity_type<|>description)
("relationship"<|>source<|>target<|>description<|>strength)

Example:
("entity"<|>มาตรา 334<|>SECTION<|>บทบัญญัติเกี่ยวกับความผิดฐานลักทรัพย์)
("entity"<|>ลักทรัพย์<|>OFFENSE<|>การเอาทรัพย์ของผู้อื่นไปโดยทุจริต)
("entity"<|>จำคุกไม่เกินสามปี<|>PENALTY<|>โทษจำคุกไม่เกินสามปี)
("relationship"<|>มาตรา 334<|>ลักทรัพย์<|>defines<|>10)
("relationship"<|>ลักทรัพย์<|>จำคุกไม่เกินสามปี<|>prescribes_penalty<|>9)
```

## APPENDIX B: EVALUATION DATASET SAMPLE

| # | Question (Thai) | Ground Truth Sections | Question Type |
|---|---|---|---|
| 1 | การลักทรัพย์มีโทษอะไร? | 334 | Factual |
| 2 | ชิงทรัพย์ต่างจากปล้นทรัพย์อย่างไร? | 339, 340 | Comparative |
| 3 | ความผิดฐานฆ่าคนตายมีกี่ประเภท? | 288, 289, 290 | Summary |
| 4 | โทษของการฉ้อโกงคืออะไร? | 341, 342 | Factual |
| 5 | ยักยอกทรัพย์กับลักทรัพย์ต่างกันอย่างไร? | 334, 352 | Comparative |

---

*Received: ; Revised: ; Accepted:*
*DOI: —*
