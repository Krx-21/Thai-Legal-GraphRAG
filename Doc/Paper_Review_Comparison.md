# Related Work Review — Thai Legal GraphRAG

## สิ่งที่เราทำ (สรุปโปรเจกต์ของเรา)
- **หัวข้อ:** ไว้ก่อน
- **Target users:** Anyone
- **Design methodology:**
  - Data prep: NitiBench → test (n2n), ลอก text กฎหมายไทยมา
  - Training: หาวิธีที่เหมาะสมที่สุด
  - Evaluation: Hit Rate (ฝั่งดึงข้อมูล) + Legal Accuracy (ฝั่งตอบคำถาม)
- **คุณลักษณะเด่น:**
  - GraphRAG บนกฎหมายไทย + ภาษาไทย (PyThaiNLP)
  - Ontology เฉพาะ: LAW, SECTION, OFFENSE, PENALTY ใน Neo4j
  - Dual-level Search (Global + Local)
  - Evaluation ด้วย evaluate.py + RAGAS

---

## ตารางเปรียบเทียบ Paper ที่ยังไม่ได้ Review

### 1. Law GraphRAG: An Advanced Legal Question-Answering System

| หัวข้อ | รายละเอียด |
|---|---|
| **ชื่อ Paper** | Law GraphRAG: An Advanced Legal Question-Answering System |
| **แหล่งที่มา** | IEEE (Zhejiang University, China) |
| **เขาทำอะไรบ้าง** | สร้างระบบ Legal QA โดยใช้ GraphRAG บนกฎหมาย data compliance ของจีน (PIPL, Data Security Law, Cybersecurity Law ฯลฯ) ขั้นตอน: Text Chunking → Entity/Relationship Extraction → Community Detection → Bottom-up Retrieval → Answer Generation สร้างคำถาม 100 ข้อ evaluate ด้วย LLM-as-judge (win rate comparison กับ naive LLM) |
| **สิ่งที่เราต่างออกไป** | ref paper ทำบนกฎหมายจีน (data compliance เฉพาะทาง) เราทำบนกฎหมายไทยที่ครอบคลุมกว้างกว่า มี Ontology ที่ออกแบบเฉพาะ (LAW, SECTION, OFFENSE, PENALTY) ใช้ PyThaiNLP ในการตัดคำ **Evaluation:** ref paper ใช้ LLM ตัดสิน (win rate) ซึ่งมี bias เราใช้ Hit Rate + Legal Accuracy ที่วัดได้เป็นตัวเลขชัดเจนกว่า + ใช้ RAGAS framework **Graph Structure:** ref paper ใช้ general entity extraction เราออกแบบ node types เฉพาะสำหรับโครงสร้างกฎหมายไทย |

---

### 2. Retrieval-Augmented Generation with Graphs (GraphRAG) — Survey Paper

| หัวข้อ | รายละเอียด |
|---|---|
| **ชื่อ Paper** | Retrieval-Augmented Generation with Graphs (GraphRAG) |
| **แหล่งที่มา** | arxiv.org (arXiv:2501.00309, 88 หน้า — Michigan State University, Meta, Adobe Research, Snap Inc. ฯลฯ) |
| **เขาทำอะไรบ้าง (Methodology & Contributions)** | เป็น Comprehensive Survey ของ GraphRAG ที่เสนอ holistic framework ประกอบด้วย Query Processor, Retriever, Organizer, Generator แยกชัดเจน วิเคราะห์ 3 ความแตกต่างหลักจาก RAG ธรรมดา: (1) Homogeneous vs. Heterogeneous Structure (2) Independent vs. Interdependent Information (3) Domain Invariance vs. Domain-specific Information สำรวจ GraphRAG ใน 6+ domains (knowledge graph, social network, biological graph, scene graph ฯลฯ) รวบรวม benchmark datasets และ tool resources |
| **สิ่งที่เราต่างออกไป / ข้อได้เปรียบของเรา (Thai Legal GraphRAG)** | ref paper เป็น survey ไม่มี implementation จริง เราคือ practical system ที่นำแนวคิด GraphRAG มาสร้างจริงบน domain กฎหมายไทย ref paper ระบุว่า graph-structured data เป็น domain-specific (Difference 3) ซึ่งตรงกับแนวทางของเราที่ออกแบบ Ontology เฉพาะ (LAW, SECTION, OFFENSE, PENALTY) แทนที่จะใช้ general graph construction และเรามี evaluation จริงบน NitiBench ไม่ใช่แค่ทบทวนวรรณกรรม |

---

### 3. Challenges for Generative AI in Legal Reasoning

| หัวข้อ | รายละเอียด |
|---|---|
| **ชื่อ Paper** | Challenges for Generative AI in Legal Reasoning |
| **แหล่งที่มา** | arXiv.org |
| **เขาทำอะไรบ้าง (Methodology & Contributions)** | วิเคราะห์ความท้าทายของ GenAI/LLM ในการให้เหตุผลทางกฎหมาย ครอบคลุม: การเลือก legal framework ข้ามเขตอำนาจ, การแยก ratio decidendi กับ obiter dicta, การจัดการ general clauses (เช่น "ความสมเหตุสมผล"), การแก้ข้อขัดแย้งของบทกฎหมาย, ภาระการพิสูจน์ จากนั้น map เทคนิค RAG / Multi-agent / Neuro-symbolic AI เข้ากับแต่ละความท้าทาย เสนอ evaluation framework แบ่งเป็น normative, doctrinal, evidential, technical |
| **สิ่งที่เราต่างออกไป / ข้อได้เปรียบของเรา (Thai Legal GraphRAG)** | ref paper เป็นงานวิเคราะห์/ทฤษฎี ไม่มี implementation จริง เราคือ practical system ที่ลงมือสร้างจริง เราใช้ GraphRAG ซึ่ง ref paper ระบุว่าเป็นหนึ่งในเทคนิคที่ช่วยแก้ปัญหาเหล่านี้ได้ โดยเฉพาะการจัดโครงสร้างความสัมพันธ์ระหว่างมาตรา ข้อกฎหมาย และบทลงโทษ ช่วยลด hallucination ได้ตรงตามที่ ref paper เสนอ |

---

### 4. Legal Query RAG (LQ-RAG)

| หัวข้อ | รายละเอียด |
|---|---|
| **ชื่อ Paper** | Legal Query RAG |
| **แหล่งที่มา** | IEEE Access (2025) |
| **เขาทำอะไรบ้าง (Methodology & Contributions)** | เสนอ LQ-RAG framework ที่มี recursive feedback mechanism สำหรับ legal applications มี 4 องค์ประกอบ: (1) custom evaluation agent (2) specialized response generation model (3) prompt engineering agent (4) fine-tuned legal embedding LLM ได้ผลลัพธ์ Hit Rate เพิ่ม 13%, MRR เพิ่ม 15% และ generative LLM (HFM) ดีกว่า general LLMs 24% |
| **สิ่งที่เราต่างออกไป / ข้อได้เปรียบของเรา (Thai Legal GraphRAG)** | **รูปแบบ Retrieval:** ref paper ใช้ vector-based RAG ปกติ (embedding + retrieval) แต่เราใช้ Graph RAG ที่มองเห็นความสัมพันธ์เชิงโครงสร้างระหว่างมาตรากฎหมาย ทำให้ retrieve ได้แม่นยำกว่าในกรณีที่คำถามซับซ้อน **ภาษา:** ref paper ทำบนกฎหมายเกาหลีใต้ เราทำบนกฎหมายไทย **Evaluation:** เราใช้ Hit Rate เหมือนกัน แต่เพิ่ม Legal Accuracy เพื่อวัดความถูกต้องของคำตอบด้วย |

---

### 5. LegalRAG: A Hybrid RAG System for Multilingual Legal Information Retrieval

| หัวข้อ | รายละเอียด |
|---|---|
| **ชื่อ Paper** | LegalRAG: A Hybrid RAG System for Multilingual Legal Information Retrieval |
| **แหล่งที่มา** | IEEE (North South University, Bangladesh + Fordham University, USA) |
| **เขาทำอะไรบ้าง (Methodology & Contributions)** | สร้างระบบ RAG สองภาษา (English + Bangla) สำหรับ QA บน Bangladesh Police Gazettes ใช้ 2 pipeline: (1) Vanilla RAG — chunk → embed → retrieve → generate (2) Advanced RAG — เพิ่ม Llama 3.2 (3B) เป็น relevance check + query refinement agent ทำ iterative refinement สูงสุด 3 รอบ ใช้ OCR (PyTesseract) ประมวลผลเอกสาร ทดสอบกับ LLM 3 ตัว (Mixtral 8×7B, Llama 3.1 8B, Gemma 2 9B) บน 168 QA pairs Evaluate ด้วย human evaluation (1-5) + semantic similarity |
| **สิ่งที่เราต่างออกไป / ข้อได้เปรียบของเรา (Thai Legal GraphRAG)** | **Retrieval:** ref paper ใช้ vector-based RAG + relevance check agent เราใช้ Graph RAG ที่เห็นความสัมพันธ์เชิงโครงสร้างระหว่างมาตรากฎหมาย ไม่ต้องพึ่ง iterative query refinement **ภาษา:** ref paper ทำ Bangla + English (low-resource language) เราทำภาษาไทย (low-resource เช่นกัน) แต่ใช้ PyThaiNLP แทน OCR **โครงสร้างข้อมูล:** ref paper ใช้ Police Gazettes (เอกสารราชการแบบไม่มีโครงสร้าง) เราใช้กฎหมายที่มีโครงสร้างชัดเจน (LAW → SECTION → OFFENSE → PENALTY) ทำให้ graph มีคุณภาพสูงกว่า **Evaluation:** ref paper ใช้ human evaluation (subjective) เราใช้ Hit Rate + Legal Accuracy (objective + automated) |

---

### 6. A Survey on RAG with LLMs

| หัวข้อ | รายละเอียด |
|---|---|
| **ชื่อ Paper** | A Survey on RAG with LLMs |
| **แหล่งที่มา** | ScienceDirect, Procedia Computer Science (KES 2024) |
| **เขาทำอะไรบ้าง (Methodology & Contributions)** | เป็นปเปอร์แนว Survey ที่สำรวจเทคนิค RAG ร่วมกับ LLM ในบริบท Digital Transformation ครอบคลุมการ integrate external data retrieval เข้ากับ text generation เพื่อเพิ่มความถูกต้องและความเกี่ยวข้องของคำตอบ วิเคราะห์ข้อดี ข้อเสีย และแนวโน้มของ RAG |
| **สิ่งที่เราต่างออกไป / ข้อได้เปรียบของเรา (Thai Legal GraphRAG)** | ref paper เป็น survey ทั่วไปเกี่ยวกับ RAG ไม่เจาะ domain ใดโดยเฉพาะ เราคือ domain-specific implementation ที่ใช้ Graph RAG (ไม่ใช่แค่ RAG ธรรมดา) บน domain กฎหมายไทย มีการ evaluate จริงบน benchmark (NitiBench) ไม่ใช่แค่ทบทวนวรรณกรรม |

---

### 7. GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation

| หัวข้อ | รายละเอียด |
|---|---|
| **ชื่อ Paper** | GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation |
| **แหล่งที่มา** | arxiv.org (Monash University, Griffith University ฯลฯ) |
| **เขาทำอะไรบ้าง (Methodology & Contributions)** | เสนอ Graph Foundation Model (GFM) ขนาด 8M parameters สำหรับ RAG ใช้ GNN ที่ reason ผ่านโครงสร้างกราฟ ผ่านการ train 2 ขั้นตอนบน 60 knowledge graphs (14M+ triples, 700K+ documents) เป็น foundation model ตัวแรกที่ใช้กับ dataset ใหม่ได้โดยไม่ต้อง fine-tune ทดสอบบน 3 multi-hop QA datasets + 7 domain-specific RAG datasets ได้ SOTA |
| **สิ่งที่เราต่างออกไป / ข้อได้เปรียบของเรา (Thai Legal GraphRAG)** | **ความเชื่อมโยง:** ref paper เป็น general-purpose graph foundation model เราสามารถนำแนวคิดของ GFM มาปรับใช้กับ domain กฎหมายไทยได้ **ข้อแตกต่าง:** ref paper ไม่ได้ focus ที่ legal domain เราออกแบบ graph structure เฉพาะสำหรับกฎหมายไทย (Ontology: LAW, SECTION, OFFENSE, PENALTY) ซึ่ง domain-specific knowledge ทำให้แม่นยำกว่า general model ใน legal context |

---

### 8. LegalOne: A Family of Foundation Models for Reliable Legal Reasoning

| หัวข้อ | รายละเอียด |
|---|---|
| **ชื่อ Paper** | LegalOne: A Family of Foundation Models for Reliable Legal Reasoning |
| **แหล่งที่มา** | arxiv.org (Tsinghua University) |
| **เขาทำอะไรบ้าง (Methodology & Contributions)** | สร้าง foundation model ตระกูล LegalOne สำหรับกฎหมายจีน ผ่าน 3 ขั้นตอน: (1) Mid-training ด้วย Plasticity-Adjusted Sampling (PAS) เพื่อ domain adaptation (2) SFT ด้วย Legal Agentic CoT Distillation (LEAD) เพื่อ distill reasoning จาก legal texts (3) Curriculum Reinforcement Learning ผ่าน memorization → understanding → reasoning ได้ SOTA บน legal tasks หลายตัว |
| **สิ่งที่เราต่างออกไป / ข้อได้เปรียบของเรา (Thai Legal GraphRAG)** | ref paper เน้นสร้าง foundation model ใหม่ทั้งตัว (pre-train + fine-tune + RL) สำหรับกฎหมายจีน เราไม่ได้สร้าง model ใหม่ แต่ใช้ GraphRAG เป็น retrieval layer บน LLM ที่มีอยู่ ทำให้ lightweight กว่าและ deploy ง่ายกว่า กฎหมายจีนกับไทยมีโครงสร้างต่างกัน Ontology ของเราออกแบบมาเฉพาะสำหรับกฎหมายไทย |

---

### 9. Knowledge Graph-Assisted LLM Post-Training for Enhanced Legal Reasoning

| หัวข้อ | รายละเอียด |
|---|---|
| **ชื่อ Paper** | Knowledge Graph-Assisted LLM Post-Training for Enhanced Legal Reasoning |
| **แหล่งที่มา** | arxiv.org (Thomson Reuters) |
| **เขาทำอะไรบ้าง (Methodology & Contributions)** | ใช้ Knowledge Graph ช่วย post-training LLM เพื่อเพิ่มความสามารถ legal reasoning สร้าง KG ตาม IRAC framework (Issue, Rule, Analysis, Conclusion) จาก 12K legal cases จากนั้นสร้าง training data จาก KG แล้วทำ SFT + DPO กับ LLM 3 ตัว (30B, 49B, 70B) ได้ผลดีกว่า baseline บน 4/5 legal benchmarks โดยเฉพาะ 70B DPO model ชนะ 4/6 reasoning tasks แม้เทียบกับ 141B legal LLM |
| **สิ่งที่เราต่างออกไป / ข้อได้เปรียบของเรา (Thai Legal GraphRAG)** | ref paper ใช้ KG สำหรับ training (post-training LLM) แต่เราใช้ KG สำหรับ retrieval (GraphRAG) ซึ่งเป็นคนละจุดประสงค์: ref paper ปรับปรุงตัว model เอง เราปรับปรุงข้อมูลที่ส่งเข้า model ref paper ทำบนกฎหมาย common law (ใช้ IRAC) ของเราทำบนกฎหมายไทย (civil law) ที่มีโครงสร้างต่างกัน เราไม่ต้อง fine-tune LLM ทำให้ใช้ได้กับ LLM ตัวใดก็ได้ |

---

## สรุปภาพรวม

| Paper | ประเภท | Domain/ภาษา | ใช้ Graph? | มี Evaluation? | จุดต่างหลักจากเรา |
|---|---|---|---|---|---|
| Law GraphRAG | System | กฎหมายจีน (data compliance) | ✅ GraphRAG | ✅ LLM-as-judge (win rate) | เราทำไทย + Ontology เฉพาะ + metric ชัดกว่า |
| GraphRAG (Survey) | Survey | ทั่วไป (6+ domains) | ✅ | ❌ (survey) | เราเป็น implementation จริง + domain-specific |
| Challenges for GenAI | Analysis/Theory | กฎหมายทั่วไป | ❌ | ❌ (เสนอ framework) | เราเป็น practical system |
| Legal Query RAG | System | กฎหมายเกาหลี | ❌ Vector RAG | ✅ Hit Rate, MRR | เราใช้ Graph RAG ไม่ใช่ vector RAG |
| LegalRAG (Hybrid) | System | กฎหมาย Bangladesh (Bangla+EN) | ❌ Vector RAG + Agent | ✅ Human eval + Semantic sim | เราใช้ Graph RAG + Ontology เฉพาะ + objective metrics |
| A Survey on RAG | Survey | ทั่วไป | ❌ | ❌ (survey) | เราเป็น domain-specific implementation |
| GFM-RAG | Foundation Model | ทั่วไป | ✅ GNN | ✅ SOTA | เราเจาะ legal domain เฉพาะ |
| LegalOne | Foundation Model | กฎหมายจีน | ❌ | ✅ SOTA | เราไม่สร้าง model ใหม่ ใช้ GraphRAG แทน |
| KG-Assisted Post-Training | Training Method | Common Law | ✅ KG for Training | ✅ 4/5 benchmarks | เราใช้ KG สำหรับ retrieval ไม่ใช่ training |

---

## Paper ที่ PDF อ่านไม่ได้ (image-based — ต้องโหลดใหม่หรือใช้ OCR)
- **LegalRAG A Hybrid RAG System for Multilingual.pdf** — image-based PDF (มีไฟล์ใหม่ LegalRAG_A_Hybrid_...pdf แล้ว)

---

## สรุปท้ายการ Review (สำหรับนำเสนอ)

### Research Gap ที่พบจาก Related Work ทั้งหมด (24 papers)

จากการ review งานวิจัยที่เกี่ยวข้องทั้งหมด พบ **5 ช่องว่าง (Gap)** สำคัญที่ยังไม่มีงานวิจัยใดตอบได้ครบ:

| # | Gap | หลักฐานจาก Related Work |
|---|---|---|
| **1** | **ไม่มี GraphRAG สำหรับกฎหมายไทย** | งานที่ทำ Legal GraphRAG มีเฉพาะจีน (Law GraphRAG), บราซิล (Graph RAG for Legal Norms), เมียนมา (Myanmar Law Cases), อินเดีย (NyayGraph, JudgmentGraph), สหรัฐ (GraphRAG Makes it Possible) — ไม่มีใครทำภาษาไทยเลย |
| **2** | **ไม่มี Ontology เฉพาะโครงสร้างกฎหมายไทย** | งานที่ใช้ graph ส่วนใหญ่ใช้ general entity extraction (Law GraphRAG, WEB3 DATA ANALYTICS) หรือ Ontology ของระบบกฎหมายอื่น (IRAC ของ common law, UMLS ของการแพทย์) — ไม่มีใครออกแบบ node types เฉพาะสำหรับกฎหมายไทย (LAW, SECTION, OFFENSE, PENALTY) |
| **3** | **Legal RAG ภาษาไทยมีแค่ Vector-based** | งานที่ทำ RAG บนกฎหมายไทย (A Retrieval-Augmented Generation System, NitiBench) ใช้แค่ Vector Search / Local Search ธรรมดา — ยังไม่ยกระดับเป็น Graph RAG ที่เห็นความสัมพันธ์เชิงโครงสร้าง |
| **4** | **Evaluation ไม่ครอบคลุมทั้ง Retrieval + Generation** | หลาย paper วัดแค่ด้านเดียว: LegalBench-RAG วัดแค่ Retrieval, Law GraphRAG ใช้ LLM-as-judge (มี bias), LegalRAG ใช้ human eval (subjective) — น้อยมากที่วัดทั้ง Hit Rate (Retrieval) + Legal Accuracy (Generation) พร้อมกัน |
| **5** | **ไม่มี Dual-level Search บนกฎหมายไทย** | WEB3 DATA ANALYTICS ใช้ Dual-level Search (Global + Local) แต่บน Blockchain data ไม่ใช่กฎหมาย — ยังไม่มีใครนำเทคนิคนี้มาใช้กับกฎหมายไทยเพื่อตอบได้ทั้งคำถามกว้าง (Global) และเจาะรายมาตรา (Local) |

---

### สิ่งที่โปรเจกต์ของเราตอบ Gap เหล่านี้

| Gap | วิธีที่เราแก้ |
|---|---|
| ไม่มี GraphRAG ไทย | สร้าง **GraphRAG ตัวแรก**บนกฎหมายไทย ใช้ PyThaiNLP ตัดคำ + Neo4j เก็บกราฟ |
| ไม่มี Ontology ไทย | ออกแบบ **Ontology เฉพาะ** 4 ประเภท: LAW, SECTION, OFFENSE, PENALTY |
| มีแค่ Vector-based | **ยกระดับ**จาก Traditional RAG เป็น Graph RAG ที่เห็นความสัมพันธ์เชิงโครงสร้าง |
| Evaluation ไม่ครอบคลุม | วัด**ทั้ง 2 ฝั่ง**: Hit Rate (Retrieval) + Legal Accuracy (Generation) ด้วย RAGAS |
| ไม่มี Dual-level Search | ใช้ **Global Search** (สรุปภาพรวม) + **Local Search** (เจาะรายมาตรา) บนกฎหมายไทย |

---

### แนวโน้มสำคัญจาก Related Work

1. **Graph-based > Vector-based ใน Legal domain** — Paper ที่ใช้ Graph (Law GraphRAG, Myanmar, NyayGraph) แสดงให้เห็นว่าโครงสร้างกราฟช่วยจับความสัมพันธ์ระหว่าง entity ทางกฎหมายได้ดีกว่า vector similarity ธรรมดา ตรงกับที่ GraphRAG Survey ระบุว่า graph-structured data เป็น "domain-specific" (Difference 3)

2. **Low-resource language ยังเป็นความท้าทาย** — LegalRAG (Bangla), Myanmar Law Cases, NitiBench (Thai) แสดงว่าภาษา low-resource ต้องการเครื่องมือเฉพาะ (OCR, PyThaiNLP) และยังมีช่องว่างให้ปรับปรุง

3. **Evaluation ในงาน Legal ยังไม่มีมาตรฐาน** — มีทั้ง LLM-as-judge (Law GraphRAG), human eval (LegalRAG), Hit Rate/MRR (Legal Query RAG), RAGAS (เรา) — ชี้ให้เห็นว่าวงการยังหามาตรฐานร่วมกันไม่ได้ การที่เราวัดทั้ง Retrieval + Generation ถือเป็นแนวทางที่ครอบคลุม

4. **Foundation Model vs. RAG ยังเป็นที่ถกเถียง** — LegalOne และ KG-Assisted Post-Training เลือกทาง fine-tune model ใหม่ (ต้นทุนสูง) ในขณะที่เราเลือกทาง RAG (lightweight, ใช้ LLM ตัวใดก็ได้) ซึ่งเหมาะกับ target users: anyone ที่ต้องการ deploy ง่าย
