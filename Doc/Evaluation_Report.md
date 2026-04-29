# รายงานผลการประเมิน Thai Legal GraphRAG

> **อัปเดต 2026-04-29:** ปรับวิธีวัดให้ honest หลังการ audit — ลบ keyword
> ที่จำคำตอบเป็นรายข้อ, ลบ echo คำถามในคำตอบ, เพิ่ม semantic similarity
> ใน faithfulness/relevancy fallback, และ tighten citation regex.

## 1. สรุปผลลัพธ์ (หลังแก้)

| Metric                | Original Set (n=87) | Held-out Set (n=25) | ∆ (gen. gap) |
|-----------------------|---------------------|---------------------|--------------|
| **Hit Rate**          | 0.954               | 0.880               | −0.074       |
| **Context Recall**    | 0.954               | 0.880               | −0.074       |
| **Context Precision** | 0.053               | 0.046               | −0.007       |
| **Faithfulness**      | 0.762               | 0.745               | −0.017       |
| **Answer Relevancy**  | 0.495               | 0.498               | +0.003       |
| **Citation F1**       | 0.522               | 0.565               | +0.043       |

**Generalization gap ปิดสนิททุก metric** (|∆| ≤ 0.074) → ระบบทำงานสม่ำเสมอ
ระหว่างชุดที่ใช้ tune กับชุดที่ไม่เคยเห็นมาก่อน

## 2. ค่าก่อน/หลังการแก้

| Metric           | ก่อนแก้ (87Q) | หลังแก้ (87Q) | ก่อนแก้ (Holdout) | หลังแก้ (Holdout) |
|------------------|---------------|----------------|-------------------|--------------------|
| Hit Rate         | 1.000 ⚠       | 0.954          | 1.000 ⚠           | 0.880              |
| Faithfulness     | 0.961 ⚠       | 0.762          | 0.796             | 0.745              |
| Answer Relevancy | 1.000 ⚠       | 0.495          | 0.474             | 0.498              |
| Citation F1      | 0.995 ⚠       | 0.522          | 0.350             | 0.565              |

ค่าเดิมที่ติดเครื่องหมาย ⚠ ทั้งหมดเป็น artifact จากปัญหา methodology ที่
รายละเอียดอยู่ในหัวข้อ 3

## 3. ปัญหาเดิมและวิธีแก้

### 3.1 Citation F1 = 0.995 → 0.522 (ลบ keyword ที่จำคำตอบ)

**ปัญหา:** `_LEGAL_QUERY_BOOST` มี ~180 keys โดยส่วนใหญ่เป็น sentence
fragment / question-pattern ที่ถูกใส่เข้าไปเพื่อให้คำถามแต่ละข้อ map ไปยังมาตรา
เฉพาะที่เป็นคำตอบ — ตัวอย่าง:

- `"กลฉ้อฉลมีผลต่อนิติกรรม": [("[แพ่ง]", "159")]`
- `"คำวินิจฉัยของศาลรัฐธรรมนูญ": [("[รธน]", "211")]`
- `"นิติกรรมที่มีวัตถุประสงค์ต้องห้าม": [("[แพ่ง]", "150")]`

นี่คือ **answer-key memorization** — ระบบไม่ได้เรียนรู้ concept แต่ท่องคำตอบ
ของชุด tune จึงร่วงเมื่อเจอคำถามใหม่ (Holdout F1 = 0.350)

**การแก้:** ตัด dict เหลือ ~70 keys ที่เป็น **legal-concept term** จริง
(ลักทรัพย์, ฉ้อโกง, อายุความ, นิติกรรม, สมรส, ฯลฯ) เท่านั้น และให้ BM25 + Dense
จัดการ disambiguation

**ผล:**
- 87Q F1 ตกจาก 0.995 → 0.522 (เห็นค่าจริงโดยปราศจาก memorization)
- Holdout F1 ขึ้นจาก 0.350 → 0.565 (ไม่ถูก keyword ที่ over-specific รบกวนแล้ว)
- Generalization gap: 0.645 → 0.043

### 3.2 Answer Relevancy = 1.000 → 0.495 (ลบ echo คำถาม)

**ปัญหา:** `_format_context_answer()` echo บรรทัด `"คำถาม: {query}"`
ในคำตอบทุกครั้ง → keyword overlap ระหว่างคำตอบกับคำถามสูงเทียมทุกข้อ

**การแก้:** ลบ echo + strip บรรทัด `"คำถาม: ..."` ออกใน `answer_relevancy()`
ก่อนคำนวณ + ใช้ embedding cosine (multilingual MiniLM) เป็น fallback แทน
keyword overlap เพียงอย่างเดียว
> สูตร: `score = 0.6 × cosine(emb(answer), emb(question)) + 0.4 × keyword_overlap`

### 3.3 Faithfulness = 0.961 → 0.762 (เพิ่ม semantic component)

**ปัญหา:** keyword overlap-only fallback ให้คะแนนสูงเกินจริง — เพราะคำตอบมัก
copy เนื้อหามาจาก ground truth คำต่อคำ (ทุก section พิมพ์ตรงตามต้นฉบับ)

**การแก้:** ใช้สูตรเดียวกับ relevancy — `0.6 × emb_sim + 0.4 × keyword_overlap`
embedding cosine ลดอคติของการ copy คำซ้ำ และจับ semantic mismatch ได้มากขึ้น

**ข้อจำกัดที่เหลือ:** ทั้งคู่ยังเป็น **proxy metric** ไม่ใช่ LLM judge —
embedding similarity ไม่สามารถจับ factual hallucination ได้ตรงๆ
ควร re-run ด้วย Gemini เมื่อ credits กลับมา

### 3.4 Citation regex หลวม

**ปัญหา:** `_extract_cited_sections` ใช้ `r"(\d+(?:/\d+)?)"` →
จับเลขใดๆ ในบรรทัด เช่น "พ.ศ. 2560" กลายเป็น section "2560"

**การแก้:** เข้มงวดเป็น `r"มาตรา\s*(\d+(?:/\d+)?)"` — ต้องมี prefix
`มาตรา` ก่อนเลข

## 4. ความน่าเชื่อถือของแต่ละ metric (หลังแก้)

| Metric                    | ความน่าเชื่อถือ | หมายเหตุ |
|---------------------------|-----------------|----------|
| Hit Rate ≈ 0.95           | ✅ สูง          | วัดจาก ground-truth section ใน retrieved entities ตรงไปตรงมา |
| Context Recall ≈ 0.95     | ✅ สูง          | สอดคล้องกับ Hit Rate, top_k ใหญ่พอเก็บคำตอบครบ |
| Context Precision ≈ 0.05  | ✅ สูง (แต่ต่ำ) | ค่าต่ำสะท้อนของจริง — top_k ใหญ่ทำให้ noise เยอะ |
| Faithfulness ≈ 0.76       | ⚠️ กลาง         | semantic+keyword blend, ยังเป็น proxy |
| Answer Relevancy ≈ 0.50   | ⚠️ กลาง         | กว้างขึ้นด้วย embedding แต่ไม่เท่า LLM judge |
| Citation F1 ≈ 0.52–0.57   | ✅ สูง          | gap ปิดสนิท → สะท้อนประสิทธิภาพจริง |

## 5. Trade-off ที่สำคัญ

`top_k` ใหญ่ → recall สูง (0.95) แต่ precision ต่ำ (0.05)
ถ้าต้องการดัน F1 ต้องเลือกหนึ่งใน:

- ลด `top_k` ของ section ที่ส่งเข้า citation line
- ใช้ reranker (เช่น cross-encoder) บน top-50 → คัดเหลือ top-3
- เพิ่ม community-report-based re-ranking

ไม่ควรเพิ่ม keyword boost เพิ่ม — เพราะนั่นคือทางลัดที่นำไปสู่ memorization อีกครั้ง

## 6. ข้อเสนอแนะถัดไป

1. **Reranker:** เพิ่ม cross-encoder rerank step หลัง RRF fusion เพื่อดัน
   citation precision โดยไม่ต้อง memorize keyword
2. **LLM judge:** เมื่อ Gemini ใช้ได้อีกครั้ง re-run faithfulness/relevancy
   เพื่อ calibrate proxy metrics ปัจจุบัน
3. **Question-type breakdown:** แยกสถิติตามประเภทคำถาม (definition / penalty
   / procedure) เพื่อระบุจุดอ่อน
4. **Hard-negative mining:** จาก fold ที่ผลแย่ที่สุดใน k-fold ใช้เป็นชุด
   training สำหรับ reranker ในอนาคต

## 7. Method

- **Original set:** 87 คำถาม (81 single-GT + 6 multi-GT)
- **Held-out set:** 25 คำถามใหม่ — มาตราที่ไม่เคยอยู่ใน original
  (อาญา 10, แพ่ง 8, รัฐธรรมนูญ 7)
- **Retrieval stack:** TF-IDF + BM25 + Dense (paraphrase-multilingual-MiniLM-L12-v2)
  → RRF (k=60) → graph traversal
- **Tokenizer:** pythainlp `newmm`
- **Eval mode:** `local` (Gemini API depleted, semantic+keyword fallback)
- **Boost dictionary:** ~70 concept-only keys (ลดจาก ~180)

## 8. รายการการแก้ไขในรอบนี้ (audit fix)

1. ลบ ~110 question-specific keywords ออกจาก `_LEGAL_QUERY_BOOST`
2. ลบบรรทัด `parts.append(f"คำถาม: {query}")` ใน `_format_context_answer()`
3. เข้มงวด `_extract_cited_sections` regex ให้ต้องมี prefix `มาตรา`
4. Sync system prompt กับ parser format (`**มาตราที่เกี่ยวข้อง:**`)
5. ลบ unused `entity_name_filter` parameter ใน 3 ฟังก์ชัน
6. แก้ indentation bug (12-space → 8-space) ใน expansion loop
7. ตั้งชื่อ magic numbers เป็น `_BOOST_BASE` / `_BOOST_RANK_STEP`
8. เพิ่ม `threading.Lock` ใน `_RetrievalIndex` (thread-safe singleton)
9. เพิ่ม embedding cosine ในสูตร faithfulness และ answer_relevancy
10. ลบไฟล์ duplicate `*-Kx_NB.py` (2 ไฟล์)
11. ปรับ `_keyword_overlap` stop-words: ย้ายเป็น module-level set, เพิ่ม
    template/markdown noise tokens, แยก `_tokens()` เป็น helper
12. เพิ่ม disk cache สำหรับ `dense_mat` (`output/parquet/dense_mat_<hash>.npy`)
    — invalidates อัตโนมัติเมื่อ texts/model เปลี่ยน, build time 30s → 14s
13. เพิ่ม `scripts/kfold_eval.py` — k-fold cross-validation พร้อม mean±std
    summary และ per-fold JSON output
14. เพิ่ม `tests/` — 11 unit tests (regex/F1/k-fold splitter) + 5 retrieval
    snapshot tests, รัน `python -m pytest tests/`

### K-fold validation results (k=5, mode=local, seed=42, n=112)

| Metric | Mean | ± Std | Min | Max |
|---|---|---|---|---|
| hit_rate | 0.937 | 0.054 | 0.864 | 1.000 |
| faithfulness | 0.758 | 0.033 | 0.716 | 0.814 |
| answer_relevancy | 0.496 | 0.022 | 0.458 | 0.520 |
| citation_f1 | 0.532 | 0.054 | 0.470 | 0.614 |

ความแปรปรวนต่ำใน 5 folds → ระบบ generalize อย่างแท้จริง ไม่ overfit ต่อชุด
ทดสอบใดชุดหนึ่ง
