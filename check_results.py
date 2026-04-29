import json

with open("output/qa/qa_all.json", "r", encoding="utf-8") as f:
    qa = json.load(f)

with open("output/results/eval_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for i, r in enumerate(data):
    secs = r.get("predicted_sections", [])
    gt = qa[i]["relevant_sections"]
    f1 = r["citation_f1"]
    pred_set = set(secs)
    gt_set = set(gt)
    overlap = pred_set & gt_set
    print(f"Q{i+1}: pred({len(secs)})={secs}  gt={gt}  overlap={overlap}  F1={f1:.2f}")
    if not overlap:
        # Show first 200 chars of answer to see what's going on
        ans = r.get("predicted_answer", "")
        cite_line = [l for l in ans.split("\n") if "มาตราที่เกี่ยวข้อง" in l]
        if cite_line:
            print(f"  CITE: {cite_line[0][:200]}")
