"""Thai Legal GraphRAG – Streamlit Web Interface"""

import streamlit as st
import json
import time
from pathlib import Path

import config
from graphrag.graph_builder import KnowledgeGraph
from graphrag.query_engine import answer, extract_cited_sections

# ── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thai Legal GraphRAG",
    page_icon="⚖️",
    layout="wide",
)

# ── Load Knowledge Graph (cached) ──────────────────────────────────

@st.cache_resource
def load_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    parquet_dir = config.PARQUET_DIR
    if (parquet_dir / "entities.parquet").exists():
        kg.load_parquet()
        kg.compute_embeddings()
        return kg
    st.error("ไม่พบข้อมูล Knowledge Graph กรุณารัน `python main.py index` ก่อน")
    st.stop()


# ── Sidebar ─────────────────────────────────────────────────────────

st.sidebar.title("⚖️ Thai Legal GraphRAG")
st.sidebar.markdown("ระบบถาม-ตอบกฎหมายไทยด้วย Knowledge Graph")

mode = st.sidebar.selectbox(
    "Search Mode",
    ["hybrid", "local", "global"],
    format_func=lambda x: {
        "hybrid": "🔀 Hybrid (Local + Global)",
        "local": "🎯 Local Search",
        "global": "🌐 Global Search",
    }[x],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Local Search** – ค้นหาเฉพาะเจาะจง (มาตรา/ความผิด)\n\n"
    "**Global Search** – สรุปภาพรวม\n\n"
    "**Hybrid** – รวมทั้งสองโหมด"
)

# Clear chat button
if st.sidebar.button("🗑️ ล้างประวัติการสนทนา"):
    st.session_state.messages = []
    st.rerun()

# ── Graph Stats ─────────────────────────────────────────────────────

kg = load_kg()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Graph Statistics")
st.sidebar.metric("Entities", len(kg.entities))
st.sidebar.metric("Relationships", len(kg.relationships))
st.sidebar.metric("Communities", len(kg.communities))
st.sidebar.metric("Text Chunks", len(kg.text_chunks))

# Entity type distribution
type_counts = {}
for ent in kg.entities.values():
    type_counts[ent.entity_type] = type_counts.get(ent.entity_type, 0) + 1
if type_counts:
    st.sidebar.markdown("**Entity Types:**")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        st.sidebar.text(f"  {t}: {c}")

# ── Main Area ───────────────────────────────────────────────────────

st.title("⚖️ Thai Legal GraphRAG")
st.markdown("ระบบถาม-ตอบกฎหมายไทยโดยใช้ Knowledge Graph")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_assistant_msg(msg: dict):
    """Render an assistant message with optional metadata expander."""
    st.markdown(msg["content"])
    meta = msg.get("metadata")
    if meta:
        with st.expander("📋 รายละเอียดการค้นหา", expanded=False):
            st.markdown(f"**Mode:** {meta.get('mode', '?')}  |  **Latency:** {meta.get('elapsed', 0):.2f}s")

            cited = extract_cited_sections(msg["content"])
            if cited:
                st.markdown(
                    f"**มาตราที่อ้างอิง:** "
                    f"{', '.join(sorted(cited, key=lambda x: int(x.split('/')[0])))}"
                )

            entities = meta.get("entities", [])
            if entities:
                st.markdown("**Matched Entities:**")
                for ed in entities[:10]:
                    st.text(f"  [{ed['type']}] {ed['name']}: {ed['description'][:80]}…")

            reports = meta.get("community_reports", [])
            if reports:
                st.markdown("**Community Reports Used:**")
                for cr in reports:
                    st.text(f"  📄 {cr['title']}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_assistant_msg(msg)
        else:
            st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("ถามคำถามเกี่ยวกับกฎหมายไทย …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาและวิเคราะห์ …"):
            t0 = time.time()
            result = answer(prompt, kg, mode=mode)
            elapsed = time.time() - t0

        sr = result["search_result"]
        assistant_msg = {
            "role": "assistant",
            "content": result["answer"],
            "metadata": {
                "mode": result["mode"],
                "elapsed": elapsed,
                "entities": sr.entities[:10],
                "community_reports": sr.community_reports[:5],
            },
        }
        _render_assistant_msg(assistant_msg)

    st.session_state.messages.append(assistant_msg)

# ── Graph Visualisation Tab ─────────────────────────────────────────

st.markdown("---")

with st.expander("🕸️ Knowledge Graph Visualization", expanded=False):
    graph_json_path = config.GRAPH_OUTPUT_DIR / "knowledge_graph.json"
    if graph_json_path.exists():
        try:
            from pyvis.network import Network
            import tempfile
            import os

            graph_data = json.loads(graph_json_path.read_text(encoding="utf-8"))

            _TYPE_COLORS = {
                "LAW": "#e74c3c",
                "SECTION": "#3498db",
                "OFFENSE": "#e67e22",
                "PENALTY": "#9b59b6",
                "LEGAL_CONCEPT": "#2ecc71",
                "ORGANIZATION": "#1abc9c",
                "PERSON_TYPE": "#f1c40f",
                "COURT": "#34495e",
                "LEGAL_PROCEDURE": "#95a5a6",
            }

            max_nodes = st.slider("Max nodes to display", 10, 200, 50)

            net = Network(height="600px", width="100%", bgcolor="#1a1a2e", font_color="white")
            net.barnes_hut(gravity=-3000, central_gravity=0.3)

            for node in graph_data["nodes"][:max_nodes]:
                color = _TYPE_COLORS.get(node["type"], "#bdc3c7")
                net.add_node(
                    node["id"],
                    label=node["label"],
                    title=f"[{node['type']}] {node['description'][:100]}",
                    color=color,
                    size=15 if node["type"] in ("LAW", "OFFENSE") else 10,
                )

            node_ids = {n["id"] for n in graph_data["nodes"][:max_nodes]}
            for edge in graph_data["edges"]:
                if edge["source"] in node_ids and edge["target"] in node_ids:
                    net.add_edge(
                        edge["source"],
                        edge["target"],
                        title=f"[{edge['type']}] {edge['description'][:80]}",
                        width=edge.get("weight", 1) * 2,
                    )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w",
                                             encoding="utf-8") as f:
                net.save_graph(f.name)
                html_content = Path(f.name).read_text(encoding="utf-8")
            st.components.v1.html(html_content, height=620, scrolling=True)
            try:
                os.unlink(f.name)
            except PermissionError:
                pass

        except ImportError:
            st.info("ติดตั้ง pyvis เพื่อดู graph visualization: `pip install pyvis`")
    else:
        st.info("ยังไม่มีข้อมูล graph — กรุณารัน `python main.py index` ก่อน")
