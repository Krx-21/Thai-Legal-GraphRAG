"""Thai Legal GraphRAG – CLI Entry Point

Usage:
  python main.py index          # Preprocess + extract + build graph
  python main.py query "คำถาม"  # Ask a question
  python main.py evaluate       # Run evaluation on QA dataset
  python main.py export-neo4j   # Export graph to Neo4j
"""

from __future__ import annotations

import argparse
import re
import time
import json
import sys
from pathlib import Path

import config
from graphrag import llm_utils
from graphrag.preprocessor import process_all_laws, save_chunks_metadata, prepare_graphrag_input
from graphrag.graph_builder import KnowledgeGraph
from graphrag.query_engine import answer, extract_cited_sections
from eval.evaluation import load_qa_pairs, evaluate_single, evaluate_batch, save_results


def cmd_index(args):
    """Full indexing pipeline: preprocess → extract → communities → embed → save."""
    print("=" * 60)
    print("  Thai Legal GraphRAG — Indexing Pipeline")
    print("=" * 60)

    # 1. Preprocess
    print("\n[1/5] Preprocessing law texts …")
    chunks = process_all_laws()
    save_chunks_metadata(chunks)
    prepare_graphrag_input(chunks)

    # 2. Entity & Relationship Extraction
    print("\n[2/5] Extracting entities & relationships (regex) …")
    kg = KnowledgeGraph()
    kg.build_from_chunks(chunks, use_regex=True)

    # 3. Community Detection
    print("\n[3/5] Detecting communities …")
    kg.detect_communities()

    # 4. Community Summarisation
    print("\n[4/5] Summarizing communities …")
    kg.summarize_communities()

    # 5. Embeddings & Save
    print("\n[5/5] Computing embeddings & saving …")
    kg.compute_embeddings()
    kg.save_parquet()
    kg.save_graph_json()

    print("\n[OK] Indexing complete!")
    print(f"  Entities:      {len(kg.entities)}")
    print(f"  Relationships: {len(kg.relationships)}")
    print(f"  Communities:   {len(kg.communities)}")
    print(f"  Output:        {config.OUTPUT_DIR}")


def cmd_query(args):
    """Answer a single question."""
    kg = KnowledgeGraph()
    kg.load_parquet()
    kg.compute_embeddings()

    mode = args.mode
    query_text = args.query

    print(f"\nQuery: {query_text}")
    print(f"Mode:  {mode}\n")

    t0 = time.time()
    result = answer(query_text, kg, mode=mode)
    elapsed = time.time() - t0

    print("─" * 60)
    print(result["answer"])
    print("─" * 60)
    print(f"\nCited sections: {extract_cited_sections(result['answer'])}")
    print(f"Latency: {elapsed:.2f}s")


def cmd_evaluate(args):
    """Run evaluation on a QA dataset."""
    qa_path = Path(args.qa_file)
    if not qa_path.exists():
        print(f"QA file not found: {qa_path}")
        sys.exit(1)

    qa_pairs = load_qa_pairs(qa_path)
    print(f"Loaded {len(qa_pairs)} QA pairs from {qa_path}")

    kg = KnowledgeGraph()
    kg.load_parquet()
    kg.compute_embeddings()

    llm_status = "LLM" if llm_utils.is_available() else "Heuristic (LLM unavailable)"
    print(f"Faithfulness/Relevancy scoring: {llm_status}")

    modes = args.modes.split(",")
    all_results = []

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Evaluating mode: {mode}")
        print(f"{'='*60}")
        mode_results = []
        for i, qa in enumerate(qa_pairs, 1):
            t0 = time.time()
            res = answer(qa.question, kg, mode=mode)
            elapsed = time.time() - t0
            # Extract retrieved sections from entities (top-K by score,
            # SECTION-type only) so context_precision is meaningful.
            sr = res["search_result"]
            section_ents = [e for e in sr.entities if e.get("type") == "SECTION"]
            section_ents.sort(key=lambda e: e.get("score", 0) or 0.0, reverse=True)
            retrieved: list[str] = []
            for ed in section_ents[:20]:
                name = ed.get("name", "")
                secs = re.findall(r"มาตรา\s*(\d+(?:/\d+)?)", name)
                retrieved.extend(secs)
            eval_res = evaluate_single(qa, res["answer"], retrieved, mode, elapsed)
            mode_results.append(eval_res)
            print(f"  [{i}/{len(qa_pairs)}] {qa.question[:45]}")
            print(f"         hit={eval_res.hit_rate:.0f}  prec={eval_res.context_precision:.2f}  "
                  f"recall={eval_res.context_recall:.2f}  F1={eval_res.citation_f1:.2f}  "
                  f"faith={eval_res.faithfulness:.2f}  rel={eval_res.answer_relevancy:.2f}  "
                  f"{elapsed:.1f}s")

        all_results.extend(mode_results)
        agg = evaluate_batch(mode_results)
        print(f"\n  [{mode}] Aggregate:")
        for k, v in agg.items():
            print(f"    {k:>20s}: {v:.3f}" if isinstance(v, float) else f"    {k:>20s}: {v}")

    save_results(all_results)
    print(f"\n[OK] Evaluation complete! ({len(all_results)} results)")


def cmd_export_neo4j(args):
    """Export graph to Neo4j."""
    from scripts.neo4j_exporter import Neo4jExporter

    kg = KnowledgeGraph()
    kg.load_parquet()

    with Neo4jExporter() as exporter:
        exporter.export(kg, clear=not args.no_clear)

    print("[OK] Neo4j export complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Thai Legal GraphRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # index
    sub.add_parser("index", help="Run full indexing pipeline")

    # query
    p_query = sub.add_parser("query", help="Ask a question")
    p_query.add_argument("query", type=str, help="Question in Thai")
    p_query.add_argument("--mode", choices=["local", "global", "hybrid"],
                         default="hybrid", help="Search mode")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run evaluation")
    p_eval.add_argument("--qa-file", type=str,
                        default=str(config.QA_ALL),
                        help="Path to QA pairs JSON")
    p_eval.add_argument("--modes", type=str, default="local,global,hybrid",
                        help="Comma-separated search modes to evaluate")

    # export-neo4j
    p_neo4j = sub.add_parser("export-neo4j", help="Export to Neo4j")
    p_neo4j.add_argument("--no-clear", action="store_true",
                         help="Don't clear existing Neo4j data")

    args = parser.parse_args()
    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "export-neo4j":
        cmd_export_neo4j(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
