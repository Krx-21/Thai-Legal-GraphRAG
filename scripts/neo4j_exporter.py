"""Neo4j Knowledge Graph Exporter

Exports the in-memory KnowledgeGraph to a Neo4j graph database.
"""

from __future__ import annotations

from neo4j import GraphDatabase

import config
from graphrag.graph_builder import KnowledgeGraph


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


class Neo4jExporter:
    def __init__(
        self,
        uri: str = config.NEO4J_URI,
        user: str = config.NEO4J_USER,
        password: str = config.NEO4J_PASSWORD,
    ):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── Export ───────────────────────────────────────────────────────

    def export(self, kg: KnowledgeGraph, clear: bool = True):
        """Write KnowledgeGraph nodes + edges to Neo4j."""
        with self._driver.session() as session:
            if clear:
                session.run("MATCH (n) DETACH DELETE n")
                print("[Neo4j] Cleared existing data")

            # Create constraints for faster lookups
            for etype in config.ENTITY_TYPES:
                try:
                    session.run(
                        f"CREATE CONSTRAINT IF NOT EXISTS "
                        f"FOR (n:{etype}) REQUIRE n.name IS UNIQUE"
                    )
                except Exception:
                    pass

            # Nodes
            count_nodes = 0
            for name, ent in kg.entities.items():
                label = ent.entity_type
                session.run(
                    f"MERGE (n:{label} {{name: $name}}) "
                    f"SET n.description = $desc, n.color = $color, "
                    f"n.source_chunks = $chunks",
                    name=name,
                    desc=ent.description,
                    color=_TYPE_COLORS.get(label, "#bdc3c7"),
                    chunks=", ".join(ent.source_chunk_ids),
                )
                count_nodes += 1

            # Edges
            count_edges = 0
            for rel in kg.relationships:
                if rel.source not in kg.entities or rel.target not in kg.entities:
                    continue
                src_label = kg.entities[rel.source].entity_type
                tgt_label = kg.entities[rel.target].entity_type
                rel_type = rel.rel_type.replace(" ", "_").upper()
                session.run(
                    f"MATCH (a:{src_label} {{name: $src}}), (b:{tgt_label} {{name: $tgt}}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    f"SET r.description = $desc, r.weight = $weight",
                    src=rel.source,
                    tgt=rel.target,
                    desc=rel.description,
                    weight=rel.weight,
                )
                count_edges += 1

            # Community labels
            for comm in kg.communities:
                for eid in comm.entity_ids:
                    if eid in kg.entities:
                        session.run(
                            "MATCH (n {name: $name}) "
                            "SET n.community_id = $cid, n.community_level = $level",
                            name=eid,
                            cid=comm.community_id,
                            level=comm.level,
                        )

            print(f"[Neo4j] Exported {count_nodes} nodes, {count_edges} edges")

    # ── Query Helpers ───────────────────────────────────────────────

    def get_neighbors(self, name: str, depth: int = 1) -> list[dict]:
        """Retrieve node + neighbors up to given depth."""
        with self._driver.session() as session:
            result = session.run(
                f"MATCH path = (n {{name: $name}})-[*1..{depth}]-(m) "
                "RETURN path",
                name=name,
            )
            records = []
            for record in result:
                path = record["path"]
                for node in path.nodes:
                    records.append(dict(node))
                for rel in path.relationships:
                    records.append({
                        "source": rel.start_node["name"],
                        "target": rel.end_node["name"],
                        "type": rel.type,
                    })
            return records

    def cypher(self, query: str, **params) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(query, **params)
            return [dict(r) for r in result]
