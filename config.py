"""Thai Legal GraphRAG - Configuration"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
LAW_DATA_DIR = BASE_DIR / os.getenv("LAW_DATA_DIR", "Code of Laws")
OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "output")
GRAPH_OUTPUT_DIR = OUTPUT_DIR / "graph"
PARQUET_DIR = OUTPUT_DIR / "parquet"
QA_DIR = OUTPUT_DIR / "qa"
QA_ALL = QA_DIR / "qa_all.json"
QA_HOLDOUT = QA_DIR / "qa_holdout.json"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in (OUTPUT_DIR, GRAPH_OUTPUT_DIR, PARQUET_DIR, QA_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Gemini ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
EMBEDDING_DIMENSION = 768  # Gemini text-embedding-004 dimension

# ── Neo4j ───────────────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ── Chunking ────────────────────────────────────────────────────────────
CHUNK_SIZE = 1200          # tokens
CHUNK_OVERLAP = 100        # tokens
TOKEN_ENCODING = "cl100k_base"

# ── Entity Extraction ───────────────────────────────────────────────────
MAX_EXTRACTION_TOKENS = 4000
EXTRACTION_GLEANINGS = 1   # extra pass for missed entities

ENTITY_TYPES = [
    "LAW",
    "SECTION",
    "OFFENSE",
    "PENALTY",
    "LEGAL_CONCEPT",
    "ORGANIZATION",
    "PERSON_TYPE",
    "COURT",
    "LEGAL_PROCEDURE",
]

RELATIONSHIP_TYPES = [
    "CONTAINS",
    "DEFINES",
    "PRESCRIBES_PENALTY",
    "REFERENCES",
    "RELATED_TO",
    "AGGRAVATED_FORM_OF",
    "EXCEPTION_TO",
    "APPLIES_TO",
]

# ── Community Detection ─────────────────────────────────────────────────
MAX_COMMUNITY_SIZE = 10
COMMUNITY_REPORT_MAX_TOKENS = 2000
COMMUNITY_REPORT_INPUT_LIMIT = 8000

# ── Search ──────────────────────────────────────────────────────────────
LOCAL_SEARCH_TOP_K = 25
GLOBAL_SEARCH_TOP_COMMUNITIES = 5
SEARCH_EMBEDDING_TOP_K = 20

# ── LLM Generation ─────────────────────────────────────────────────────
MAX_GENERATION_TOKENS = 2000
GENERATION_TEMPERATURE = 0.0
MAX_RETRIES = 3

# ── Rate Limits (Gemini Free Tier) ──────────────────────────────────
RATE_LIMIT_RPM = 5             # requests per minute
RATE_LIMIT_TPM = 250_000       # input tokens per minute
RATE_LIMIT_RPD = 20            # requests per day
MAX_CONCURRENT_REQUESTS = 1    # sequential to stay within RPM
REQUEST_STAGGER_SECONDS = 12.0 # 60s / 5 RPM = 12s between requests
RATE_LIMIT_REQUESTS_PER_MIN = 10_000
