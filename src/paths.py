from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
HUMAN_DIR = DATA / "human"
AI_DIR = DATA / "ai"
SCORES_DIR = DATA / "scores"
METADATA = DATA / "metadata.csv"
ASSIGNMENTS = AI_DIR / "assignments.csv"

REWRITE_ESSAYS = DATA / "rewrites" / "essays"
REWRITE_CHUNKS = DATA / "rewrites" / "chunks.csv"

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
MANIFEST = RESULTS / "manifest.json"
FIGURES = RESULTS / "figures"

CACHE = ROOT / ".cache"
