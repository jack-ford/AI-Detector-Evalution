from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
HUMAN_DIR = DATA / "human"
AI_DIR = DATA / "ai"
SCORES_DIR = DATA / "scores"
METADATA = DATA / "metadata.csv"
ASSIGNMENTS = AI_DIR / "assignments.csv"

LEGACY_ESSAYS = DATA / "legacy" / "essays"
LEGACY_SAPLING_RAW = DATA / "legacy" / "sapling_raw"
LEGACY_CHUNKS = DATA / "legacy" / "chunks.csv"

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

# Downloads that are too big or too boring to commit (Gutenberg catalog, PERSUADE csv).
CACHE = ROOT / ".cache"
