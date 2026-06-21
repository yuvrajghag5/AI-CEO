from pathlib import Path

# Root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Main folders
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "cleaned"
VECTORDB = DATA_DIR / "vector_DB"
EVIDENCE = DATA_DIR / "evidence"



# Files

RAW_DOCUMENTS_FILE = RAW_DIR / "newsapi_data.json"