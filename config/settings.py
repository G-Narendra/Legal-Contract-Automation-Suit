"""
Application-level settings for the Legal Contract Automation Suite.
"""

from pathlib import Path

# =============================================================================
# SYSTEM SETTINGS
# =============================================================================
APP_NAME = "Legal Contract Automation Suite"
APP_VERSION = "1.0.0"
DEBUG = False

# =============================================================================
# DATA PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_PATH = DATA_DIR / "chroma_db"
AUDIT_DB_PATH = DATA_DIR / "audit.db"
CONTRACTS_DIR = DATA_DIR / "contracts"
TEMPLATES_DIR = DATA_DIR / "templates"

# =============================================================================
# LLM SETTINGS (overridable via environment variables)
# =============================================================================
LLM_MODEL = "gemini-2.5-flash"
LLM_LITE_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "models/gemini-embedding-2"
PROVIDER = "google"  # google, openai, anthropic, local

# =============================================================================
# RETRIEVAL SETTINGS
# =============================================================================
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_RETRIEVAL = 5
BM25_K1 = 1.5
BM25_B = 0.75

# =============================================================================
# COLLECTION NAMES
# =============================================================================
COLLECTIONS = {
    "legal_laws": "uae_federal_laws",
    "case_law": "uae_case_law",
    "contract_templates": "contract_templates",
    "legal_terminology": "legal_terminology",
    "compliance_rules": "compliance_rules",
}

# =============================================================================
# CACHE SETTINGS (cost optimization)
# =============================================================================
CACHE_TTL_SECONDS = 300  # 5 minutes default
EMBEDDING_CACHE_TTL = 3600  # 1 hour for embeddings
MAX_CACHE_SIZE = 1000  # Max cached responses

# =============================================================================
# CONTRACT PROCESSING
# =============================================================================
SUPPORTED_FORMATS = ["pdf", "docx", "txt"]
MAX_FILE_SIZE_MB = 25
MAX_CONTRACT_LENGTH_CHARS = 100000

# =============================================================================
# COMPLIANCE
# =============================================================================
AUDIT_ALL_ACTIONS = True
LAWYER_REVIEW_REQUIRED_FOR_HIGH_RISK = True
AUDIT_RETENTION_DAYS = 365  # Keep audit logs for 1 year

# =============================================================================
# HUMAN-IN-LOOP
# =============================================================================
AUTO_APPROVE_LOW_RISK = True
REQUIRE_LAWYER_NOTES = True
