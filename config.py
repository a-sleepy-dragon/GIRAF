import os

API_KEY = os.environ.get("API_KEY", "")
BASE_URL = "<your-llm-api-base-url>"
MODEL_NAME = "<your-model-name>"
TEMPERATURE = 0.0
MAX_TOKENS = 16384
TIMEOUT = 180
REASONING_EFFORT = "low"

NEO4J_URI = "<your-neo4j-uri>"
NEO4J_AUTH = ("<your-neo4j-username>", "<your-neo4j-password>")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
SCHEMA_JSON = "<path-to-schema-json>"
DATASET_DIR = "<path-to-dataset-dir>"
TOOL_DIR = HERE
OUT_DIR = os.path.join(HERE, "out")
SINGLE_PROP_SETS_JSON = "<path-to-single-prop-sets-json>"

STORAGE_BUDGET_MB = 2048.0
MAX_ROUNDS = 12

RAG_KB_DIR = "<path-to-rag-kb-dir>"
RAG_EMBED_MODEL = "BAAI/bge-m3"
RAG_TOP_K = 4
RAG_ENABLED = True
RAG_ASSISTANT_ENABLED = True
