import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file  

class Config:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    MOODLE_URL = os.environ.get("MOODLE_URL", "")
    MOODLE_TOKEN = os.environ.get("MOODLE_TOKEN", "")

    EMBEDDING_MODEL = "gemini-embedding-001"
    GENERATION_MODEL = "gemini-3.6-flash"

    VECTOR_DB_PATH = os.environ.get(
        "VECTOR_DB_PATH",
        os.path.join(os.path.dirname(__file__), "..", "data", "vectors.db"),
    )

    # Retrieval settings
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.55  # below this, treat as "no good match"

    # Chunking settings
    CHUNK_SIZE_CHARS = 1800  # roughly 400-500 tokens
    CHUNK_OVERLAP_CHARS = 200

    @classmethod
    def validate(cls):
        missing = []
        if not cls.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}"
            )
