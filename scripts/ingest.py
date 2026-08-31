"""
Ingest a Moodle course's content into the local vector store.

Usage:
    python scripts/ingest.py --course-id 123

Run this whenever course content changes (nightly cron job is a
reasonable default for v1).
"""
import argparse
import hashlib
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config
from app.gemini_client import embed_batch
from app.vector_store import Chunk, VectorStore
from scripts.moodle_client import MoodleClient


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple sliding-window chunker on characters, breaking on paragraph
    boundaries where possible so chunks don't cut sentences awkwardly."""
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            # If a single paragraph is itself too long, hard-split it
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i : i + chunk_size])
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


def make_chunk_id(course_id: int, module_name: str, index: int) -> str:
    raw = f"{course_id}:{module_name}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def ingest_course(course_id: int):
    Config.validate()

    if not Config.MOODLE_URL or not Config.MOODLE_TOKEN:
        raise RuntimeError("MOODLE_URL and MOODLE_TOKEN must be set")

    client = MoodleClient(Config.MOODLE_URL, Config.MOODLE_TOKEN)
    store = VectorStore(Config.VECTOR_DB_PATH)

    print(f"Fetching content for course {course_id}...")
    raw_items = client.extract_text_content(course_id)
    print(f"Found {len(raw_items)} content items.")

    all_chunks: List[Chunk] = []
    all_texts: List[str] = []

    for item in raw_items:
        pieces = chunk_text(
            item["text"], Config.CHUNK_SIZE_CHARS, Config.CHUNK_OVERLAP_CHARS
        )
        for i, piece in enumerate(pieces):
            chunk_id = make_chunk_id(course_id, item["module_name"], len(all_chunks))
            all_chunks.append(
                Chunk(
                    id=chunk_id,
                    course_id=course_id,
                    text=piece,
                    source_type=item["source_type"],
                    source_url=item["source_url"],
                    module_name=item["module_name"],
                )
            )
            all_texts.append(piece)

    print(f"Chunked into {len(all_chunks)} pieces. Embedding...")
    embeddings = embed_batch(all_texts, task_type="RETRIEVAL_DOCUMENT")

    print("Clearing old data for this course and storing new chunks...")
    store.clear_course(course_id)
    store.add_chunks(all_chunks, embeddings)

    print(f"Done. {len(all_chunks)} chunks indexed for course {course_id}.")


def main():
    parser = argparse.ArgumentParser(description="Ingest a Moodle course into the vector store")
    parser.add_argument("--course-id", type=int, required=True)
    args = parser.parse_args()
    ingest_course(args.course_id)


if __name__ == "__main__":
    main()
