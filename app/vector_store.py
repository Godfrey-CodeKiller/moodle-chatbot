"""
Minimal vector store backed by SQLite, using numpy for cosine similarity.

This is intentionally simple so the project runs with zero external
infrastructure. It is fine for a single course or a handful of courses.
If you outgrow it (many courses, high query volume), swap this module
for a pgvector-backed implementation — keep the same add_chunks/search
function signatures and nothing else in the project needs to change.
"""
import json
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Chunk:
    id: str
    course_id: int
    text: str
    source_type: str
    source_url: str
    module_name: str


@dataclass
class ScoredChunk(Chunk):
    score: float


class VectorStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    course_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    source_type TEXT,
                    source_url TEXT,
                    module_name TEXT,
                    embedding TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_course_id ON chunks(course_id)"
            )

    def clear_course(self, course_id: int):
        """Wipe existing chunks for a course before re-ingesting it."""
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE course_id = ?", (course_id,))

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        assert len(chunks) == len(embeddings), "chunks and embeddings must align"
        with self._connect() as conn:
            for chunk, emb in zip(chunks, embeddings):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks
                    (id, course_id, text, source_type, source_url, module_name, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.id,
                        chunk.course_id,
                        chunk.text,
                        chunk.source_type,
                        chunk.source_url,
                        chunk.module_name,
                        json.dumps(emb),
                    ),
                )

    def search(
        self,
        course_id: int,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[ScoredChunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, course_id, text, source_type, source_url, module_name, embedding
                FROM chunks WHERE course_id = ?
                """,
                (course_id,),
            ).fetchall()

        if not rows:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec) or 1e-8

        scored = []
        for row in rows:
            (id_, cid, text, source_type, source_url, module_name, emb_json) = row
            emb = np.array(json.loads(emb_json), dtype=np.float32)
            emb_norm = np.linalg.norm(emb) or 1e-8
            score = float(np.dot(query_vec, emb) / (query_norm * emb_norm))
            scored.append(
                ScoredChunk(
                    id=id_,
                    course_id=cid,
                    text=text,
                    source_type=source_type,
                    source_url=source_url,
                    module_name=module_name,
                    score=score,
                )
            )

        scored.sort(key=lambda c: c.score, reverse=True)

        if min_score is not None:
            scored = [c for c in scored if c.score >= min_score]

        return scored[:top_k]

    def count_for_course(self, course_id: int) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE course_id = ?", (course_id,)
            ).fetchone()
            return row[0] if row else 0
