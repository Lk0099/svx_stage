"""
SmartVision-X — Database Module
================================
Async PostgreSQL + pgvector client for identity storage and ANN search.

All queries use the pgvector HNSW index via the <=> cosine distance operator.
Search latency target: < 10ms for up to 1M identities.

Usage
-----
    from database import db
    await db.init()
    await db.register_identity("Alice", embedding_array)
    results = await db.search_identity(query_embedding, threshold=0.45)
"""

from __future__ import annotations

import os
import json
import logging
from typing import Optional

import asyncpg
import numpy as np

logger = logging.getLogger(__name__)

# ── Connection string ──────────────────────────────────────────────────────────
DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://admin:admin@postgres:5432/smartvision",
)


class Database:
    """Singleton async database client wrapping an asyncpg connection pool."""

    def __init__(self) -> None:
        self._pool: Optional[asyncpg.Pool] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Initialise the connection pool. Call once at application startup."""
        if self._pool is not None:
            return

        logger.info("Connecting to PostgreSQL at %s", DB_DSN.split("@")[-1])
        self._pool = await asyncpg.create_pool(
            dsn=DB_DSN,
            min_size=2,
            max_size=10,
            command_timeout=10,
            server_settings={"application_name": "svx-face-service"},
        )

        # Register the pgvector codec so asyncpg can encode/decode vector columns
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # Register custom type codec for vector columns
            await conn.execute(
                "SELECT typname FROM pg_type WHERE typname = 'vector'"
            )

        logger.info("Database pool ready (min=2, max=10)")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _pool_or_raise(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database not initialised. Call db.init() first.")
        return self._pool

    # ── Vector helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _vec_to_str(embedding: np.ndarray) -> str:
        """Convert float32 ndarray to pgvector literal string '[v1,v2,...]'."""
        return "[" + ",".join(f"{float(v):.8f}" for v in embedding) + "]"

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec

    # ── Identity CRUD ──────────────────────────────────────────────────────────

    async def register_identity(
        self,
        name: str,
        embedding: np.ndarray,
        source: str = "manual",
        metadata: dict | None = None,
    ) -> dict:
        """
        Insert or update a named identity with its 512-d embedding.

        Returns the created/updated row as a dict.
        Raises ValueError if name is empty.
        """
        if not name or not name.strip():
            raise ValueError("Identity name must not be empty")

        embedding = self._normalise(embedding.flatten().astype(np.float32))
        vec_str   = self._vec_to_str(embedding)
        meta_json = json.dumps(metadata or {})

        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO identities (name, embedding, source, metadata)
                VALUES ($1, $2::vector, $3, $4::jsonb)
                ON CONFLICT (name) DO UPDATE SET
                    embedding  = EXCLUDED.embedding,
                    source     = EXCLUDED.source,
                    metadata   = EXCLUDED.metadata,
                    updated_at = NOW()
                RETURNING id, name, source, created_at, updated_at
                """,
                name.strip(), vec_str, source, meta_json,
            )
        return dict(row)

    async def search_identity(
        self,
        embedding: np.ndarray,
        threshold: float = 0.45,
        top_k: int = 1,
    ) -> list[dict]:
        """
        Find the closest registered identity using HNSW cosine ANN search.

        Parameters
        ----------
        embedding  : Query embedding, will be L2-normalised internally.
        threshold  : Minimum cosine similarity to be considered a match.
        top_k      : Number of candidates to return.

        Returns
        -------
        List of dicts: [{name, similarity, id}] sorted by similarity desc.
        Empty list if no match above threshold.

        Latency target: < 5ms at 100k identities, < 10ms at 1M identities.
        """
        embedding = self._normalise(embedding.flatten().astype(np.float32))
        vec_str   = self._vec_to_str(embedding)

        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id,
                    name,
                    1 - (embedding <=> $1::vector) AS similarity
                FROM identities
                WHERE 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                vec_str, float(threshold), int(top_k),
            )

        return [
            {"id": r["id"], "name": r["name"], "similarity": float(r["similarity"])}
            for r in rows
        ]

    async def delete_identity(self, name: str) -> bool:
        """Delete a named identity. Returns True if a row was deleted."""
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM identities WHERE name = $1", name
            )
        return result == "DELETE 1"

    async def list_identities(self) -> list[dict]:
        """Return all registered identities (name, source, created_at)."""
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, name, source, created_at FROM identities ORDER BY name"
            )
        return [dict(r) for r in rows]

    async def count_identities(self) -> int:
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM identities")

    async def identity_exists(self, name: str) -> bool:
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM identities WHERE name = $1", name
            )
        return row is not None

    # ── Event logging ──────────────────────────────────────────────────────────

    async def log_recognition_event(
        self,
        person_name: str,
        confidence: float,
        matched: bool,
        camera_id: str | None = None,
        track_id:  int | None = None,
        bbox:      dict | None = None,
        source:    str = "api",
    ) -> None:
        """
        Write a recognition event to the event log for analytics.
        Non-blocking best-effort — does not raise on failure.
        """
        try:
            pool = self._pool_or_raise()
            bbox = bbox or {}
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO recognition_events
                        (camera_id, track_id, person_name, confidence,
                         matched, bbox_x1, bbox_y1, bbox_x2, bbox_y2, source)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                    """,
                    camera_id, track_id, person_name,
                    float(confidence), matched,
                    bbox.get("x1"), bbox.get("y1"),
                    bbox.get("x2"), bbox.get("y2"),
                    source,
                )
        except Exception as exc:
            logger.warning("Failed to log recognition event: %s", exc)

    # ── Unknown face buffer ────────────────────────────────────────────────────

    async def store_candidate_face(
        self,
        embedding:  np.ndarray,
        camera_id:  str | None = None,
        track_id:   int | None = None,
        thumbnail:  bytes | None = None,
    ) -> None:
        """Store an unmatched face embedding for future clustering."""
        try:
            embedding = self._normalise(embedding.flatten().astype(np.float32))
            vec_str   = self._vec_to_str(embedding)
            pool      = self._pool_or_raise()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO candidate_faces
                        (embedding, camera_id, track_id, thumbnail)
                    VALUES ($1::vector, $2, $3, $4)
                    """,
                    vec_str, camera_id, track_id, thumbnail,
                )
        except Exception as exc:
            logger.warning("Failed to store candidate face: %s", exc)

    # ── Camera registry ────────────────────────────────────────────────────────

    async def register_camera(
        self,
        camera_id:    str,
        name:         str = "",
        rtsp_url:     str | None = None,
        location:     str | None = None,
        frame_sample: int = 3,
    ) -> None:
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO cameras (camera_id, name, rtsp_url, location, frame_sample)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (camera_id) DO UPDATE SET
                    name         = EXCLUDED.name,
                    rtsp_url     = EXCLUDED.rtsp_url,
                    location     = EXCLUDED.location,
                    frame_sample = EXCLUDED.frame_sample
                """,
                camera_id, name, rtsp_url, location, frame_sample,
            )

    async def list_cameras(self) -> list[dict]:
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM cameras WHERE active = TRUE ORDER BY camera_id"
            )
        return [dict(r) for r in rows]


# ── Module-level singleton ─────────────────────────────────────────────────────
db = Database()
