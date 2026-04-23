#!/usr/bin/env python3
"""
SmartVision-X — face_db.json → PostgreSQL Migration
=====================================================
Run once to migrate existing JSON identity store to pgvector.

Usage:
    python scripts/migrate_to_postgres.py [--db-path path/to/face_db.json]

Environment:
    DATABASE_URL  postgresql://admin:admin@localhost:5432/smartvision
"""

import argparse
import asyncio
import json
import os
import sys

import asyncpg
import numpy as np

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://admin:admin@localhost:5432/smartvision",
)


def normalise(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-9 else vec


def vec_to_str(embedding: np.ndarray) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in embedding) + "]"


async def migrate(db_path: str) -> None:
    if not os.path.exists(db_path):
        print(f"[ERROR] face_db.json not found at: {db_path}")
        sys.exit(1)

    with open(db_path) as f:
        raw = f.read().strip()
        if not raw:
            print("[INFO] face_db.json is empty — nothing to migrate")
            return
        face_db = json.loads(raw)

    if not face_db:
        print("[INFO] face_db.json contains no entries — nothing to migrate")
        return

    print(f"[INFO] Found {len(face_db)} identities to migrate")

    conn = await asyncpg.connect(dsn=DB_DSN)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        migrated = 0
        skipped  = 0

        for name, embedding_list in face_db.items():
            try:
                vec = normalise(
                    np.array(embedding_list, dtype=np.float32)
                )
                vec_str = vec_to_str(vec)

                await conn.execute(
                    """
                    INSERT INTO identities (name, embedding, source)
                    VALUES ($1, $2::vector, 'migration')
                    ON CONFLICT (name) DO UPDATE SET
                        embedding  = EXCLUDED.embedding,
                        updated_at = NOW()
                    """,
                    name, vec_str,
                )
                print(f"  ✓ {name}")
                migrated += 1
            except Exception as exc:
                print(f"  ✗ {name}: {exc}")
                skipped += 1

        print(f"\n[OK] Migration complete: {migrated} migrated, {skipped} failed")

        # Verify count
        count = await conn.fetchval("SELECT COUNT(*) FROM identities")
        print(f"[OK] Total identities in database: {count}")

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate face_db.json to PostgreSQL")
    parser.add_argument(
        "--db-path",
        default="services/face-service/face_db.json",
        help="Path to face_db.json",
    )
    args = parser.parse_args()
    asyncio.run(migrate(args.db_path))
