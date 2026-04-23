-- =============================================================================
-- SmartVision-X — PostgreSQL Schema Bootstrap
-- Run automatically on first container start via docker-entrypoint-initdb.d/
-- =============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Identity registry ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS identities (
    id           SERIAL       PRIMARY KEY,
    name         VARCHAR(255) NOT NULL,
    embedding    vector(512)  NOT NULL,
    source       VARCHAR(64)  NOT NULL DEFAULT 'manual',   -- manual | stream | cluster
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    metadata     JSONB                 DEFAULT '{}'
);

-- Unique constraint on name (one embedding per named identity)
CREATE UNIQUE INDEX IF NOT EXISTS idx_identities_name
    ON identities (name);

-- HNSW index for approximate nearest-neighbour cosine similarity search
-- m=16             : graph connectivity — higher = better recall, more memory
-- ef_construction=128 : build quality — higher = better index, slower build
CREATE INDEX IF NOT EXISTS idx_identities_embedding_hnsw
    ON identities USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- ── Recognition event log ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS recognition_events (
    id           BIGSERIAL    PRIMARY KEY,
    camera_id    VARCHAR(64),
    track_id     INTEGER,
    identity_id  INTEGER      REFERENCES identities(id) ON DELETE SET NULL,
    person_name  VARCHAR(255) NOT NULL DEFAULT 'unknown',
    confidence   FLOAT        NOT NULL DEFAULT 0.0,
    matched      BOOLEAN      NOT NULL DEFAULT FALSE,
    bbox_x1      INTEGER,
    bbox_y1      INTEGER,
    bbox_x2      INTEGER,
    bbox_y2      INTEGER,
    source       VARCHAR(32)  NOT NULL DEFAULT 'stream',   -- stream | api
    occurred_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_camera_time
    ON recognition_events (camera_id, occurred_at DESC);

CREATE INDEX IF NOT EXISTS idx_events_person
    ON recognition_events (person_name, occurred_at DESC);

-- ── Unknown face buffer (for identity clustering - Stage 6) ───────────────────
CREATE TABLE IF NOT EXISTS candidate_faces (
    id           BIGSERIAL    PRIMARY KEY,
    embedding    vector(512)  NOT NULL,
    camera_id    VARCHAR(64),
    track_id     INTEGER,
    thumbnail    BYTEA,                 -- compressed face crop JPEG
    cluster_id   INTEGER,              -- assigned by HDBSCAN job
    reviewed     BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_candidates_embedding_hnsw
    ON candidate_faces USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_candidates_cluster
    ON candidate_faces (cluster_id, reviewed);

-- ── Camera registry ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cameras (
    camera_id    VARCHAR(64)  PRIMARY KEY,
    name         VARCHAR(255) NOT NULL DEFAULT '',
    rtsp_url     TEXT,
    location     VARCHAR(255),
    active       BOOLEAN      NOT NULL DEFAULT TRUE,
    frame_sample INTEGER      NOT NULL DEFAULT 3,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── System metrics snapshot (hourly rollup) ────────────────────────────────────
CREATE TABLE IF NOT EXISTS metrics_hourly (
    id                BIGSERIAL    PRIMARY KEY,
    hour_bucket       TIMESTAMPTZ  NOT NULL,
    camera_id         VARCHAR(64),
    frames_processed  BIGINT       NOT NULL DEFAULT 0,
    faces_detected    BIGINT       NOT NULL DEFAULT 0,
    matches           BIGINT       NOT NULL DEFAULT 0,
    unknowns          BIGINT       NOT NULL DEFAULT 0,
    avg_latency_ms    FLOAT,
    UNIQUE (hour_bucket, camera_id)
);
