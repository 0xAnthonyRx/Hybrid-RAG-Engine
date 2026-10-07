-- 1. Enable the vector extension inside Postgres
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop previous table definition to apply dimensional change cleanly
DROP TABLE IF EXISTS documents CASCADE;

-- 2. Create the hybrid document storage table
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Dense Vector Column: Gemini text-embedding-004 uses 768 dimensions
    embedding vector(768),
    
    -- Sparse Keyword Column: Automatically computes English text tokens on insert/update
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Create GIN index for high-speed keyword search
CREATE INDEX idx_documents_tsv 
ON documents USING gin(tsv);

-- 4. Create HNSW index for high-speed semantic search using Cosine Distance
CREATE INDEX idx_documents_embedding 
ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
