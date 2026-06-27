-- Quant pipeline schema (crypto_db). Runs on first ClickHouse start (empty volume).
-- Source of truth: ../init.sql — keep both files in sync.

CREATE DATABASE IF NOT EXISTS crypto_db;

USE crypto_db;

-- #TODO ПЕРЕСМОТРЕТЬ аналоги TABLES ENGINES
-- #TODO ПЕРЕСМОТРЕТЬ ORDER BY и SETTINGS

-- #TODO сделать is_taker/is_maker binary flags и отдельные тесты для них 
-- #TODO перевести всё что можно в отделньые бинарные флаги
-- #TODO сделать в фичи шаблонные бид/аск левелс фичи
-- #TODO ПРОВЕРИТЬ ЛОГИЧНОСТЬ DEFAULTS

-- =============================================================================
-- Orderbook (ZSTD array compression)
-- =============================================================================
CREATE TABLE IF NOT EXISTS crypto_db.spot_orderbook (
    exchange_ts DateTime64(3, 'UTC') CODEC(Delta, ZSTD(3)),
    topic String CODEC(ZSTD(3)),
    update_type Enum8('snapshot'=1, 'delta'=2) CODEC(ZSTD(1)),
    update_id UInt64 CODEC(T64, ZSTD(1)),
    seq UInt64 CODEC(T64, ZSTD(1)),
    -- Nested колонки — каждая как отдельный массив
    bids_price Array(Decimal64(8)) CODEC(ZSTD(3)),
    bids_qty Array(Decimal64(18)) CODEC(ZSTD(3)),
    asks_price Array(Decimal64(8)) CODEC(ZSTD(3)),
    asks_qty Array(Decimal64(18)) CODEC(ZSTD(3))
) ENGINE = MergeTree()
ORDER BY (topic, exchange_ts)
SETTINGS index_granularity = 8192;

-- =============================================================================
-- Spot trades (stage 00 upload + feature pipeline)
-- =============================================================================
CREATE TABLE IF NOT EXISTS crypto_db.spot_trades (
    exchange_ts DateTime64(3, 'UTC') CODEC(Delta, ZSTD(3)),
    symbol String CODEC(ZSTD(5)),
    trade_id String CODEC(ZSTD(3)),
    price Float64 CODEC(Gorilla, ZSTD(3)),
    quantity Float64 CODEC(Gorilla, ZSTD(3)),
    side String CODEC(ZSTD(3))
) ENGINE = MergeTree()
ORDER BY (symbol, exchange_ts)
PARTITION BY toYYYYMM(exchange_ts)
SETTINGS index_granularity = 8192;

-- =============================================================================
-- Features key-value
-- =============================================================================
CREATE TABLE IF NOT EXISTS crypto_db.features_kv (
    symbol String CODEC(ZSTD(5)),
    exchange_ts DateTime64(3, 'UTC') CODEC(Delta, ZSTD(3)),
    feature_name String CODEC(ZSTD(5)),
    feature_value Float64 CODEC(Gorilla, ZSTD(3))
) ENGINE = MergeTree()
ORDER BY (symbol, feature_name, exchange_ts)
SETTINGS index_granularity = 8192;

-- =============================================================================
-- Chunk marking results
-- =============================================================================
CREATE TABLE IF NOT EXISTS crypto_db.chunks_marking_results (
    symbol String CODEC(ZSTD(5)),
    created_at DateTime64(3, 'UTC') CODEC(Delta, ZSTD(3)),
    chunks_list Array(FixedString(64)) CODEC(ZSTD(3)),
    git_hash String CODEC(ZSTD(3)),
    chunks_lish_hash FixedString(64) CODEC(ZSTD(3))
) ENGINE = MergeTree()
ORDER BY (symbol, created_at)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS crypto_db.chunks_instances (
    symbol String CODEC(ZSTD(5)),
    source_table String CODEC(ZSTD(3)),
    chunk_start DateTime64(3, 'UTC') CODEC(Delta, ZSTD(3)),
    chunk_end DateTime64(3, 'UTC') CODEC(Delta, ZSTD(3)),
    chunk_type Enum8('train' = 1, 'validation' = 2, 'test' = 3) CODEC(ZSTD(1)),
    chunk_signature_hash FixedString(64) CODEC(ZSTD(3)),
    create_time DateTime64(3, 'UTC') CODEC(Delta, ZSTD(3)),
    git_hash String CODEC(ZSTD(3))
) ENGINE = MergeTree()
ORDER BY (symbol, chunk_signature_hash)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS crypto_db.manual_marked_chunks (
    symbol String CODEC(ZSTD(5)),
    source_table String CODEC(ZSTD(3)),
    chunk_description String CODEC(ZSTD(3)),
    chunk_effective_start DateTime64(3, 'UTC') CODEC(Delta, ZSTD(3)),
    chunk_effective_end DateTime64(3, 'UTC') CODEC(Delta, ZSTD(3)),
    created_at DateTime64(3, 'UTC') CODEC(Delta, ZSTD(3)),
    chunk_hash FixedString(64) CODEC(ZSTD(3))
) ENGINE = MergeTree()
ORDER BY (symbol, created_at)
SETTINGS index_granularity = 8192;
