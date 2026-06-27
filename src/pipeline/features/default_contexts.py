from types import MappingProxyType

FEATURE_BUCKET_GRANULARITY_KEY = "feature_bucket_granularity_ms"  # temporarily unused

PRICE_MOMENTUM_DEFAULT_CONTEXT = MappingProxyType(
    {
        "time_column": "exchange_ts",
        "symbol_column": "symbol",
        "price_column": "price",
        "table_name": "spot_trades",
        "where_clause": "symbol = 'BTCUSDT'",
        "own_lookback_ms": 300_000,
        # "feature_bucket_granularity_ms": 60_000,
    }
)

VOLUME_BALANCE_DEFAULT_CONTEXT = MappingProxyType(
    {
        "time_column": "exchange_ts",
        "symbol_column": "symbol",
        "volume_column": "quantity",
        "side_column": "side",
        "table_name": "spot_trades",
        "where_clause": "symbol = 'ETHUSDT'",
        "own_lookback_ms": 300_000,
        # "feature_bucket_granularity_ms": 300_000,
    }
)
