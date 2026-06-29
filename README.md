# Research Framework

Python/ClickHouse framework for market microstructure research and dataset preparation.
Current focus is a reproducible data/feature pipeline with leakage-aware labeling.

## What Is Implemented

- Historical market data upload into ClickHouse.
- Feature and target calculation pipeline with batch processing for constrained RAM.
- Declarative feature layer with dependency validation (`src/pipeline/features`).
- Chunk marking and train/validation/test splits for walk-forward experiments.

## Quick Start

1) Start ClickHouse:

```bash
cd containers/clickhouse_db
docker-compose up -d
```

2) Download spot public trades from Bybit:

https://www.bybit.com/ru-RU/derivative-activity/history-data

3) Put `*.csv.gz` files into:

`coins/_historical_data/spot_pt`

4) Run the full happy path:

```bash
./scripts/run_happy_path.sh BTCUSDT 10000000
```

Notes:
- If RAM is limited, reduce `MAX_ROWS_BATCH`.
- The internal alias `./do_coffee.sh` runs the same sequence.
- `*.csv.gz` source files are intentionally kept locally as a stability fallback due to upstream data availability interruptions.
- If there are not enough local `*.csv.gz` files, download additional months from the Bybit historical archive and place them into `coins/_historical_data/spot_pt`.

## Manual Stage-by-Stage Run

```bash
python 00_upload_history.py --only-symbols BTCUSDT
python 02_add_features.py --symbol BTCUSDT --max-rows-batch 10000000
python 03_add_targets.py --symbol BTCUSDT --max-rows-batch 10000000
python 05_chunks_marker.py --symbol BTCUSDT --start-date 2024-01-01T00:00:00Z --end-date 2026-03-01T00:00:00Z --tables-to-mark spot_trades --debug
```

## Project Status Snapshot

- In progress: feature self-metrics and result storage format.
- Planned: business metrics views (slippage/fees) and stricter anti-autocorrelation chunking modes.

Active development continues in a private fork. This public repository will not receive further updates.
## License

Licensed under `CC BY-NC-ND 4.0`.
Commercial use and distribution of modified versions are prohibited.
See `LICENSE` or https://creativecommons.org/licenses/by-nc-nd/4.0/.
This license applies to the entire repository history unless otherwise noted.
