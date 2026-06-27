"""
Fast analogue of 04_sync_models.py.

Design goals:
- keep the same source/target tables and feature declaration contract;
- preserve causal online prediction semantics (predict current row from past fit);
- move timestamp/feature alignment from Python loops to ClickHouse;
- keep a numerically stable ridge state via X'X / X'y accumulation.
"""

from __future__ import annotations

import argparse
import math
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import bootstrap_path
import numpy as np
from tqdm.auto import tqdm

bootstrap_path.ensure_src_on_path(Path(__file__))

from pipeline.features import build_feature_bundle  # noqa: E402
from quant_system import quant_env  # noqa: E402
from quant_system.clickhouse_insert import clickhouse_insert_rows  # noqa: E402
from quant_system.clickhouse_io import (  # noqa: E402
    add_connection_args,
    client_from_args,
    delete_features_kv_range,
    parse_cli_datetime,
    require_insert_batch_rows,
)
from quant_system.time_chunk_processor import exchange_ts_for_ch  # noqa: E402


@dataclass(frozen=True)
class ModelConfig:
    name: str
    feature_tag: str
    params: dict[str, Any]


# Comment out any model in this list to disable it.
ACTIVE_MODELS: tuple[ModelConfig, ...] = (
    ModelConfig(
        name="ridge",
        feature_tag="Ridge",
        params={},
    ),
    # ModelConfig(
    #     name="xgboost",
    #     feature_tag="XGBoostRegressor",
    #     params={
    #         "objective": "reg:squarederror",
    #         "eval_metric": "rmse",
    #         "n_estimators": 200,
    #         "learning_rate": 0.05,
    #         "max_depth": 4,
    #         "subsample": 0.9,
    #         "colsample_bytree": 0.9,
    #         "random_state": 42,
    #         "verbosity": 0,
    #     },
    # ),
)


@dataclass(frozen=True)
class TargetSplitRange:
    full_start: Any
    train_end: Any
    full_end: Any
    total_rows: int
    train_rows: int
    test_rows: int


@dataclass
class OnlineRidgeState:
    feature_count: int
    alpha: float
    xtx: np.ndarray
    xty: np.ndarray
    coef_cache: np.ndarray | None
    coef_dirty: bool
    trained_rows: int = 0

    @classmethod
    def create(cls, feature_count: int, alpha: float) -> "OnlineRidgeState":
        if feature_count <= 0:
            raise ValueError("feature_count must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        return cls(
            feature_count=feature_count,
            alpha=float(alpha),
            xtx=np.eye(feature_count, dtype=float) * float(alpha),
            xty=np.zeros(feature_count, dtype=float),
            coef_cache=None,
            coef_dirty=True,
        )

    def predict_before_fit(self, x: np.ndarray) -> float | None:
        if self.trained_rows <= 0:
            return None
        if self.coef_dirty or self.coef_cache is None:
            self.coef_cache = np.linalg.solve(self.xtx, self.xty)
            self.coef_dirty = False
        return float(np.dot(x, self.coef_cache))

    def fit_current(self, x: np.ndarray, y: float) -> None:
        self.xtx += np.outer(x, x)
        self.xty += x * float(y)
        self.trained_rows += 1
        self.coef_dirty = True

    def fit_block(self, x_block: np.ndarray, y_block: np.ndarray) -> None:
        if x_block.size == 0:
            return
        self.xtx += x_block.T @ x_block
        self.xty += x_block.T @ y_block
        self.trained_rows += int(x_block.shape[0])
        self.coef_dirty = True


class BaseModelRunner(ABC):
    trained_rows: int

    def __init__(self) -> None:
        self.trained_rows = 0

    @abstractmethod
    def predict_before_fit(self, x: np.ndarray) -> float | None:
        ...

    @abstractmethod
    def fit_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> None:
        ...

    def fit_current(self, x: np.ndarray, y: float) -> None:
        self.fit_batch(x.reshape(1, -1), np.asarray([float(y)], dtype=float))


class RidgeModelRunner(BaseModelRunner):
    def __init__(self, *, feature_count: int, ridge_alpha: float) -> None:
        super().__init__()
        self._state = OnlineRidgeState.create(
            feature_count=feature_count, alpha=float(ridge_alpha)
        )

    def predict_before_fit(self, x: np.ndarray) -> float | None:
        return self._state.predict_before_fit(x)

    def fit_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> None:
        self._state.fit_block(x_batch, y_batch)
        self.trained_rows = self._state.trained_rows


class XGBoostModelRunner(BaseModelRunner):
    def __init__(self, *, params: dict[str, Any]) -> None:
        super().__init__()
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise SystemExit(
                "Model 'xgboost' requires package 'xgboost'. Install it to enable this model."
            ) from exc
        self._XGBRegressor = XGBRegressor
        self._params = dict(params)
        self._model: Any | None = None
        self._x_history: list[np.ndarray] = []
        self._y_history: list[np.ndarray] = []

    def predict_before_fit(self, x: np.ndarray) -> float | None:
        if self._model is None:
            return None
        pred = self._model.predict(x.reshape(1, -1))
        if pred.shape[0] <= 0:
            return None
        return float(pred[0])

    def fit_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> None:
        if x_batch.size == 0:
            return
        self._x_history.append(np.asarray(x_batch, dtype=float))
        self._y_history.append(np.asarray(y_batch, dtype=float))
        x_train = np.vstack(self._x_history)
        y_train = np.concatenate(self._y_history)
        self._model = self._XGBRegressor(**self._params)
        self._model.fit(x_train, y_train)
        self.trained_rows = int(y_train.shape[0])


def _create_model_runner(
    *,
    model_config: ModelConfig,
    feature_count: int,
    ridge_alpha: float,
) -> BaseModelRunner:
    if model_config.name == "ridge":
        return RidgeModelRunner(feature_count=feature_count, ridge_alpha=ridge_alpha)
    if model_config.name == "xgboost":
        return XGBoostModelRunner(params=model_config.params)
    raise SystemExit(f"Unknown model '{model_config.name}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast causal per-target ridge training from features_kv; "
            "writes prediction features to features_kv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_connection_args(
        parser,
        include_symbol=True,
        include_replace_range=True,
        include_insert_batch=True,
        include_io_batch=False,
    )
    parser.set_defaults(database="crypto_db")
    parser.add_argument("--table", default="features_kv", help="KV source table.")
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength.",
    )
    parser.add_argument(
        "--predict-feature-prefix",
        default="",
        help="Optional prefix before generated signal feature name.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show per-target training progress.",
    )
    parser.add_argument(
        "--target-chunk-rows",
        type=int,
        default=250_000,
        help="Max target rows per matrix query chunk.",
    )
    parser.add_argument(
        "--fit-block-rows",
        type=int,
        default=64,
        help=(
            "Rows per block model update. 1 keeps strict row-online updates; "
            "higher values reduce solve frequency."
        ),
    )
    return parser.parse_args()


def _execute_columnar(
    client: Any,
    sql: str,
    params: dict[str, Any],
    column_names: tuple[str, ...],
    settings: dict[str, Any] | None = None,
) -> dict[str, list[Any]]:
    data = client.execute(sql, params, columnar=True, settings=settings)
    return {
        name: (data[idx] if idx < len(data) else [])
        for idx, name in enumerate(column_names)
    }


def _load_target_rows(
    client: Any,
    *,
    database: str,
    table: str,
    symbol: str,
    target_feature: str,
    start: Any | None,
    end: Any | None,
) -> tuple[list[Any], list[Any]]:
    sql = f"""
SELECT exchange_ts, feature_value
FROM {database}.{table}
PREWHERE symbol = %(symbol)s
  AND feature_name = %(target_feature)s
"""
    params: dict[str, Any] = {
        "symbol": symbol,
        "target_feature": target_feature,
    }
    if start is not None:
        sql += "\n  AND exchange_ts >= %(start)s"
        params["start"] = exchange_ts_for_ch(start)
    if end is not None:
        sql += "\n  AND exchange_ts < %(end)s"
        params["end"] = exchange_ts_for_ch(end)
    sql += "\nORDER BY exchange_ts"
    columns = _execute_columnar(client, sql, params, ("exchange_ts", "feature_value"))
    target_ts = [exchange_ts_for_ch(ts) for ts in columns["exchange_ts"]]
    return target_ts, columns["feature_value"]


def _resolve_target_split_from_rows(
    *,
    target_feature: str,
    target_ts: list[Any],
    start: Any | None,
    end: Any | None,
) -> TargetSplitRange:
    if (start is None) != (end is None):
        raise SystemExit("Pass both --start and --end, or neither for auto 70/30 split.")

    total_rows = len(target_ts)
    if total_rows <= 0:
        raise SystemExit(f"No rows found for target={target_feature!r} in selected range.")
    if total_rows < 2:
        raise SystemExit(
            f"Need at least 2 rows for train/test split; target={target_feature!r} has {total_rows}."
        )
    split_rows = int(total_rows * 0.7)
    if split_rows <= 0:
        split_rows = 1
    if split_rows >= total_rows:
        split_rows = total_rows - 1

    boundary_ts = target_ts[split_rows - 1]
    train_end_idx = split_rows
    while train_end_idx < total_rows and target_ts[train_end_idx] <= boundary_ts:
        train_end_idx += 1
    if train_end_idx >= total_rows:
        raise SystemExit("70/30 split leaves empty test range after boundary.")
    train_end = exchange_ts_for_ch(target_ts[train_end_idx])

    if start is not None and end is not None:
        full_start = exchange_ts_for_ch(start)
        full_end = exchange_ts_for_ch(end)
    else:
        full_start = exchange_ts_for_ch(target_ts[0])
        full_end = exchange_ts_for_ch(target_ts[-1]) + timedelta(microseconds=1)
    train_rows = train_end_idx
    test_rows = total_rows - train_rows
    return TargetSplitRange(
        full_start=full_start,
        train_end=train_end,
        full_end=full_end,
        total_rows=total_rows,
        train_rows=train_rows,
        test_rows=test_rows,
    )


def _build_signal_feature_name(
    *,
    target_name: str,
    model_tag: str,
    feature_names: list[str],
) -> str:
    features_part = ",".join(feature_names)
    return f"signal {target_name} {model_tag} {features_part}"


def _build_target_feature_matrix_sql(
    *,
    database: str,
    table: str,
    feature_names: list[str],
) -> tuple[str, tuple[str, ...], dict[str, str]]:
    raw_aliases = [f"f_{idx}" for idx in range(len(feature_names))]
    feature_param_keys = [f"feature_name_{idx}" for idx in range(len(feature_names))]
    pivot_expr = []
    for idx, feature_param_key in enumerate(feature_param_keys):
        pivot_expr.append(
            "maxIf(toFloat64OrNull(toString(feature_value)), feature_name = "
            f"%({feature_param_key})s) AS {raw_aliases[idx]}"
        )
    pivot_expr_sql = ",\n        ".join(pivot_expr)
    select_cols = ", ".join(f"f.{alias} AS {alias}" for alias in raw_aliases)
    sql = f"""
WITH
t AS
(
    SELECT
        exchange_ts,
        toFloat64OrNull(toString(feature_value)) AS target_value
    FROM {database}.{table}
    PREWHERE symbol = %(symbol)s
      AND feature_name = %(target_feature)s
      AND exchange_ts >= %(start)s
      AND exchange_ts < %(end)s
),
target_ts AS
(
    SELECT DISTINCT exchange_ts
    FROM t
),
f AS
(
    SELECT
        src.exchange_ts AS exchange_ts,
        {pivot_expr_sql}
    FROM {database}.{table} AS src
    INNER JOIN target_ts ON target_ts.exchange_ts = src.exchange_ts
    PREWHERE src.symbol = %(symbol)s
      AND src.feature_name IN %(feature_names)s
      AND src.exchange_ts >= %(start)s
      AND src.exchange_ts < %(end)s
    GROUP BY src.exchange_ts
)
SELECT
    t.exchange_ts AS exchange_ts,
    t.target_value AS target_value,
    {select_cols}
FROM t
LEFT JOIN f
    ON f.exchange_ts = t.exchange_ts
ORDER BY exchange_ts
"""
    feature_name_params = dict(zip(feature_param_keys, feature_names))
    return sql, tuple(["exchange_ts", "target_value", *raw_aliases]), feature_name_params


def _load_present_feature_names(
    client: Any,
    *,
    database: str,
    table: str,
    symbol: str,
    feature_names: tuple[str, ...],
    start: Any,
    end: Any,
) -> set[str]:
    sql = f"""
SELECT feature_name
FROM {database}.{table}
PREWHERE symbol = %(symbol)s
  AND feature_name IN %(feature_names)s
  AND exchange_ts >= %(start)s
  AND exchange_ts < %(end)s
GROUP BY feature_name
"""
    rows = client.execute(
        sql,
        {
            "symbol": symbol,
            "feature_names": feature_names,
            "start": start,
            "end": end,
        },
    )
    return {str(row[0]) for row in rows}


def _prediction_insert_sql(*, database: str, table: str) -> str:
    return f"INSERT INTO {database}.{table} (symbol, exchange_ts, feature_name, feature_value) VALUES"


def _iter_insert_chunks(
    *,
    symbol: str,
    timestamps: list[Any],
    prediction_feature_name: str,
    predictions: np.ndarray,
    chunk_rows: int,
):
    total = len(timestamps)
    for start in range(0, total, chunk_rows):
        end = min(start + chunk_rows, total)
        chunk = [
            (
                symbol,
                timestamps[idx],
                prediction_feature_name,
                float(predictions[idx]),
            )
            for idx in range(start, end)
        ]
        yield chunk


def main() -> None:
    args = parse_args()
    if args.ridge_alpha <= 0:
        raise SystemExit("--ridge-alpha must be positive.")
    if args.target_chunk_rows <= 0:
        raise SystemExit("--target-chunk-rows must be positive.")
    if args.fit_block_rows <= 0:
        raise SystemExit("--fit-block-rows must be positive.")
    require_insert_batch_rows(args.insert_batch_rows)
    insert_batch_rows = quant_env.insert_batch_rows(args.insert_batch_rows)

    vprint = (
        (lambda *a, **kw: print(*a, **kw, flush=True))
        if not args.quiet
        else (lambda *_a, **_kw: None)
    )
    client = client_from_args(args)

    bundle = build_feature_bundle()
    active_features = list(bundle.non_targets)
    active_targets = list(bundle.targets)
    if not active_features:
        raise SystemExit("No active non-target features declared.")
    if not active_targets:
        raise SystemExit("No active targets declared.")

    feature_policy: dict[str, dict[str, Any]] = {
        feature.name: dict(feature.ml_fill_policy)
        for feature in active_features
    }
    base_feature_names = [feature.name for feature in active_features]
    base_feature_names_tuple = tuple(base_feature_names)
    target_names = [target.name for target in active_targets]

    onehot_feature_names: list[str] = []
    onehot_key_by_feature: dict[str, str | None] = {}
    special_fill_value: dict[str, float | None] = {}
    for feature_name in base_feature_names:
        policy = feature_policy[feature_name]
        if bool(policy.get("onehot_encoding", False)):
            onehot_name = f"{feature_name}__invalid_onehot"
            onehot_feature_names.append(onehot_name)
            onehot_key_by_feature[feature_name] = onehot_name
        else:
            onehot_key_by_feature[feature_name] = None

        special_value = policy.get("special_value")
        special_float: float | None = None
        if special_value is not None:
            special_float = float(special_value)
            if not math.isfinite(special_float):
                raise SystemExit(
                    f"Feature={feature_name!r} has non-finite special_value={special_value!r}"
                )
        special_fill_value[feature_name] = special_float

    if not ACTIVE_MODELS:
        raise SystemExit("No active models declared in ACTIVE_MODELS.")
    start = exchange_ts_for_ch(parse_cli_datetime(args.start))
    end = exchange_ts_for_ch(parse_cli_datetime(args.end))
    prediction_insert_sql = _prediction_insert_sql(database=args.database, table=args.table)

    total_inserted = 0
    global_started = perf_counter()
    for target_name in target_names:
        for model_config in ACTIVE_MODELS:
            started = perf_counter()
            all_target_ts, _all_target_values = _load_target_rows(
                client,
                database=args.database,
                table=args.table,
                symbol=args.symbol,
                target_feature=target_name,
                start=start,
                end=end,
            )
            split = _resolve_target_split_from_rows(
                target_feature=target_name,
                target_ts=all_target_ts,
                start=start,
                end=end,
            )
            runtime_invalid_counts: dict[str, int] = defaultdict(int)

            present_feature_names = _load_present_feature_names(
                client,
                database=args.database,
                table=args.table,
                symbol=args.symbol,
                feature_names=base_feature_names_tuple,
                start=split.full_start,
                end=split.full_end,
            )
            target_base_feature_names = [
                feature_name
                for feature_name in base_feature_names
                if feature_name in present_feature_names
            ]
            if not target_base_feature_names:
                raise SystemExit(
                    f"Target={target_name!r}: no active features found in range "
                    f"[{split.full_start}, {split.full_end})."
                )
            skipped_features = [
                feature_name
                for feature_name in base_feature_names
                if feature_name not in present_feature_names
            ]

            target_onehot_feature_names: list[str] = []
            target_onehot_index_by_feature: dict[str, int] = {}
            for feature_idx, feature_name in enumerate(target_base_feature_names):
                onehot_name = onehot_key_by_feature[feature_name]
                if onehot_name is None:
                    target_onehot_index_by_feature[feature_name] = -1
                else:
                    target_onehot_index_by_feature[feature_name] = (
                        len(target_base_feature_names) + len(target_onehot_feature_names)
                    )
                    target_onehot_feature_names.append(onehot_name)

            model_feature_names = [*target_base_feature_names, *target_onehot_feature_names]
            model_feature_count = len(model_feature_names)
            matrix_sql, matrix_cols, matrix_feature_name_params = _build_target_feature_matrix_sql(
                database=args.database,
                table=args.table,
                feature_names=target_base_feature_names,
            )

            prediction_feature_name = (
                f"{args.predict_feature_prefix}"
                f"{_build_signal_feature_name(target_name=target_name, model_tag=model_config.feature_tag, feature_names=model_feature_names)}"
            )
            vprint(
                f"Target={target_name} Model={model_config.name}: split train/test rows={split.train_rows:,}/{split.test_rows:,} "
                f"range=[{split.full_start}, {split.full_end}) train_end={split.train_end}"
            )
            if skipped_features:
                vprint(
                    f"Target={target_name} Model={model_config.name}: inactive features skipped={len(skipped_features)} "
                    f"({', '.join(skipped_features)})"
                )
            if args.replace_range:
                delete_features_kv_range(
                    client=client,
                    database=args.database,
                    symbol=args.symbol,
                    start=split.full_start,
                    end=split.full_end,
                    feature_names=[prediction_feature_name],
                )
                vprint(
                    f"Target={target_name} Model={model_config.name}: deleted old rows "
                    f"for feature={prediction_feature_name}"
                )

            model = _create_model_runner(
                model_config=model_config,
                feature_count=model_feature_count,
                ridge_alpha=float(args.ridge_alpha),
            )
            pending_x: list[np.ndarray] = []
            pending_y: list[float] = []
            target_total_rows = len(all_target_ts)
            processed_rows = 0
            progress_bar = None
            if args.progress and not args.quiet:
                progress_bar = tqdm(
                    total=target_total_rows,
                    desc=f"sync {target_name}:{model_config.name}",
                    unit="rows",
                    file=sys.stdout,
                    ascii=True,
                    mininterval=0.1,
                    dynamic_ncols=True,
                    leave=True,
                )
            while processed_rows < target_total_rows:
                chunk_end_idx = min(processed_rows + int(args.target_chunk_rows), target_total_rows)
                if chunk_end_idx < target_total_rows:
                    boundary_ts = all_target_ts[chunk_end_idx - 1]
                    while chunk_end_idx < target_total_rows and all_target_ts[chunk_end_idx] <= boundary_ts:
                        chunk_end_idx += 1
                chunk_start_ts = all_target_ts[processed_rows]
                chunk_end_ts = (
                    all_target_ts[chunk_end_idx]
                    if chunk_end_idx < target_total_rows
                    else split.full_end
                )
                expected_chunk_rows = chunk_end_idx - processed_rows
                col = _execute_columnar(
                    client,
                    matrix_sql,
                    {
                        "symbol": args.symbol,
                        "target_feature": target_name,
                        "feature_names": tuple(target_base_feature_names),
                        "start": chunk_start_ts,
                        "end": chunk_end_ts,
                        **matrix_feature_name_params,
                    },
                    matrix_cols,
                    settings={
                        "max_bytes_before_external_group_by": 256 * 1024 * 1024,
                        "max_bytes_before_external_sort": 256 * 1024 * 1024,
                    },
                )
                timestamps = [exchange_ts_for_ch(ts) for ts in col["exchange_ts"]]
                if not timestamps:
                    skipped = chunk_end_idx - processed_rows
                    processed_rows = chunk_end_idx
                    if progress_bar is not None:
                        progress_bar.update(skipped)
                    continue

                y = np.asarray(col["target_value"], dtype=float)
                n_rows = len(timestamps)
                if n_rows != expected_chunk_rows:
                    runtime_invalid_counts["chunk_row_mismatch"] += abs(expected_chunk_rows - n_rows)
                if y.shape[0] != n_rows:
                    raise SystemExit(f"Target={target_name!r}: target column length mismatch.")

                x_raw = np.empty((n_rows, len(target_base_feature_names)), dtype=float)
                for idx in range(len(target_base_feature_names)):
                    x_raw[:, idx] = np.asarray(col[f"f_{idx}"], dtype=float)

                x_parts: list[np.ndarray] = []
                for idx, feature_name in enumerate(target_base_feature_names):
                    feat_col = x_raw[:, idx].copy()
                    invalid = ~np.isfinite(feat_col)
                    runtime_invalid_counts[feature_name] += int(np.count_nonzero(invalid))
                    fill = special_fill_value[feature_name]
                    if fill is not None:
                        feat_col[invalid] = float(fill)
                    x_parts.append(feat_col.reshape(-1, 1))
                    onehot_idx = target_onehot_index_by_feature[feature_name]
                    if onehot_idx >= 0:
                        x_parts.append(invalid.astype(float).reshape(-1, 1))
                x = np.hstack(x_parts)
                if x.shape[1] != model_feature_count:
                    raise SystemExit(
                        f"Target={target_name!r}: model feature shape mismatch "
                        f"{x.shape[1]} != {model_feature_count}."
                    )

                predictions = np.full(n_rows, np.nan, dtype=float)
                x_valid = np.isfinite(x).all(axis=1)
                y_valid = np.isfinite(y)
                for row_idx in range(n_rows):
                    if x_valid[row_idx]:
                        pred = model.predict_before_fit(x[row_idx])
                        if pred is not None:
                            predictions[row_idx] = pred
                    if x_valid[row_idx] and y_valid[row_idx]:
                        if args.fit_block_rows == 1:
                            model.fit_current(x[row_idx], float(y[row_idx]))
                        else:
                            pending_x.append(x[row_idx])
                            pending_y.append(float(y[row_idx]))
                            if len(pending_x) >= int(args.fit_block_rows):
                                x_block = np.vstack(pending_x)
                                y_block = np.asarray(pending_y, dtype=float)
                                model.fit_batch(x_block, y_block)
                                pending_x.clear()
                                pending_y.clear()
                    else:
                        if not x_valid[row_idx]:
                            runtime_invalid_counts["model_input_invalid"] += 1
                        if not y_valid[row_idx]:
                            runtime_invalid_counts[target_name] += 1

                for chunk in _iter_insert_chunks(
                    symbol=args.symbol,
                    timestamps=timestamps,
                    prediction_feature_name=prediction_feature_name,
                    predictions=predictions,
                    chunk_rows=insert_batch_rows,
                ):
                    clickhouse_insert_rows(
                        client,
                        prediction_insert_sql,
                        chunk,
                        batch_rows=insert_batch_rows,
                    )
                    total_inserted += len(chunk)

                if progress_bar is not None:
                    progress_bar.update(n_rows)
                processed_rows = chunk_end_idx
            if pending_x:
                x_block = np.vstack(pending_x)
                y_block = np.asarray(pending_y, dtype=float)
                model.fit_batch(x_block, y_block)
                pending_x.clear()
                pending_y.clear()
            if progress_bar is not None:
                progress_bar.close()

            elapsed_s = perf_counter() - started
            rows_per_s = (target_total_rows / elapsed_s) if elapsed_s > 0 else float("inf")
            vprint(
                f"Target={target_name} Model={model_config.name}: inserted predictions={target_total_rows:,} "
                f"into {args.database}.{args.table} feature={prediction_feature_name}"
            )
            vprint(
                f"Target={target_name} Model={model_config.name}: throughput rows/s={rows_per_s:,.2f}, "
                f"elapsed={elapsed_s:,.2f}s, trained_rows={model.trained_rows:,}"
            )
            total_invalid = sum(runtime_invalid_counts.values())
            vprint(
                f"Target={target_name} Model={model_config.name}: invalid values logged={total_invalid:,}"
            )
            if runtime_invalid_counts:
                for feature_name, count in sorted(
                    runtime_invalid_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:10]:
                    vprint(f"  invalid[{feature_name}]={count:,}")

    global_elapsed = perf_counter() - global_started
    vprint(f"Done, inserted_rows={total_inserted:,}, elapsed={global_elapsed:,.2f}s")


if __name__ == "__main__":
    main()

