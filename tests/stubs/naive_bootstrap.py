"""Minimal bootstrap helpers for tests that expect stage 06 functionality."""

from __future__ import annotations

import argparse
from typing import Any, Iterable, Sequence

import numpy as np


def chunk_corr_rows(pair: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn legacy chunk tuples into dicts for downstream helpers."""
    return [
        {
            "global_chunk_index": int(chunk[0]) if len(chunk) > 0 else idx,
            "corr": float(chunk[1]) if len(chunk) > 1 else 0.0,
            "rows": int(chunk[2]) if len(chunk) > 2 else 0,
        }
        for idx, chunk in enumerate(pair.get("chunks", []))
    ]


def summarize_metric(values: Sequence[float]) -> dict[str, float]:
    """Some aggregate fields that the tests inspect."""
    if not values:
        raise ValueError("chunks must contain at least one correlation")
    arr = np.asarray(values, dtype=float)
    return {
        "avg_corr": float(np.mean(arr)),
        "weighted_score": float(np.mean(np.abs(arr))),
        "p_value_le_zero": float(np.mean(arr <= 0.0)),
    }


def bootstrap_metric_samples(
    chunks: Iterable[dict[str, Any]],
    *,
    n_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    rows = list(chunks)
    if not rows or n_samples <= 0:
        return []
    corrs = np.asarray([row["corr"] for row in rows], dtype=float)
    samples: list[dict[str, float]] = []
    for _ in range(n_samples):
        sampled = rng.choice(corrs, size=len(corrs), replace=True).tolist()
        samples.append(summarize_metric(sampled))
    return samples


def bootstrap_pair_row(pair: dict[str, Any], *, n_samples: int, rng: np.random.Generator) -> dict[str, Any] | None:
    rows = chunk_corr_rows(pair)
    if len(rows) <= 1:
        return None
    return {
        "pair": pair,
        "bootstrap_samples": bootstrap_metric_samples(rows, n_samples=n_samples, rng=rng),
    }


def bootstrap_metrics_payload(payload: dict[str, Any], *, n_samples: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    pairs = list(payload.get("pairs", []))
    for pair in pairs:
        rows = chunk_corr_rows(pair)
        if rows:
            summary = summarize_metric([row["corr"] for row in rows])
            pair["avg_corr"] = summary["avg_corr"]
            pair["weighted_score"] = summary["weighted_score"]
    pairs.sort(key=lambda pair: float(pair.get("avg_corr", 0.0)), reverse=True)
    return {
        "symbol": payload.get("symbol"),
        "stage": "bootstrap",
        "bootstrap_samples": n_samples,
        "pairs": pairs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight bootstrap helpers used just for tests."
    )
    parser.add_argument("--bootstrap-samples", type=int, default=30)
    return parser.parse_args(argv)
