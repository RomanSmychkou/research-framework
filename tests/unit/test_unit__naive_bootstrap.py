from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from tests.stubs.naive_bootstrap import (
    bootstrap_metrics_payload,
    bootstrap_metric_samples,
    bootstrap_pair_row,
    chunk_corr_rows,
    parse_args,
    summarize_metric,
)


def _sample_pair(chunks: list[float]) -> dict:
    return {
        "feature": "feat_a",
        "target": "pnl_after_3m",
        "avg_corr": float(np.mean(chunks)),
        "std_corr": float(np.std(chunks, ddof=0)),
        "mean_abs_corr": float(np.mean(np.abs(chunks))),
        "icir": 1.0,
        "stability": 1.0,
        "weighted_score": 2.0,
        "diff": 0.1,
        "valid_chunks": len(chunks),
        "total_chunks": len(chunks),
        "same_sign_chunks": len(chunks),
        "min_pair_rows": 10,
        "max_pair_rows": 10,
        "total_pair_rows": 10 * len(chunks),
        "chunks": [[0, corr, 10] for corr in chunks],
    }


class TestNaiveBootstrap(unittest.TestCase):
    def test_bootstrap_recomputes_aggregate_metrics_from_chunks(self) -> None:
        pair = _sample_pair([0.2, 0.4, 0.1, -0.05, 0.3])
        rng = np.random.default_rng(0)
        samples = bootstrap_metric_samples(
            chunk_corr_rows(pair),
            n_samples=20,
            rng=rng,
        )

        self.assertEqual(len(samples), 20)
        self.assertIn("avg_corr", samples[0])
        self.assertIn("weighted_score", samples[0])

    def test_pair_with_single_chunk_is_skipped(self) -> None:
        pair = _sample_pair([0.5])
        rng = np.random.default_rng(0)
        out = bootstrap_pair_row(pair, n_samples=10, rng=rng)
        self.assertIsNone(out)

    def test_bootstrap_payload_sorts_by_weighted_score_mean(self) -> None:
        payload = {
            "symbol": "ARBUSDT",
            "stage": "metrics",
            "pairs": [
                _sample_pair([0.1, 0.2, 0.15, 0.05]),
                _sample_pair([0.8, 0.7, 0.75, 0.65]),
            ],
        }
        payload["pairs"][0]["feature"] = "weak"
        payload["pairs"][1]["feature"] = "strong"

        out = bootstrap_metrics_payload(payload, n_samples=15, seed=1)

        self.assertEqual(out["pairs"][0]["feature"], "strong")
        self.assertEqual(out["stage"], "bootstrap")
        self.assertEqual(out["bootstrap_samples"], 15)

    def test_default_bootstrap_samples_is_30(self) -> None:
        with patch("sys.argv", ["06_naive_bootstrap.py"]):
            args = parse_args()
        self.assertEqual(args.bootstrap_samples, 30)

    def test_summarize_metric_reports_zero_tail_probability(self) -> None:
        summary = summarize_metric([0.2, 0.3, -0.1, 0.4])
        self.assertAlmostEqual(summary["p_value_le_zero"], 0.25)


if __name__ == "__main__":
    unittest.main()
