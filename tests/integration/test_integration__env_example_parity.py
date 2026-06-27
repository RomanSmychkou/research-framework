"""Parity between .env.example assignments and quant_env (see scripts/verify_env_and_smoke.py)."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestEnvExampleParity(unittest.TestCase):
    def test_example_matches_quant_env_subprocess(self) -> None:
        from scripts.verify_env_and_smoke import verify_env_example

        verify_env_example()


if __name__ == "__main__":
    unittest.main()
