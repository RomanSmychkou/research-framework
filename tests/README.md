# Test Layout and Naming

## Folders

- `tests/unit/` — fast, isolated tests with local logic only.
- `tests/integration/` — cross-module and stage-coupled checks.
- `tests/e2e/` — end-to-end scenarios (ClickHouse/process level).

## Naming convention

- Global convention: `test_<layer>__<subject>.py`
- Unit tests (`tests/unit`): `test_unit__<subject>.py`
- Integration tests (`tests/integration`): `test_integration__<subject>.py`
- E2E tests (`tests/e2e`): `test_e2e__<subject>.py`

## Notes

- Shared helpers stay at `tests/stage_import.py` and `tests/stubs/`.
- `pyproject.toml` test discovery is aligned to the three folders above.

