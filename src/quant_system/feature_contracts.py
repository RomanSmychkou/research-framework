"""Persisted feature contract snapshots and ClickHouse row helpers.

Runtime compute contracts live in ``pipeline.features.feature_contract``;
``FeatureContractSnapshot`` is the serialization/registry representation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from clickhouse_driver import Client


FeatureType = Literal["time_based", "event_based", "time_point", "target"]
ComputeBackend = Literal["clickhouse_batch"]
FillPolicy = Literal["zero", "causal_ffill", "null", "derived"]


@dataclass(frozen=True)
class FeatureParentContract:
    feature_name: str
    feature_version: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"feature_name": self.feature_name}
        if self.feature_version is not None:
            out["feature_version"] = self.feature_version
        return out


@dataclass(frozen=True)
class BatchBackfillContract:
    source_table: str
    sql_template: str
    compute_backend: ComputeBackend = "clickhouse_batch"
    fill_policy: FillPolicy = "null"
    fill_value: Any | None = None
    chunk_span: str | None = None
    resource_settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_backfill_contract(self)

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "compute_backend": self.compute_backend,
            "source_table": self.source_table,
            "sql_template": self.sql_template,
            "fill_policy": self.fill_policy,
            "resource_settings": self.resource_settings,
        }
        if self.fill_value is not None:
            out["fill_value"] = self.fill_value
        if self.chunk_span is not None:
            out["chunk_span"] = self.chunk_span
        return out

    def to_json(self) -> str:
        return canonical_json(self.to_json_dict())


@dataclass(frozen=True)
class FeatureContractSnapshot:
    feature_name: str
    feature_version: int
    defined_start_ts: datetime
    defined_end_ts: datetime
    feature_type: FeatureType
    is_target: bool
    is_causal: bool
    own_lookback_ms: int = 0
    own_lookforward_ms: int = 0
    parent_contracts: tuple[FeatureParentContract, ...] = field(default_factory=tuple)
    backfill: BatchBackfillContract | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_feature_contract(self)

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "feature_name": self.feature_name,
            "feature_version": self.feature_version,
            "defined_start_ts": iso_z(self.defined_start_ts),
            "defined_end_ts": iso_z(self.defined_end_ts),
            "feature_type": self.feature_type,
            "is_target": self.is_target,
            "is_causal": self.is_causal,
            "own_lookback_ms": self.own_lookback_ms,
            "own_lookforward_ms": self.own_lookforward_ms,
            "parent_contracts": [dep.to_json_dict() for dep in self.parent_contracts],
        }
        if self.backfill is not None:
            payload["backfill"] = self.backfill.to_json_dict()
        if self.extra:
            payload["extra"] = self.extra
        return payload

    def full_contract_json(self) -> str:
        return canonical_json(self.to_json_dict())

    def parent_contracts_json(self) -> str:
        return canonical_json([dep.to_json_dict() for dep in self.parent_contracts])

    def backfill_json(self) -> str:
        if self.backfill is None:
            return "{}"
        return self.backfill.to_json()

    def backfill_value_json(self) -> str:
        if self.backfill is None or self.backfill.fill_value is None:
            return ""
        return canonical_json(self.backfill.fill_value)

    def contract_hash(self) -> str:
        return hashlib.sha256(self.full_contract_json().encode("utf-8")).hexdigest()

def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def utc_naive(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def iso_z(dt: datetime) -> str:
    return utc_naive(dt).replace(tzinfo=timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def validate_backfill_contract(contract: BatchBackfillContract) -> None:
    if not contract.source_table:
        raise ValueError("source_table is required.")
    if not contract.sql_template.strip():
        raise ValueError("sql_template is required for batch backfill contracts.")
    if contract.compute_backend != "clickhouse_batch":
        raise ValueError(f"Unsupported compute_backend: {contract.compute_backend!r}.")
    if contract.fill_policy not in ("zero", "causal_ffill", "null", "derived"):
        raise ValueError(f"Unsupported fill_policy: {contract.fill_policy!r}.")


def validate_feature_contract(contract: FeatureContractSnapshot) -> None:
    if not contract.feature_name:
        raise ValueError("feature_name is required.")
    if contract.feature_version < 1:
        raise ValueError("feature_version must be >= 1.")
    if utc_naive(contract.defined_start_ts) > utc_naive(contract.defined_end_ts):
        raise ValueError("defined_start_ts must be <= defined_end_ts.")
    if contract.feature_type not in ("time_based", "event_based", "time_point", "target"):
        raise ValueError(f"Unsupported feature_type: {contract.feature_type!r}.")
    if contract.is_target and contract.feature_type != "target":
        raise ValueError("Target contracts must use feature_type='target'.")
    if contract.feature_type == "target" and not contract.is_target:
        raise ValueError("feature_type='target' requires is_target=True.")
    if not isinstance(contract.own_lookback_ms, int):
        raise TypeError("own_lookback_ms must be int.")
    if not isinstance(contract.own_lookforward_ms, int):
        raise TypeError("own_lookforward_ms must be int.")
    if contract.own_lookback_ms < 0:
        raise ValueError("own_lookback_ms must be >= 0.")
    if contract.own_lookforward_ms < 0:
        raise ValueError("own_lookforward_ms must be >= 0.")
    if not contract.is_target and contract.own_lookforward_ms != 0:
        raise ValueError("Non-target features must not have forward_horizon.")
    if contract.is_target and contract.own_lookback_ms != 0:
        raise ValueError("Target contracts should not have lookback_horizon.")


