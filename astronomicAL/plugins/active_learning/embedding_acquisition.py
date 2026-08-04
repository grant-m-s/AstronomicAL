from __future__ import annotations

import hashlib
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from astronomicAL.plugins.core_ml.artifacts import (
    StoredEmbeddingRef,
    embedding_ref_from_payload,
)
from astronomicAL.plugins.core_ml.embedding_storage import iter_embedding_batches

from . import state as al_state
from .membership_storage import ExclusionPlan, MembershipLookup, iter_membership_rows
from .streaming_io import check_cancelled, prediction_records_by_ids

GLOBAL_EMBEDDING_STRATEGY_TOKENS = (
    "core_set",
    "coreset",
    "k_center",
    "kcenter",
    "farthest",
    "diversity",
    "embedding",
)


@dataclass(frozen=True)
class EmbeddingAcquisitionResult:
    records: list[Dict[str, Any]]
    stats: Dict[str, Any]
    embedding_ref: Dict[str, Any]


def strategy_supports_global_embeddings(
    strategy: Any,
    strategy_id: str,
    params: Mapping[str, Any],
) -> bool:
    if bool(params.get("use_embedding_acquisition")):
        return True
    if bool(getattr(strategy, "requires_embeddings", False)):
        return True
    normalized = str(strategy_id or "").strip().lower().replace("-", "_")
    return any(token in normalized for token in GLOBAL_EMBEDDING_STRATEGY_TOKENS)


def resolve_embedding_ref(
    *,
    context: Any,
    session: Mapping[str, Any],
    predictions_payload: Mapping[str, Any],
    params: Mapping[str, Any],
) -> tuple[Optional[StoredEmbeddingRef], Optional[str]]:
    direct_candidates = [
        params.get("embedding_ref"),
        params.get("embeddings_ref"),
        predictions_payload.get("embedding_ref"),
        predictions_payload.get("embeddings_ref"),
    ]
    for candidate in direct_candidates:
        if not isinstance(candidate, Mapping):
            continue
        try:
            return StoredEmbeddingRef.from_value(candidate), None
        except Exception:
            continue

    payload_ref = embedding_ref_from_payload(predictions_payload)
    if payload_ref is not None:
        return payload_ref, None

    latest = dict(session.get("latest") or {})
    artifact_id = str(
        params.get("embedding_artifact_id")
        or params.get("embeddings_artifact_id")
        or latest.get("embedding_artifact_id")
        or ""
    ).strip()
    if not artifact_id:
        return None, None
    artifacts = getattr(context, "artifacts", None)
    get = getattr(artifacts, "get", None)
    if not callable(get):
        raise RuntimeError("Embedding artifact lookup requires an artifact store.")
    payload = get(artifact_id)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Embedding artifact {artifact_id!r} is not a mapping payload.")
    ref = embedding_ref_from_payload(payload)
    if ref is None and isinstance(payload.get("storage"), Mapping):
        ref = StoredEmbeddingRef.from_value(payload.get("storage"))
    if ref is None:
        raise ValueError(
            f"Embedding artifact {artifact_id!r} does not contain a standard embedding_ref."
        )
    return ref, artifact_id


def acquire_global_embedding_batch(
    *,
    context: Any,
    session: Mapping[str, Any],
    predictions_payload: Mapping[str, Any],
    strategy: Any,
    strategy_id: str,
    k: int,
    seed: int,
    params: Mapping[str, Any],
    exclusion_plan: ExclusionPlan,
    cancel_token: Any = None,
) -> Optional[EmbeddingAcquisitionResult]:
    if not strategy_supports_global_embeddings(strategy, strategy_id, params):
        return None
    ref, artifact_id = resolve_embedding_ref(
        context=context,
        session=session,
        predictions_payload=predictions_payload,
        params=params,
    )
    if ref is None:
        return None

    pool_dataset_id = str(session.get("pool_dataset_id") or session.get("dataset_id") or "")
    if ref.dataset_id and pool_dataset_id and ref.dataset_id != pool_dataset_id:
        raise ValueError(
            "Embedding sidecar dataset mismatch: "
            f"expected {pool_dataset_id!r}, got {ref.dataset_id!r}."
        )

    metric = str(params.get("embedding_metric") or "euclidean").strip().lower()
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("embedding_metric must be euclidean or cosine")
    limit = max(1, int(k))
    batch_size = max(1, int(params.get("embedding_scan_batch_size") or 8192))

    if _duckdb_available() and (ref.parquet_parts or _duckdb_index_available(ref)):
        records, stats = _duckdb_farthest_first(
            ref=ref,
            session=session,
            exclusion_plan=exclusion_plan,
            k=limit,
            seed=int(seed),
            metric=metric,
            params=params,
            cancel_token=cancel_token,
        )
    else:
        records, stats = _streaming_farthest_first(
            ref=ref,
            session=session,
            exclusion_plan=exclusion_plan,
            k=limit,
            seed=int(seed),
            metric=metric,
            batch_size=batch_size,
            params=params,
            cancel_token=cancel_token,
        )

    selected_ids = [str(record["row_id"]) for record in records]
    prediction_metadata = prediction_records_by_ids(
        predictions_payload,
        selected_ids,
        batch_size=max(1, int(params.get("prediction_scan_batch_size") or 8192)),
        cancel_token=cancel_token,
    )
    for record in records:
        metadata = dict(prediction_metadata.get(str(record["row_id"])) or {})
        metadata.update(record)
        record.clear()
        record.update(metadata)
        record["row_id"] = str(record.get("row_id") or metadata.get("record_id") or "")
        record.setdefault("record_id", record["row_id"])
        record.setdefault("_al_score_source", "embedding_sidecar")

    stats.update(
        {
            "strategy_id": str(strategy_id),
            "streaming": True,
            "global_embedding_acquisition": True,
            "embedding_artifact_id": artifact_id,
            "embedding_row_count": int(ref.row_count),
            "embedding_dimensions": int(ref.dimensions),
            "embedding_metric": metric,
            "requested_k": limit,
            "selected_count": len(records),
        }
    )
    return EmbeddingAcquisitionResult(
        records=records,
        stats=stats,
        embedding_ref=ref.to_dict(),
    )


def _streaming_farthest_first(
    *,
    ref: StoredEmbeddingRef,
    session: Mapping[str, Any],
    exclusion_plan: ExclusionPlan,
    k: int,
    seed: int,
    metric: str,
    batch_size: int,
    params: Mapping[str, Any],
    cancel_token: Any,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    # Keep membership state on disk.  The compatibility label dictionary is only
    # used when no durable table exists.
    compatibility_labelled_ids = {
        str(item.get("row_id"))
        for item in al_state.labelled_training_items(session)
        if isinstance(item, Mapping) and item.get("row_id") not in (None, "")
    }
    max_initial = max(0, int(params.get("embedding_initial_centres_max") or 2048))
    lookup = (
        MembershipLookup(exclusion_plan.membership_ref, excluded_only=False)
        if exclusion_plan.membership_ref
        else None
    )
    centres: list[np.ndarray] = []
    if max_initial:
        for row_ids, vectors, _metadata in iter_embedding_batches(
            ref,
            batch_size=batch_size,
            cancel_token=cancel_token,
        ):
            durable_labelled = (
                lookup.matching_many(row_ids, labelled=True)
                if lookup is not None
                else set()
            )
            for row_id, vector in zip(row_ids, vectors):
                if row_id in durable_labelled or row_id in compatibility_labelled_ids:
                    centres.append(np.asarray(vector, dtype=np.float32))
                    if len(centres) >= max_initial:
                        break
            if len(centres) >= max_initial:
                break

    in_memory_excluded = exclusion_plan.in_memory_row_ids
    selected_ids: set[str] = set()
    selected: list[Dict[str, Any]] = []
    eligible_count = 0
    excluded_count = 0
    source_count = 0
    passes = 0
    initial_centre_count = len(centres)
    try:
        for selection_index in range(k):
            check_cancelled(cancel_token)
            best_row_id = ""
            best_vector: Optional[np.ndarray] = None
            best_distance = -math.inf
            deterministic_key: Optional[int] = None
            pass_eligible = 0
            pass_excluded = 0
            pass_source = 0
            passes += 1
            for row_ids, vectors, _metadata in iter_embedding_batches(
                ref,
                batch_size=batch_size,
                cancel_token=cancel_token,
            ):
                check_cancelled(cancel_token)
                durable_excluded = (
                    lookup.matching_many(row_ids, excluded=True)
                    if lookup is not None
                    else set()
                )
                for row_id, vector in zip(row_ids, vectors):
                    row_id = str(row_id)
                    pass_source += 1
                    if (
                        not row_id
                        or row_id in selected_ids
                        or row_id in in_memory_excluded
                        or row_id in durable_excluded
                    ):
                        if row_id not in selected_ids:
                            pass_excluded += 1
                        continue
                    pass_eligible += 1
                    vector32 = np.asarray(vector, dtype=np.float32)
                    if not centres:
                        key = _stable_hash(row_id, seed)
                        if deterministic_key is None or key < deterministic_key or (
                            key == deterministic_key and row_id < best_row_id
                        ):
                            deterministic_key = key
                            best_row_id = row_id
                            best_vector = vector32.copy()
                            best_distance = 0.0
                        continue
                    distance = _minimum_distance(vector32, centres, metric)
                    if distance > best_distance or (
                        distance == best_distance and (not best_row_id or row_id < best_row_id)
                    ):
                        best_row_id = row_id
                        best_vector = vector32.copy()
                        best_distance = float(distance)
            if selection_index == 0:
                source_count = pass_source
                excluded_count = pass_excluded
                eligible_count = pass_eligible
            if not best_row_id or best_vector is None:
                break
            selected_ids.add(best_row_id)
            centres.append(best_vector)
            selected.append(
                {
                    "row_id": best_row_id,
                    "record_id": best_row_id,
                    "score": float(best_distance),
                    "embedding_distance": float(best_distance),
                    "selection_rank": selection_index + 1,
                    "rank": selection_index + 1,
                    "_al_score_source": "embedding_sidecar",
                }
            )
    finally:
        if lookup is not None:
            lookup.close()

    return selected, {
        "streaming_mode": "embedding_multi_pass",
        "embedding_passes": passes,
        "embedding_source_row_count": source_count,
        "eligible_pool_count": eligible_count,
        "initial_centre_count": initial_centre_count,
        "excluded_count": excluded_count,
    }

def _duckdb_farthest_first(
    *,
    ref: StoredEmbeddingRef,
    session: Mapping[str, Any],
    exclusion_plan: ExclusionPlan,
    k: int,
    seed: int,
    metric: str,
    params: Mapping[str, Any],
    cancel_token: Any,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    import duckdb

    using_index = _duckdb_index_available(ref)
    index_ref = ref.index_ref if using_index else None
    embedding_columns = list(
        (index_ref.embedding_columns if index_ref is not None else None)
        or ref.embedding_columns
        or []
    )
    source_record_id_column = str(
        (index_ref.record_id_column if index_ref is not None else "")
        or ref.record_id_column
    )
    if not embedding_columns:
        raise ValueError(
            "DuckDB embedding acquisition requires scalar embedding_columns in the sidecar reference."
        )
    max_initial = max(0, int(params.get("embedding_initial_centres_max") or 2048))
    with tempfile.TemporaryDirectory(prefix="astronomical-al-embedding-") as temp_dir:
        database_path = str(Path(temp_dir) / "acquisition.duckdb")
        spill_path = Path(temp_dir) / "spill"
        spill_path.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(database=database_path)
        try:
            escaped_spill = str(spill_path).replace("'", "''")
            connection.execute(f"PRAGMA temp_directory='{escaped_spill}'")
            source_sql = _embedding_source_sql(connection, ref)
            _install_membership_table(connection, exclusion_plan)
            selected_columns = ", ".join(_quote_identifier(column) for column in embedding_columns)
            row_id_expr = _quote_identifier(source_record_id_column)
            connection.execute(
                "CREATE TABLE candidates AS "
                f"SELECT CAST(e.{row_id_expr} AS VARCHAR) AS row_id, {selected_columns}, "
                "CAST(1e308 AS DOUBLE) AS min_distance, FALSE AS selected "
                f"FROM {source_sql} AS e "
                "WHERE NOT EXISTS (SELECT 1 FROM membership_filter m "
                f"WHERE m.row_id = CAST(e.{row_id_expr} AS VARCHAR) AND m.excluded) "
                "AND NOT EXISTS (SELECT 1 FROM explicit_exclusions x "
                f"WHERE x.row_id = CAST(e.{row_id_expr} AS VARCHAR))"
            )
            try:
                connection.execute("CREATE UNIQUE INDEX candidates_row_id_idx ON candidates(row_id)")
            except Exception:
                pass
            source_count = int(
                connection.execute(f"SELECT COUNT(*) FROM {source_sql}").fetchone()[0] or 0
            )
            eligible_count = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] or 0)
            initial_count = 0
            if max_initial > 0:
                connection.execute(
                    "CREATE TEMP TABLE initial_centres AS "
                    f"SELECT {', '.join('e.' + _quote_identifier(column) for column in embedding_columns)} "
                    f"FROM {source_sql} e JOIN membership_filter m "
                    f"ON m.row_id = CAST(e.{row_id_expr} AS VARCHAR) "
                    "WHERE m.labelled "
                    f"ORDER BY hash(CAST(e.{row_id_expr} AS VARCHAR)) LIMIT {max_initial}"
                )
                initial_count = int(connection.execute("SELECT COUNT(*) FROM initial_centres").fetchone()[0] or 0)
                if initial_count:
                    distance = _duckdb_distance_expression("c", "s", embedding_columns, metric)
                    connection.execute(
                        "CREATE TEMP TABLE initial_distances AS "
                        f"SELECT c.row_id, MIN({distance}) AS distance "
                        "FROM candidates c CROSS JOIN initial_centres s GROUP BY c.row_id"
                    )
                    connection.execute(
                        "UPDATE candidates c SET min_distance = d.distance "
                        "FROM initial_distances d WHERE c.row_id = d.row_id"
                    )

            selected: list[Dict[str, Any]] = []
            for selection_index in range(k):
                check_cancelled(cancel_token)
                if initial_count == 0 and selection_index == 0:
                    row = connection.execute(
                        "SELECT row_id, min_distance FROM candidates WHERE NOT selected "
                        "ORDER BY hash(row_id || ?) ASC, row_id ASC LIMIT 1",
                        [str(seed)],
                    ).fetchone()
                else:
                    row = connection.execute(
                        "SELECT row_id, min_distance FROM candidates WHERE NOT selected "
                        "ORDER BY min_distance DESC, row_id ASC LIMIT 1"
                    ).fetchone()
                if row is None:
                    break
                row_id = str(row[0])
                raw_distance = float(row[1] or 0.0)
                if raw_distance >= 1e307:
                    score = 0.0
                elif metric == "euclidean":
                    score = math.sqrt(max(0.0, raw_distance))
                else:
                    score = raw_distance
                connection.execute("UPDATE candidates SET selected = TRUE WHERE row_id = ?", [row_id])
                selected.append(
                    {
                        "row_id": row_id,
                        "record_id": row_id,
                        "score": score,
                        "embedding_distance": score,
                        "selection_rank": selection_index + 1,
                        "rank": selection_index + 1,
                        "_al_score_source": "embedding_sidecar",
                    }
                )
                connection.execute("DROP TABLE IF EXISTS current_center")
                connection.execute(
                    "CREATE TEMP TABLE current_center AS "
                    f"SELECT {', '.join(_quote_identifier(column) for column in embedding_columns)} "
                    "FROM candidates WHERE row_id = ?",
                    [row_id],
                )
                distance = _duckdb_distance_expression("c", "s", embedding_columns, metric)
                connection.execute(
                    "UPDATE candidates c SET min_distance = LEAST(c.min_distance, "
                    f"{distance}) FROM current_center s WHERE NOT c.selected"
                )

            return selected, {
                "streaming_mode": "duckdb_external_memory_embedding_index",
                "embedding_source_row_count": source_count,
                "eligible_pool_count": eligible_count,
                "initial_centre_count": initial_count,
                "excluded_count": max(0, source_count - eligible_count),
                "duckdb_external_memory": True,
            }
        finally:
            connection.close()


def _embedding_source_sql(connection: Any, ref: StoredEmbeddingRef) -> str:
    index_ref = ref.index_ref
    if index_ref is not None and str(index_ref.format).lower() in {"duckdb", "duckdb_table"}:
        uri = str(index_ref.uri).replace("'", "''")
        table = str(index_ref.table or "embeddings")
        connection.execute(f"ATTACH '{uri}' AS embedding_index (READ_ONLY)")
        return f"embedding_index.{_quote_identifier(table)}"
    paths = list(ref.parquet_parts or [])
    if not paths:
        raise ValueError("DuckDB embedding acquisition requires parquet_parts or a DuckDB index_ref.")
    return _read_parquet_sql(paths)


def _install_membership_table(connection: Any, exclusion_plan: ExclusionPlan) -> None:
    ref = exclusion_plan.membership_ref
    if ref is not None and ref.parquet_parts:
        source_sql = _read_parquet_sql(ref.parquet_parts)
        connection.execute(
            "CREATE TEMP TABLE membership_filter AS "
            "SELECT CAST(row_id AS VARCHAR) AS row_id, "
            "COALESCE(CAST(excluded AS BOOLEAN), FALSE) AS excluded, "
            "COALESCE(CAST(labelled AS BOOLEAN), FALSE) AS labelled, "
            "COALESCE(CAST(training AS BOOLEAN), FALSE) AS training "
            f"FROM {source_sql}"
        )
        try:
            connection.execute(
                "CREATE UNIQUE INDEX membership_filter_row_id_idx "
                "ON membership_filter(row_id)"
            )
        except Exception:
            pass
    else:
        connection.execute(
            "CREATE TEMP TABLE membership_filter("
            "row_id VARCHAR PRIMARY KEY, excluded BOOLEAN, labelled BOOLEAN, training BOOLEAN)"
        )
        if ref is not None:
            rows: list[tuple[str, bool, bool, bool]] = []
            for row in iter_membership_rows(ref):
                rows.append(
                    (
                        str(row.get("row_id") or ""),
                        bool(row.get("excluded")),
                        bool(row.get("labelled")),
                        bool(row.get("training")),
                    )
                )
                if len(rows) >= 8192:
                    connection.executemany(
                        "INSERT OR REPLACE INTO membership_filter VALUES (?, ?, ?, ?)",
                        rows,
                    )
                    rows = []
            if rows:
                connection.executemany(
                    "INSERT OR REPLACE INTO membership_filter VALUES (?, ?, ?, ?)",
                    rows,
                )

    connection.execute("CREATE TEMP TABLE explicit_exclusions(row_id VARCHAR PRIMARY KEY)")
    if exclusion_plan.in_memory_row_ids:
        connection.executemany(
            "INSERT OR IGNORE INTO explicit_exclusions VALUES (?)",
            [(row_id,) for row_id in sorted(exclusion_plan.in_memory_row_ids)],
        )

def _minimum_distance(vector: np.ndarray, centres: Sequence[np.ndarray], metric: str) -> float:
    centre_matrix = np.asarray(centres, dtype=np.float32)
    if metric == "cosine":
        vector_norm = float(np.linalg.norm(vector))
        centre_norms = np.linalg.norm(centre_matrix, axis=1)
        denominator = np.maximum(centre_norms * max(vector_norm, 1e-12), 1e-12)
        similarity = np.sum(centre_matrix * vector[None, :], axis=1) / denominator
        return float(np.min(1.0 - similarity))
    differences = centre_matrix - vector[None, :]
    return float(np.sqrt(np.min(np.sum(differences * differences, axis=1))))


def _duckdb_distance_expression(left: str, right: str, columns: Sequence[str], metric: str) -> str:
    quoted = [_quote_identifier(column) for column in columns]
    dot = " + ".join(f"({left}.{column} * {right}.{column})" for column in quoted) or "0.0"
    if metric == "cosine":
        left_norm = " + ".join(f"({left}.{column} * {left}.{column})" for column in quoted) or "0.0"
        right_norm = " + ".join(f"({right}.{column} * {right}.{column})" for column in quoted) or "0.0"
        return f"(1.0 - ({dot}) / GREATEST(SQRT({left_norm}) * SQRT({right_norm}), 1e-12))"
    squared = " + ".join(
        f"(({left}.{column} - {right}.{column}) * ({left}.{column} - {right}.{column}))"
        for column in quoted
    ) or "0.0"
    return f"({squared})"


def _stable_hash(row_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{row_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _duckdb_index_available(ref: StoredEmbeddingRef) -> bool:
    index_ref = ref.index_ref
    return bool(
        index_ref is not None
        and str(index_ref.format).lower() in {"duckdb", "duckdb_table"}
        and Path(index_ref.uri).is_file()
    )


def _duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401

        return True
    except Exception:
        return False


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _read_parquet_sql(paths: Iterable[str]) -> str:
    quoted = ", ".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"read_parquet([{quoted}])"
