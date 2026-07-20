from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

def json_safe(value: Any) -> Any:
    """Convert common scientific/runtime values into JSON-safe structures.

    This is intentionally dependency-tolerant: numpy/pandas are handled when
    installed but this helper remains importable in lightweight plugin tests.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value

    if isinstance(value, Path):
        return str(value)

    try:
        import numpy as np  # type: ignore
        if isinstance(value, np.generic):
            return json_safe(value.item())
        if isinstance(value, np.ndarray):
            return [json_safe(item) for item in value.tolist()]
    except Exception:
        pass

    try:
        import pandas as pd  # type: ignore
        if isinstance(value, pd.Series):
            return [json_safe(item) for item in value.tolist()]
        if isinstance(value, pd.Index):
            return [json_safe(item) for item in value.tolist()]
        if isinstance(value, pd.DataFrame):
            return [json_safe(row) for row in value.to_dict(orient="records")]
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    except Exception:
        pass

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]

    if hasattr(value, "item") and callable(value.item):
        try:
            return json_safe(value.item())
        except Exception:
            pass

    return str(value)

def ensure_json_object(payload: Mapping[str, Any], *, schema_version: int | None = None) -> dict[str, Any]:
    safe = json_safe(dict(payload))
    if not isinstance(safe, dict):
        raise TypeError("Expected payload to coerce to a JSON object.")
    if schema_version is not None:
        safe.setdefault("schema_version", schema_version)
    return safe
