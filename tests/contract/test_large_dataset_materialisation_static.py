from __future__ import annotations

import ast
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ROOT = PROJECT_ROOT / "astronomicAL" / "plugins"


# These are known migration debts in bundled plugins.
#
# Keep this list small and remove entries as plugins are converted to
# DatasetSource / Parquet-friendly access patterns.
KNOWN_TRANSITIONAL_FULL_MATERIALISATION = {
}


def plugin_python_files() -> list[Path]:
    if not PLUGIN_ROOT.exists():
        return []

    return sorted(
        path
        for path in PLUGIN_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def parse_file(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def dotted_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr

    if isinstance(node, ast.Call):
        name = dotted_name(node.func)
        if name:
            return f"{name}()"
        return "<call>"

    if isinstance(node, ast.Subscript):
        return dotted_name(node.value)

    return None


def has_meaningful_keyword(call: ast.Call, names: set[str]) -> bool:
    """
    Return True when a call has one of the named keyword arguments and that
    keyword is not explicitly None.

    Example:
        get_df(limit=100)      -> meaningful
        get_df(limit=None)     -> not meaningful
        get_df(columns=["id"]) -> meaningful
    """

    for keyword in call.keywords:
        if keyword.arg not in names:
            continue

        value = keyword.value

        if isinstance(value, ast.Constant) and value.value is None:
            continue

        return True

    return False


def call_has_any_row_or_column_bound(call: ast.Call) -> bool:
    """
    A pandas materialisation is considered safer if it has at least one obvious
    bound:

        columns=...
        limit=...
        where_sql=...
        max_rows=...

    This is intentionally conservative. It will not prove code is efficient,
    but it catches the worst accidental full-dataset materialisations.
    """

    return has_meaningful_keyword(
        call,
        {
            "columns",
            "limit",
            "where_sql",
            "max_rows",
        },
    )


def is_dataset_get_df_call(call: ast.Call) -> bool:
    name = dotted_name(call.func)
    if not name:
        return False

    # Catch:
    #   context.datasets.get_df(...)
    #   self.context.datasets.get_df(...)
    #   datasets.get_df(...)
    return name.endswith(".get_df") or name == "get_df"


def is_to_pandas_call(call: ast.Call) -> bool:
    name = dotted_name(call.func)
    if not name:
        return False

    return name.endswith(".to_pandas") or name == "to_pandas"


def is_dataset_df_property_access(node: ast.Attribute) -> bool:
    """
    Catch Dataset.df compatibility-property access, but avoid flagging arbitrary
    dict/cache/service .get(...).df patterns.
    """

    if node.attr != "df":
        return False

    value = node.value

    if not isinstance(value, ast.Call):
        return False

    called_name = dotted_name(value.func)
    if not called_name:
        return False

    dataset_get_names = {
        "datasets.get",
        "context.datasets.get",
        "self.context.datasets.get",
        "self.datasets.get",
    }

    if called_name in dataset_get_names:
        return True

    return (
        called_name.endswith(".datasets.get")
        or called_name.endswith(".context.datasets.get")
    )


def find_full_materialisation_risks(path: Path) -> list[str]:
    tree = parse_file(path)
    risks: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if is_dataset_get_df_call(node) and not call_has_any_row_or_column_bound(node):
                risks.append(
                    f"line {node.lineno}: unbounded DatasetManager.get_df(...) call"
                )

            if is_to_pandas_call(node) and not call_has_any_row_or_column_bound(node):
                risks.append(
                    f"line {node.lineno}: unbounded DatasetSource.to_pandas(...) call"
                )

        elif isinstance(node, ast.Attribute):
            if is_dataset_df_property_access(node):
                risks.append(
                    f"line {node.lineno}: Dataset.df compatibility property access"
                )

    return risks


@pytest.mark.plugin_quality
@pytest.mark.contract
@pytest.mark.parametrize(
    "path",
    plugin_python_files(),
    ids=lambda p: str(p.relative_to(PROJECT_ROOT)),
)
def test_plugins_do_not_unconditionally_materialise_full_datasets(path: Path) -> None:
    """
    Opt-in plugin-quality test.

    New plugins should not turn a huge Parquet/DuckDB-backed dataset into a
    full pandas DataFrame at import time, panel construction time, or action
    execution time.

    This static check catches obvious unbounded uses of:
        - context.datasets.get_df()
        - context.datasets.get(...).df
        - source.to_pandas()

    Existing bundled plugins that are still being migrated can be kept in
    KNOWN_TRANSITIONAL_FULL_MATERIALISATION until they are converted.
    """

    relative_path = path.relative_to(PROJECT_ROOT)
    risks = find_full_materialisation_risks(path)

    if relative_path in KNOWN_TRANSITIONAL_FULL_MATERIALISATION and risks:
        pytest.xfail(
            f"{relative_path} still has full pandas materialisation risks:\n"
            + "\n".join(f"  - {risk}" for risk in risks)
            + "\nConvert this plugin to DatasetSource/Parquet-friendly access "
            "and then remove it from KNOWN_TRANSITIONAL_FULL_MATERIALISATION."
        )

    assert not risks, (
        f"{relative_path} may materialise a full dataset into pandas:\n"
        + "\n".join(f"  - {risk}" for risk in risks)
        + "\nUse context.datasets.get_source(...), list_columns(...), row_count(...), "
        "get_row_by_id(...), get_row_by_position(...), or bounded "
        "to_pandas(columns=..., limit=..., where_sql=...)."
    )