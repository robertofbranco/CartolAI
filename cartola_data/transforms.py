import re

import pandas as pd


def safe_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def deduplicate_by_key(
    df: pd.DataFrame,
    key_columns: list[str],
    prefer_played: bool = False,
) -> pd.DataFrame:
    """Deduplicate a dataframe using stable project conventions."""
    if df.empty:
        return df

    missing_columns = [column for column in key_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Cannot deduplicate without columns: {missing_columns}")

    result = df.copy()
    sort_columns = key_columns.copy()
    ascending = [True] * len(sort_columns)

    if prefer_played and "jogou" in result.columns:
        result["_played_sort"] = result["jogou"].fillna(False).astype(bool)
        sort_columns.append("_played_sort")
        ascending.append(True)

    result = (
        result.sort_values(sort_columns, ascending=ascending, kind="mergesort")
        .drop_duplicates(key_columns, keep="last")
        .drop(columns=["_played_sort"], errors="ignore")
        .reset_index(drop=True)
    )
    return result
