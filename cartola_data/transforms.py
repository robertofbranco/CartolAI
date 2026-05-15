import re

import pandas as pd


def safe_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def _played_mask(df: pd.DataFrame) -> pd.Series:
    if "jogou" not in df.columns:
        return pd.Series(True, index=df.index)

    played = df["jogou"]
    if played.dtype == "object":
        return played.astype(str).str.lower().isin(["true", "1", "sim", "yes"])
    return played.fillna(False).astype(bool)


def calculate_player_running_average(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate each player's pre-round average from previous played rounds."""
    required_columns = {"atleta_id", "rodada", "pontos"}
    if df.empty or not required_columns.issubset(df.columns):
        return df

    result = df.copy()
    group_keys = ["atleta_id"]
    if "temporada" in result.columns:
        group_keys.insert(0, "temporada")

    result = result.sort_values(group_keys + ["rodada"]).reset_index(drop=True)

    points = pd.to_numeric(result["pontos"], errors="coerce").fillna(0)
    played_points = points.where(_played_mask(result))
    groupers = [result[key] for key in group_keys]
    shifted_points = played_points.groupby(groupers).shift(1)

    result["media"] = (
        shifted_points.groupby(groupers)
        .expanding()
        .mean()
        .reset_index(level=list(range(len(group_keys))), drop=True)
        .fillna(0.0)
    )
    return result


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
