from pathlib import Path

import pandas as pd

from .config import DATA_DIR


def read_required_parquet(path: Path, hint: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {hint}")
    return pd.read_parquet(path)


def read_datasets(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    players_file = data_dir / "jogadores_por_rodada.parquet"
    matches_file = data_dir / "partidas.parquet"

    players = read_required_parquet(
        players_file,
        "Run collect_all_data.py to collect and merge the datasets.",
    )
    matches = read_required_parquet(
        matches_file,
        "Run collect_all_data.py to collect and merge the datasets.",
    )
    return players, matches
