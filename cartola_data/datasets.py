from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from .file_manager import MatchesDataset, OddsDataset, PlayersDataset


def read_datasets(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hint = "Run collect_all_data.py to collect and merge the datasets."

    players = PlayersDataset(data_dir=data_dir).read_required(hint)
    matches = MatchesDataset(data_dir=data_dir).read_required(hint)
    odds = OddsDataset(data_dir=data_dir).read_required(hint)

    return players, matches, odds
