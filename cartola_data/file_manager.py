from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd

from .config import DATA_DIR


YEARLY_PARQUET_RE = re.compile(r"^(?P<dataset>.+)_(?P<year>\d{4})\.parquet$")


@dataclass
class FileManager:
    data_dir: Path = DATA_DIR

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def dataset_path(self, dataset: str, season: int | None = None) -> Path:
        suffix = f"_{season}" if season is not None else ""
        return self.data_dir / f"{dataset}{suffix}.parquet"

    def dataset(self, dataset: str, season: int | None = None) -> DatasetFile:
        try:
            dataset_class = DATASET_CLASSES[dataset]
        except KeyError as exc:
            raise ValueError(f"Unknown dataset: {dataset}") from exc
        return dataset_class(season=season, data_dir=self.data_dir)

    def cache_dir(self) -> Path:
        path = self.data_dir / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def dataset_exists(self, dataset: str, season: int | None = None) -> bool:
        return self.dataset_path(dataset, season).exists()

    def read_dataset(
        self,
        dataset: str,
        season: int | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        return pd.read_parquet(self.dataset_path(dataset, season), columns=columns)

    def read_parquet(
        self,
        path: Path,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        return pd.read_parquet(path, columns=columns)

    def read_dataset_or_empty(
        self,
        dataset: str,
        season: int | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if not self.dataset_exists(dataset, season):
            return pd.DataFrame()
        return self.read_dataset(dataset, season, columns=columns)

    def read_required_dataset(
        self,
        dataset: str,
        hint: str,
        season: int | None = None,
    ) -> pd.DataFrame:
        path = self.dataset_path(dataset, season)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. {hint}")
        return self.read_dataset(dataset, season)

    def save_dataset(
        self,
        df: pd.DataFrame,
        dataset: str,
        season: int | None = None,
    ) -> Path:
        path = self.dataset_path(dataset, season)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def find_yearly_parquets(
        self,
        dataset_names: list[str] | None = None,
    ) -> dict[str, list[tuple[int, Path]]]:
        selected = set(dataset_names or [])
        groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)

        for parquet_path in self.data_dir.glob("*.parquet"):
            match = YEARLY_PARQUET_RE.match(parquet_path.name)
            if not match:
                continue

            dataset = match.group("dataset")
            if selected and dataset not in selected:
                continue

            groups[dataset].append((int(match.group("year")), parquet_path))

        return groups


@dataclass
class DatasetFile:
    season: int | None = None
    data_dir: Path = DATA_DIR

    name: ClassVar[str]

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        suffix = f"_{self.season}" if self.season is not None else ""
        return self.data_dir / f"{self.name}{suffix}.parquet"

    def exists(self) -> bool:
        return self.path.exists()

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(self.path, columns=columns)

    def read_or_empty(self, columns: list[str] | None = None) -> pd.DataFrame:
        if not self.exists():
            return pd.DataFrame()
        return self.read(columns=columns)

    def read_required(self, hint: str) -> pd.DataFrame:
        if not self.exists():
            raise FileNotFoundError(f"{self.path} not found. {hint}")
        return self.read()

    def write(self, df: pd.DataFrame) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.path, index=False)
        return self.path
    
    def append(self, df: pd.DataFrame) -> Path:
        if self.exists():
            existing = self.read()
            df = pd.concat([existing, df], ignore_index=True)
        return self.write(df)


class PlayersDataset(DatasetFile):
    name = "jogadores_por_rodada"


class MatchesDataset(DatasetFile):
    name = "partidas_com_xG"


class OddsDataset(DatasetFile):
    name = "odds"


class CurrentMarketDataset(DatasetFile):
    name = "mercado_atual"


class CartolaUsersMeanDataset(DatasetFile):
    name = "medias_cartoleiros"


class LeagueBracketsDataset(DatasetFile):
    name = "chaves_ligas"


DATASET_CLASSES = {
    dataset_class.name: dataset_class
    for dataset_class in [
        PlayersDataset,
        MatchesDataset,
        OddsDataset,
        CurrentMarketDataset,
        CartolaUsersMeanDataset,
        LeagueBracketsDataset,
    ]
}
