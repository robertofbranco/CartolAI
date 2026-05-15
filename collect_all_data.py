"""
Run all Cartola data collectors and build merged parquet datasets.

Examples:
    python collect_all_data.py
    python collect_all_data.py --historic-years 2024 2025 --current-season 2026
    python collect_all_data.py --merge-only
"""

import argparse
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cartola_data import CURRENT_SEASON, DATA_DIR, collect_current_season
from cartola_data.historic import import_historic_season
from cartola_data.transforms import deduplicate_by_key

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

YEARLY_PARQUET_RE = re.compile(r"^(?P<dataset>.+)_(?P<year>\d{4})\.parquet$")
DEFAULT_HISTORIC_YEARS = list(range(2023, CURRENT_SEASON))
MERGE_KEYS = {
    "jogadores_por_rodada": ["temporada", "rodada", "atleta_id"],
    "partidas": ["temporada", "rodada", "clube_id"],
    "odds": ["temporada", "rodada", "clube_id"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run historic and current Cartola collectors, then merge yearly parquets."
    )
    parser.add_argument("--historic-years", type=int, nargs="+", default=DEFAULT_HISTORIC_YEARS)
    parser.add_argument("--current-season", type=int, default=CURRENT_SEASON)
    parser.add_argument("--skip-historic", action="store_true")
    parser.add_argument("--skip-current", action="store_true")
    parser.add_argument("--skip-gato", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument(
        "--merge-datasets",
        nargs="+",
        default=None,
        help="Optional dataset names to merge, for example: jogadores_por_rodada partidas odds",
    )
    return parser.parse_args()


def add_year_if_missing(df: pd.DataFrame, year: int) -> pd.DataFrame:
    if "temporada" in df.columns:
        return df

    df = df.copy()
    df["temporada"] = year
    return df


def sort_for_readability(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in ["temporada", "rodada", "atleta_id", "clube_id"]
        if column in df.columns
    ]
    if not sort_columns:
        return df.reset_index(drop=True)
    return df.sort_values(sort_columns).reset_index(drop=True)


def deduplicate_merged_dataset(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    key_columns = MERGE_KEYS.get(dataset)
    if not key_columns or any(column not in df.columns for column in key_columns):
        return df

    return deduplicate_by_key(
        df,
        key_columns,
        prefer_played=dataset == "jogadores_por_rodada",
    )


def find_yearly_parquets(
    data_dir: Path,
    dataset_names: list[str] | None = None,
) -> dict[str, list[tuple[int, Path]]]:
    selected = set(dataset_names or [])
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)

    for parquet_path in data_dir.glob("*.parquet"):
        match = YEARLY_PARQUET_RE.match(parquet_path.name)
        if not match:
            continue

        dataset = match.group("dataset")
        if selected and dataset not in selected:
            continue

        groups[dataset].append((int(match.group("year")), parquet_path))

    return groups


def merge_partitioned_parquets(
    data_dir: Path = DATA_DIR,
    dataset_names: list[str] | None = None,
) -> dict[str, Path]:
    """Merge data/<dataset>_<year>.parquet into data/<dataset>.parquet."""
    merged_files = {}
    groups = find_yearly_parquets(data_dir, dataset_names)

    for dataset, yearly_files in sorted(groups.items()):
        frames = []
        for year, parquet_path in sorted(yearly_files):
            df = pd.read_parquet(parquet_path)
            frames.append(add_year_if_missing(df, year))

        if not frames:
            continue

        merged_df = pd.concat(frames, ignore_index=True)
        before_dedupe = len(merged_df)
        merged_df = deduplicate_merged_dataset(dataset, merged_df)
        merged_df = sort_for_readability(merged_df)
        output_path = data_dir / f"{dataset}.parquet"
        merged_df.to_parquet(output_path, index=False)
        merged_files[dataset] = output_path
        log.info(
            "Merged %s yearly files into %s (%s rows)",
            len(yearly_files),
            output_path,
            len(merged_df),
        )
        if len(merged_df) != before_dedupe:
            log.info("Removed %s duplicate rows from %s", before_dedupe - len(merged_df), dataset)

    return merged_files


def run_collectors(args: argparse.Namespace) -> None:
    token = os.environ.get("CARTOLA_TOKEN")

    if not args.skip_historic:
        for year in args.historic_years:
            import_historic_season(
                year=year,                
                token=token,
                collect_gato_data=not args.skip_gato,
            )

    if not args.skip_current:
        collect_current_season(
            temporada=args.current_season,            
            token=token,
        )


def main() -> None:
    args = parse_args()

    if not args.merge_only:
        run_collectors(args)

    if not args.skip_merge:
        merged_files = merge_partitioned_parquets(
            data_dir=DATA_DIR,
            dataset_names=args.merge_datasets,
        )
        if not merged_files:
            log.warning("No yearly parquet files matched the merge pattern.")


if __name__ == "__main__":
    main()
