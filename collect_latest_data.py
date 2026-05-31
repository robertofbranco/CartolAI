"""
Run all Cartola data collectors and build merged parquet datasets.

Examples:
    python collect_latest_data.py
    python collect_latest_data.py --current-season 2026
    python collect_latest_data.py --merge-only
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cartola_data import CURRENT_SEASON, DATA_DIR, collect_latest_api_data
from cartola_data.api import CartolaAPI
from cartola_data.current import (
    get_current_round,
    get_round_players_data_from_cartola,
    update_market,
    update_odds,
)
from cartola_data.file_manager import FileManager, PlayersDataset
from cartola_data.historic import get_players_data_from_caRtola, import_historic_season
from cartola_data.transforms import deduplicate_by_key

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_HISTORIC_YEARS = list(range(2023, CURRENT_SEASON))
MERGE_KEYS = {
    "jogadores_por_rodada": ["temporada", "rodada", "atleta_id"],
    "partidas": ["temporada", "rodada", "clube_id"],
    "odds": ["temporada", "rodada", "clube_id"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Get latest Cartola data, then merge it to the current season parquet."
    )
    parser.add_argument("--mercado-only", action="store_true")
    parser.add_argument("--skip-players", action="store_true")
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
    return FileManager(data_dir).find_yearly_parquets(dataset_names)


def get_previous_round_players_data(api, current_round: int) -> bool:
    target_round = current_round - 1
    players_dataset = PlayersDataset(season=CURRENT_SEASON, data_dir=DATA_DIR)
    players = players_dataset.read_or_empty()    
    
    new_players = get_players_data_from_caRtola(CURRENT_SEASON, [target_round])

    if new_players.empty:
        log.info("Coletando rodada %s pela API do cartola.", target_round)
        new_players = get_round_players_data_from_cartola(
            api,
            target_round,
            temporada=CURRENT_SEASON
        )

    if new_players.empty:
        raise RuntimeError(
            f"Dados de jogadores para rodada {target_round} ainda não estão disponíveis."
        )

    players = pd.concat([players, new_players], ignore_index=True)
    log.info(
            "Historico %s salvo: %s rodadas, %s atletas -> %s",
            CURRENT_SEASON,
            players["rodada"].nunique(),
            players["atleta_id"].nunique(),
            players_dataset.path,
        )
    
    if not players.empty:
        players = deduplicate_by_key(
            players,
            ["temporada", "rodada", "atleta_id"],
            prefer_played=True,
        )

    players_dataset.write(players)
    return True
            

def merge_partitioned_parquets(
    data_dir: Path = DATA_DIR,
    dataset_names: list[str] | None = None,
) -> dict[str, Path]:
    """Merge data/<dataset>_<year>.parquet into data/<dataset>.parquet."""
    file_manager = FileManager(data_dir)
    merged_files = {}
    groups = file_manager.find_yearly_parquets(dataset_names)

    for dataset, yearly_files in sorted(groups.items()):
        frames = []
        for year, parquet_path in sorted(yearly_files):
            df = file_manager.read_parquet(parquet_path)
            frames.append(add_year_if_missing(df, year))

        if not frames:
            continue

        merged_df = pd.concat(frames, ignore_index=True)
        before_dedupe = len(merged_df)
        merged_df = deduplicate_merged_dataset(dataset, merged_df)
        merged_df = sort_for_readability(merged_df)
        output_path = file_manager.dataset(dataset).write(merged_df)
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
    api = CartolaAPI()
    current_round = get_current_round(api)
    update_market(api)
    update_odds(api, current_round, CURRENT_SEASON, token)
    if args.mercado_only:        
        return

    collect_latest_api_data(
        api=api,
        current_round=current_round,
        season=CURRENT_SEASON
    )
    
    get_previous_round_players_data(api, current_round)


def main() -> None:
    args = parse_args()

    if not args.merge_only:
        run_collectors(args)

    if not args.skip_merge and not args.mercado_only:
        merged_files = merge_partitioned_parquets(
            data_dir=DATA_DIR,
            dataset_names=args.merge_datasets,
        )
        if not merged_files:
            log.warning("No yearly parquet files matched the merge pattern.")


if __name__ == "__main__":
    main()
