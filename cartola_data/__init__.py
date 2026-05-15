"""Data collection package for CartolaAI."""

from .api import CartolaAPI, GatoMestreAPI
from .config import (
    BASE_URL,
    BUDGET,
    CAPTAIN_BONUS,
    CURRENT_SEASON,
    DATA_DIR,
    DEFAULT_LIGAS,
    FORMATION,
    GATOMESTRE_BASE,
    POSICAO_NOME,
    SCOUT_POINTS,
)
from .current import (
    CurrentSeasonCollectionResult,
    collect_current_season,
    get_cartola_users_mean,
    get_current_market,
    get_current_round,
    get_league_brackets,
    get_odds,
    get_players_data,
    missing_rounds,
    preparar_partidas,
)
from .datasets import read_datasets
from .transforms import calculate_player_running_average, safe_filename

__all__ = [
    "BASE_URL",
    "BUDGET",
    "CAPTAIN_BONUS",
    "CURRENT_SEASON",
    "DATA_DIR",
    "DEFAULT_LIGAS",
    "FORMATION",
    "GATOMESTRE_BASE",
    "POSICAO_NOME",
    "SCOUT_POINTS",
    "CartolaAPI",
    "GatoMestreAPI",
    "CurrentSeasonCollectionResult",
    "calculate_player_running_average",
    "collect_current_season",
    "get_cartola_users_mean",
    "get_current_market",
    "get_current_round",
    "get_league_brackets",
    "get_odds",
    "get_players_data",
    "missing_rounds",
    "preparar_partidas",
    "read_datasets",
    "safe_filename",
]
