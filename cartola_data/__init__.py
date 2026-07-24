"""Data collection package for CartolaAI."""

from .api import CartolaAPI, CbfAPI, GatoMestreAPI
from .config import (
    BASE_URL,
    BUDGET,
    CAPTAIN_BONUS,
    CBF_API,
    CURRENT_SEASON,
    DATA_DIR,
    DEFAULT_LIGAS,
    FORMATION,
    GATOMESTRE_BASE,
    POSICAO_NOME,
    PROJECT_ROOT,
    SCOUT_POINTS,
    STATUS,
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
from .transforms import deduplicate_by_key, safe_filename

__all__ = [
    "BASE_URL",
    "BUDGET",
    "CAPTAIN_BONUS",
    "CBF_API",
    "CURRENT_SEASON",
    "DATA_DIR",
    "DEFAULT_LIGAS",
    "FORMATION",
    "GATOMESTRE_BASE",
    "POSICAO_NOME",
    "PROJECT_ROOT",
    "SCOUT_POINTS",
    "STATUS",
    "CartolaAPI",
    "CbfAPI",
    "GatoMestreAPI",
    "CurrentSeasonCollectionResult",
    "collect_current_season",
    "deduplicate_by_key",
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
