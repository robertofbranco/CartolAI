from pathlib import Path

BASE_URL = "https://api.cartola.globo.com"
GATOMESTRE_BASE = "https://api.gatomestre.globo.com"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CURRENT_SEASON = 2026
BUDGET = 140.0
CAPTAIN_BONUS = 1.5

FORMATION = {
    1: 1,  # GOL
    2: 2,  # LAT
    3: 2,  # ZAG
    4: 3,  # MEI
    5: 3,  # ATA
    6: 1,  # TEC
}

POSICAO_NOME = {
    1: "GOL",
    2: "LAT",
    3: "ZAG",
    4: "MEI",
    5: "ATA",
    6: "TEC",
}

SCOUT_POINTS = {
    "G": 8.0,
    "A": 5.0,
    "FT": 3.5,
    "FD": 1.2,
    "FF": 0.8,
    "FS": 0.5,
    "PS": 1.0,
    "PP": -4.0,
    "I": -0.1,
    "DP": 7.0,
    "DE": 1.3,
    "SG": 5.0,
    "DS": 1.2,
    "GC": -3.0,
    "CV": -3.0,
    "CA": -1.0,
    "GS": -1.0,
    "FC": -0.3,
    "PC": 0.3,
    "V": 1.0,
}

STATUS = {
    "Suspenso": 2,
    "Contundido": 3,
    "Duvida": 5,
    "Nulo": 6,
    "Provavel": 7
}

DEFAULT_LIGAS = [
    "1-mata-mata-brothers-do-graia-2026",
    "2o-mata-mata-brothers-do-graia",
]
