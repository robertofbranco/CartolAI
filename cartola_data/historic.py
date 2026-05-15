import logging
import unicodedata
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from .api import CartolaAPI, GatoMestreAPI
from .config import DATA_DIR
from .transforms import deduplicate_by_key

log = logging.getLogger(__name__)


@dataclass
class HistoricImportResult:
    year: int
    files: dict[str, Path] = field(default_factory=dict)


CARTOLA_REPO_BASE = (
    "https://raw.githubusercontent.com/henriquepgomide/caRtola/master"
    "/data/01_raw/{year}/rodada-{rodada}.csv"
)

CARTOLA_RENAME = {
    "atletas.rodada_id": "rodada",
    "atletas.atleta_id": "atleta_id",
    "atletas.apelido": "apelido",
    "atletas.posicao_id": "posicao_id",
    "atletas.clube_id": "clube_id",
    "atletas.clube.id.full.name": "clube_abreviacao",
    "atletas.status_id": "status_id",
    "atletas.pontos_num": "pontos",
    "atletas.preco_num": "preco",
    "atletas.media_num": "media",
    "atletas.jogos_num": "jogos",
    "atletas.entrou_em_campo": "entrou_em_campo",
}

RECOGNIZED_SCOUTS = {
    "G",
    "A",
    "FT",
    "FD",
    "FF",
    "FS",
    "PS",
    "PP",
    "I",
    "SG",
    "DS",
    "GC",
    "CV",
    "CA",
    "GS",
    "FC",
    "PC",
    "DP",
    "DE",
    "V",
}

NOME_TO_ABBR = {
    "Flamengo": "FLA",
    "Botafogo": "BOT",
    "Corinthians": "COR",
    "Bahia": "BAH",
    "Fluminense": "FLU",
    "Vasco": "VAS",
    "Palmeiras": "PAL",
    "Sao Paulo": "SAO",
    "Santos": "SAN",
    "Bragantino": "RBB",
    "Atletico-MG": "CAM",
    "Cruzeiro": "CRU",
    "Gremio": "GRE",
    "Internacional": "INT",
    "Juventude": "JUV",
    "Vitoria": "VIT",
    "Criciuma": "CRI",
    "Goias": "GOI",
    "Athletico-PR": "CAP",
    "Coritiba": "CFC",
    "America-MG": "AME",
    "Fortaleza": "FOR",
    "Atletico-GO": "ACG",
    "Cuiaba": "CUI",
    "Avai": "AVA",
    "Ceara": "CEA",
}


def baixar_rodada(year: int, rodada: int) -> pd.DataFrame:
    url = CARTOLA_REPO_BASE.format(year=year, rodada=rodada)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def normalizar_rodada(df_raw: pd.DataFrame, year: int) -> pd.DataFrame:
    columns = {source: target for source, target in CARTOLA_RENAME.items() if source in df_raw.columns}
    normalized = df_raw[list(columns)].rename(columns=columns).copy()
    normalized["temporada"] = year

    for col in df_raw.columns:
        if col in RECOGNIZED_SCOUTS:
            normalized[f"scout_{col}"] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0)

    integer_columns = ["rodada", "atleta_id", "clube_id", "posicao_id", "status_id", "jogos"]
    for col in integer_columns:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce").astype("Int64")

    for col in ["pontos", "preco", "media"]:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce").astype(float)

    jogou_col = "entrou_em_campo"
    if jogou_col in df_raw.columns:
        normalized["jogou"] = (
            pd.to_numeric(df_raw[jogou_col], errors="coerce")
            .fillna(0)
            .astype(bool)
        )
    else:
        normalized["jogou"] = normalized["pontos"].fillna(0) != 0

    return normalized.dropna(subset=["atleta_id", "rodada", "clube_id"])


def importar_historico(year: int, rodadas: list[int]) -> pd.DataFrame:
    frames = []
    for rodada in tqdm(rodadas, desc=f"caRtola {year}"):
        try:
            frames.append(normalizar_rodada(baixar_rodada(year, rodada), year))
        except Exception as exc:
            log.warning("Falha rodada %s/%s: %s", year, rodada, exc)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["clube_nome"] = df["clube_abreviacao"]
    df = deduplicate_by_key(
        df,
        ["temporada", "rodada", "atleta_id"],
        prefer_played=True,
    )
    return df.sort_values(["atleta_id", "rodada"]).reset_index(drop=True)


def importar_partidas_odds(
    gato_api: GatoMestreAPI,
    abbr_to_clube_id: dict[str, int],
    rodadas: list[int],
    temporada: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    partidas_rows = []
    odds_rows = []
    for rodada in tqdm(rodadas, desc=f"Gato Mestre {temporada}"):
        if rodada < 2:
            continue
        try:
            data = gato_api.favoritos(rodada)
        except Exception as exc:
            log.warning("Rodada %s: %s", rodada, exc)
            continue

        for match in data.get("result", []) or []:
            home_abbr = (match.get("homeTeam", {}).get("abbr") or "").upper()
            away_abbr = (match.get("awayTeam", {}).get("abbr") or "").upper()
            home_id = abbr_to_clube_id.get(home_abbr)
            away_id = abbr_to_clube_id.get(away_abbr)
            if home_id is None or away_id is None:
                log.warning("Sem mapeamento para %s/%s (r%s)", home_abbr, away_abbr, rodada)
                continue

            partidas_rows.extend(
                [
                    {
                        "temporada": temporada,
                        "rodada": rodada,
                        "clube_id": home_id,
                        "mando": 1,
                        "clube_adversario_id": away_id,
                    },
                    {
                        "temporada": temporada,
                        "rodada": rodada,
                        "clube_id": away_id,
                        "mando": -1,
                        "clube_adversario_id": home_id,
                    },
                ]
            )

            odds = match.get("odds", {}) or {}
            win = float(odds.get("win", 0)) / 100.0
            draw = float(odds.get("tie", 0)) / 100.0
            loss = float(odds.get("loss", 0)) / 100.0
            odds_rows.extend(
                [
                    {
                        "temporada": temporada,
                        "rodada": rodada,
                        "clube_id": home_id,
                        "prob_win": win,
                        "prob_draw": draw,
                        "prob_loss": loss,
                    },
                    {
                        "temporada": temporada,
                        "rodada": rodada,
                        "clube_id": away_id,
                        "prob_win": loss,
                        "prob_draw": draw,
                        "prob_loss": win,
                    },
                ]
            )

    partidas = pd.DataFrame(partidas_rows)
    odds = pd.DataFrame(odds_rows)
    if not partidas.empty:
        partidas = deduplicate_by_key(partidas, ["temporada", "rodada", "clube_id"])
    if not odds.empty:
        odds = deduplicate_by_key(odds, ["temporada", "rodada", "clube_id"])
    return partidas, odds


def _normalize_name(value: str) -> str:
    for encoding in ("cp1250", "latin1"):
        try:
            value = value.encode(encoding).decode("utf-8")
            break
        except UnicodeError:
            pass

    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


def build_abbr_to_clube_id(data_dir: Path, year: int) -> dict[str, int]:
    abbr_to_id: dict[str, int] = {}

    def ingest_parquet(parquet: Path) -> None:
        try:
            df = pd.read_parquet(parquet, columns=["clube_id", "clube_abreviacao"])
        except Exception:
            return

        df = df.dropna(subset=["clube_id", "clube_abreviacao"]).drop_duplicates("clube_abreviacao")
        for _, row in df.iterrows():
            raw = str(row["clube_abreviacao"]).strip()
            cid = int(row["clube_id"])
            if 2 <= len(raw) <= 4 and raw.isalpha():
                abbr_to_id.setdefault(raw.upper(), cid)
            else:
                abbr = NOME_TO_ABBR.get(raw) or NOME_TO_ABBR.get(_normalize_name(raw))
                if abbr:
                    abbr_to_id.setdefault(abbr, cid)

    primary = data_dir / f"jogadores_por_rodada_{year}.parquet"
    if primary.exists():
        ingest_parquet(primary)

    for parquet in sorted(data_dir.glob("jogadores_por_rodada_*.parquet")):
        if parquet != primary:
            ingest_parquet(parquet)

    try:
        for club_id, club in (CartolaAPI().clubes() or {}).items():
            abbr = (club.get("abreviacao") or "").upper()
            if abbr:
                abbr_to_id.setdefault(abbr, int(club_id))
    except Exception as exc:
        log.warning("Falha ao buscar /clubes da Cartola API: %s", exc)

    return abbr_to_id


def get_players_data_from_caRtola(year: int, rounds: list[int] | None = None) -> pd.DataFrame:
    target_rounds = rounds or list(range(1, 39))
    log.info("Importando historico de %s do caRtola.", year)
    return importar_historico(year, target_rounds)


def import_historic_season(
    year: int,
    rounds: list[int] | None = None,
    token: str | None = None,
    collect_gato_data: bool = True,
) -> HistoricImportResult:
    target_rounds = rounds or list(range(1, 39))
    result = HistoricImportResult(year=year)

    players = get_players_data_from_caRtola(year, target_rounds)
    if players.empty:
        log.error("Nenhum dado coletado para %s.", year)
        return result

    players_path = DATA_DIR / f"jogadores_por_rodada_{year}.parquet"
    players.to_parquet(players_path, index=False)
    result.files["jogadores_por_rodada"] = players_path
    log.info(
        "Historico %s salvo: %s rodadas, %s atletas -> %s",
        year,
        players["rodada"].nunique(),
        players["atleta_id"].nunique(),
        players_path,
    )

    if not collect_gato_data:
        return result

    if not token:
        log.warning("CARTOLA_TOKEN nao definido - partidas e odds de %s nao coletadas.", year)
        return result

    abbr_to_id = build_abbr_to_clube_id(DATA_DIR, year)
    gato_api = GatoMestreAPI(token=token, temporada=year)
    matches, odds = importar_partidas_odds(gato_api, abbr_to_id, target_rounds, year)

    if not matches.empty:
        matches_path = DATA_DIR / f"partidas_{year}.parquet"
        matches.to_parquet(matches_path, index=False)
        result.files["partidas"] = matches_path
        log.info("Partidas %s salvas -> %s", year, matches_path)

    if not odds.empty:
        odds_path = DATA_DIR / f"odds_{year}.parquet"
        odds.to_parquet(odds_path, index=False)
        result.files["odds"] = odds_path
        log.info("Odds %s salvas -> %s", year, odds_path)

    return result
