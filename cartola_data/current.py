import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .api import CartolaAPI, GatoMestreAPI
from .config import CURRENT_SEASON, DATA_DIR, DEFAULT_LIGAS
from .file_manager import (
    CartolaUsersMeanDataset,
    CurrentMarketDataset,
    LeagueBracketsDataset,
    MatchesDataset,
    OddsDataset,
    PlayersDataset,
)
from .transforms import deduplicate_by_key

log = logging.getLogger(__name__)


@dataclass
class CurrentSeasonCollectionResult:
    season: int
    rounds: list[int]
    files: dict[str, Path] = field(default_factory=dict)


def get_current_round(api: CartolaAPI) -> int:
    status = api.market_status()
    current_round = status.get("rodada_atual", status.get("rodada", {}).get("rodada_atual", 1))
    log.info("Rodada atual: %s", current_round)
    return int(current_round)


def missing_rounds(existing_df: pd.DataFrame, target_rounds: list[int]) -> list[int]:
    if existing_df.empty or "rodada" not in existing_df.columns:
        return target_rounds

    collected_rounds = set(existing_df["rodada"].dropna().astype(int).unique())
    return [round_number for round_number in target_rounds if round_number not in collected_rounds]


def _iter_atletas(atletas: dict | list) -> list[tuple[int | None, dict]]:
    if isinstance(atletas, dict):
        return [(int(atleta_id), atleta) for atleta_id, atleta in atletas.items()]
    return [(atleta.get("atleta_id"), atleta) for atleta in atletas]


def _load_current_market_by_atleta(data_dir: Path | None = None) -> dict[int, dict]:
    market_dataset = CurrentMarketDataset(data_dir=data_dir or DATA_DIR)
    if not market_dataset.exists():
        log.warning("Mercado atual nao encontrado: %s", market_dataset.path)
        return {}

    try:
        market = market_dataset.read()
    except Exception as exc:
        log.warning("Nao foi possivel ler mercado atual de %s: %s", market_dataset.path, exc)
        return {}

    if market.empty or "atleta_id" not in market.columns:
        return {}

    market = market.dropna(subset=["atleta_id"]).drop_duplicates("atleta_id", keep="last")
    return {
        int(row["atleta_id"]): row
        for row in market.to_dict("records")
    }


def _market_value(market_row: dict, column: str, default):
    value = market_row.get(column, default)
    return default if pd.isna(value) else value


def get_round_players_data_from_cartola(
    api: CartolaAPI,
    rodada: int,
    temporada: int = CURRENT_SEASON,
) -> pd.DataFrame:
    """Collect player scores by round from /atletas/pontuados."""
    rows = []
    clubes = {int(key): value.get("nome", "") for key, value in api.clubes().items()}
    market_by_atleta = _load_current_market_by_atleta()
    try:
        data = api.atletas_pontuados(rodada)
    except Exception as exc:
        log.warning("Rodada %s indisponivel: %s", rodada, exc)
        return pd.DataFrame()

    for atleta_id, atleta in _iter_atletas(data.get("atletas", {})):
        scout = atleta.get("scout", {}) or {}
        market_row = market_by_atleta.get(int(atleta_id or 0), {})
        rows.append(
            {
                "temporada": temporada,
                "rodada": rodada,
                "atleta_id": atleta_id,
                **{f"scout_{key}": value for key, value in scout.items()},
                "apelido": atleta.get("apelido"),
                "pontos": atleta.get("pontuacao", 0.0),
                "posicao_id": atleta.get("posicao_id"),
                "clube_id": atleta.get("clube_id"),
                "clube_nome": clubes.get(atleta.get("clube_id"), ""),
                "jogou": atleta.get("entrou_em_campo", True),
                "status_id": _market_value(market_row, "status_id", None),
                "preco": _market_value(market_row, "preco", 0.0),
                "media": _market_value(market_row, "media", 0.0),
                "jogos": _market_value(market_row, "jogos", 0),
            }
        )
    time.sleep(0.3)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = deduplicate_by_key(
        df,
        ["temporada", "rodada", "atleta_id"],
        prefer_played=True,
    )
    return df


def get_current_market(api: CartolaAPI) -> pd.DataFrame:
    """Return the current /atletas/mercado snapshot, one row per athlete."""
    data = api.mercado()
    rows = [
        {
            "atleta_id": atleta.get("atleta_id"),
            "apelido": atleta.get("apelido"),
            "posicao_id": atleta.get("posicao_id"),
            "clube_id": atleta.get("clube_id"),
            "preco": atleta.get("preco_num", 0.0),
            "media": atleta.get("media_num", 0.0),
            "status_id": atleta.get("status_id"),
            "jogos": atleta.get("jogos_num", 0),
        }
        for atleta in data.get("atletas", [])
    ]

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return (
        df.dropna(subset=["atleta_id"])
        .drop_duplicates("atleta_id", keep="last")
        .sort_values(["posicao_id", "clube_id", "atleta_id"])
        .reset_index(drop=True)
    )


def _season_clube_ids(temporada: int | None, data_dir: Path = DATA_DIR) -> set[int]:
    if temporada is None:
        return set()

    clube_ids: set[int] = set()
    for dataset in [
        PlayersDataset(season=temporada, data_dir=data_dir),
        MatchesDataset(season=temporada, data_dir=data_dir),
    ]:
        if not dataset.exists():
            continue

        try:
            df = dataset.read(columns=["clube_id"])
        except Exception as exc:
            log.warning("Nao foi possivel ler clubes de %s: %s", dataset.path, exc)
            continue

        clube_ids.update(df["clube_id"].dropna().astype(int).unique().tolist())

    return clube_ids


def _build_abbr_to_clube_id(
    clubes_raw: dict,
    temporada: int | None = None,
    data_dir: Path = DATA_DIR,
) -> dict[str, int]:
    """Build an abbreviation map while avoiding stale duplicate club IDs."""
    abbr_to_id: dict[str, int] = {}
    season_clube_ids = _season_clube_ids(temporada, data_dir=data_dir)

    for club_id, club in clubes_raw.items():
        abbr = (club.get("abreviacao") or "").upper()
        if not abbr:
            continue

        clube_id = int(club_id)
        if season_clube_ids and clube_id in season_clube_ids:
            abbr_to_id[abbr] = clube_id

    for club_id, club in clubes_raw.items():
        abbr = (club.get("abreviacao") or "").upper()
        if abbr:
            abbr_to_id.setdefault(abbr, int(club_id))

    return abbr_to_id


def get_odds(
    gato_api: GatoMestreAPI,
    clubes_raw: dict,
    rodadas_alvo: list[int],
    temporada: int | None = None,
) -> pd.DataFrame:
    abbr_to_id = _build_abbr_to_clube_id(clubes_raw, temporada=temporada)

    rows = []
    for rodada in tqdm(rodadas_alvo, desc="Coletando odds"):
        if rodada < 2:
            continue
        try:
            data = gato_api.favoritos(rodada)
        except Exception as exc:
            log.warning("Odds rodada %s indisponiveis: %s", rodada, exc)
            continue

        for match in data.get("result", []) or []:
            home_abbr = (match.get("homeTeam", {}).get("abbr") or "").upper()
            away_abbr = (match.get("awayTeam", {}).get("abbr") or "").upper()
            home_id = abbr_to_id.get(home_abbr)
            away_id = abbr_to_id.get(away_abbr)
            if home_id is None or away_id is None:
                log.warning("Rodada %s: time sem mapeamento (%s/%s)", rodada, home_abbr, away_abbr)
                continue

            odds = match.get("odds", {}) or {}
            win = float(odds.get("win", 0)) / 100.0
            draw = float(odds.get("tie", 0)) / 100.0
            loss = float(odds.get("loss", 0)) / 100.0
            home_row = {
                "rodada": rodada,
                "clube_id": home_id,
                "prob_win": win,
                "prob_draw": draw,
                "prob_loss": loss,
            }
            away_row = {
                "rodada": rodada,
                "clube_id": away_id,
                "prob_win": loss,
                "prob_draw": draw,
                "prob_loss": win,
            }
            if temporada is not None:
                home_row["temporada"] = temporada
                away_row["temporada"] = temporada
            rows.extend([home_row, away_row])
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    keys = ["rodada", "clube_id"]
    if "temporada" in df.columns:
        keys.insert(0, "temporada")
    return deduplicate_by_key(df, keys)


def preparar_partidas(
    api: CartolaAPI,
    rodadas_alvo: list[int],
    temporada: int | None = None,
) -> pd.DataFrame:
    rows = []
    for rodada in tqdm(rodadas_alvo, desc="Coletando partidas"):
        try:
            data = api.partidas(rodada)
        except Exception as exc:
            log.warning("Partidas rodada %s: %s", rodada, exc)
            continue

        for partida in data.get("partidas", []):
            casa = partida.get("clube_casa_id")
            fora = partida.get("clube_visitante_id")
            home = {
                "rodada": rodada,
                "clube_id": casa,
                "mando": 1,
                "clube_adversario_id": fora,
                "gols_feitos_clube": partida.get("placar_oficial_mandante"),
                "gols_sofridos_clube": partida.get("placar_oficial_visitante"),
            }
            away = {
                "rodada": rodada,
                "clube_id": fora,
                "mando": -1,
                "clube_adversario_id": casa,
                "gols_feitos_clube": partida.get("placar_oficial_visitante"),
                "gols_sofridos_clube": partida.get("placar_oficial_mandante"),
            }
            if temporada is not None:
                home["temporada"] = temporada
                away["temporada"] = temporada
            rows.extend([home, away])
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    keys = ["rodada", "clube_id"]
    if "temporada" in df.columns:
        keys.insert(0, "temporada")
    return deduplicate_by_key(df, keys)


def get_cartola_users_mean(api: CartolaAPI, rodadas_alvo: list[int]) -> pd.DataFrame:
    rows = []
    for rodada in rodadas_alvo:
        data = api.pos_rodada(rodada)
        rows.append({"rodada": rodada, "media_cartoleiros": data.get("media_pontos")})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["media_cartoleiros"] = pd.to_numeric(df["media_cartoleiros"], errors="coerce")
    df = df.dropna(subset=["media_cartoleiros"])
    return deduplicate_by_key(df, ["rodada"]).sort_values("rodada").reset_index(drop=True)


def get_league_brackets(api: CartolaAPI, ligas: list[str]) -> pd.DataFrame:
    rows = []
    for liga in ligas:
        chaves_mata_mata = api.league(liga).get("chaves_mata_mata", {})
        for rodada, chaves in chaves_mata_mata.items():
            for chave_index, chave in enumerate(chaves):
                if chave.get("vencedor_id") is None:
                    break

                rows.append(
                    {
                        "liga": liga,
                        "rodada": int(rodada),
                        "chave_index": chave_index,
                        "vencedor_id": chave.get("vencedor_id"),
                        "time_mandante_id": chave.get("time_mandante_id"),
                        "time_visitante_id": chave.get("time_visitante_id"),
                        "time_mandante_pontuacao": chave.get("time_mandante_pontuacao", 0),
                        "time_visitante_pontuacao": chave.get("time_visitante_pontuacao", 0),
                        "pontos": max(
                            chave.get("time_mandante_pontuacao", 0),
                            chave.get("time_visitante_pontuacao", 0),
                        ),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return deduplicate_by_key(df, ["liga", "rodada", "chave_index"])


def _has_round(df: pd.DataFrame, rodada: int) -> bool:
    return "rodada" in df.columns and rodada in df["rodada"].dropna().astype(int).unique()


def update_odds(api: CartolaAPI, current_round: int, season: int, token: str):
    if not token:
        log.info("CARTOLA_TOKEN nao definido - odds nao coletadas.")
        return
    
    odds_dataset = OddsDataset(season=season)
    odds_df = odds_dataset.read_or_empty()
    gato_api = GatoMestreAPI(token=token, temporada=season)
    new_odds = get_odds(gato_api, api.clubes(), [current_round], temporada=season)
    if new_odds.empty:
        log.warning(f"Odds indisponiveis para a rodada {current_round}")
    elif _has_round(odds_df, current_round):
        odds_df = odds_df[~((odds_df["temporada"] == season) & (odds_df["rodada"] == current_round))]
        odds_df = pd.concat([odds_df, new_odds])
        odds_file = odds_dataset.write(odds_df)
    else:
        odds_file = odds_dataset.append(new_odds)

    log.info("Odds salvas: %s -> %s", len(new_odds), odds_file)


def update_market(api: CartolaAPI):
    market = get_current_market(api)
    if market.empty:
        log.warning("Mercado atual vazio, nao sera atualizado.")

    market_file = CurrentMarketDataset().write(market)
    log.info("Mercado atual salvo: %s atletas -> %s", len(market), market_file)


def collect_latest_api_data(
    api: CartolaAPI,
    current_round: int,
    season: int = CURRENT_SEASON,    
):
    """Collect live Cartola/Gato Mestre data and persist project datasets."""
    previous_round = current_round - 1    

    if not current_round:
        log.error("Nenhuma rodada para coletar.")
        return
        
    matches_dataset = MatchesDataset(season=season)
    matches_df = matches_dataset.read_or_empty()
    if not _has_round(matches_df, current_round):
        new_matches = preparar_partidas(api, [current_round], temporada=season)
        matches_file = matches_dataset.append(new_matches)
        log.info("Partidas salvas: %s -> %s", len(new_matches), matches_file)

    users_mean_dataset = CartolaUsersMeanDataset()
    users_mean_df = users_mean_dataset.read_or_empty()
    if not _has_round(users_mean_df, previous_round):
        user_means = get_cartola_users_mean(api, [previous_round])
        if not user_means.empty:
            user_means_file = users_mean_dataset.append(user_means)
            log.info("Médias dos cartoleiros salvas: %s -> %s", len(user_means), user_means_file)

    league_brackets = get_league_brackets(api, DEFAULT_LIGAS)
    if not league_brackets.empty:
        league_brackets_file = LeagueBracketsDataset().write(league_brackets)
        log.info("Chaves das ligas salvas: %s -> %s", len(league_brackets), league_brackets_file)
