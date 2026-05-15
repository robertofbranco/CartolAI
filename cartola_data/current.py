import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .api import CartolaAPI, GatoMestreAPI
from .config import CURRENT_SEASON, DATA_DIR, DEFAULT_LIGAS
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


def get_players_data(
    api: CartolaAPI,
    rodadas_alvo: list[int],
    temporada: int = CURRENT_SEASON,
) -> pd.DataFrame:
    """Collect player scores by round from /atletas/pontuados."""
    rows = []
    clubes = {int(key): value["nome"] for key, value in api.clubes().items()}

    for rodada in tqdm(rodadas_alvo, desc="Coletando rodadas"):
        try:
            data = api.atletas_pontuados(rodada)
        except Exception as exc:
            log.warning("Rodada %s indisponivel: %s", rodada, exc)
            continue

        for atleta_id, atleta in _iter_atletas(data.get("atletas", {})):
            scout = atleta.get("scout", {}) or {}
            rows.append(
                {
                    "temporada": temporada,
                    "rodada": rodada,
                    "atleta_id": atleta_id,
                    "apelido": atleta.get("apelido"),
                    "posicao_id": atleta.get("posicao_id"),
                    "clube_id": atleta.get("clube_id"),
                    "clube_nome": clubes.get(atleta.get("clube_id"), ""),
                    "status_id": atleta.get("status_id"),
                    "pontos": atleta.get("pontuacao", atleta.get("pontos_num", 0.0)),
                    "preco": atleta.get("preco_num", 0.0),
                    "media": 0.0,
                    "jogos": 0,
                    "jogou": atleta.get("entrou_em_campo", True),
                    **{f"scout_{key}": value for key, value in scout.items()},
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


def get_odds(
    gato_api: GatoMestreAPI,
    clubes_raw: dict,
    rodadas_alvo: list[int],
    temporada: int | None = None,
) -> pd.DataFrame:
    abbr_to_id = {
        (club.get("abreviacao") or "").upper(): int(club_id)
        for club_id, club in clubes_raw.items()
        if club.get("abreviacao")
    }

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
            }
            away = {
                "rodada": rodada,
                "clube_id": fora,
                "mando": -1,
                "clube_adversario_id": casa,
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


def _save_yearly_dataset(
    df: pd.DataFrame,
    path: Path,
    result: CurrentSeasonCollectionResult,
    key: str,
) -> None:
    if df.empty:
        return
    df.to_parquet(path, index=False)
    result.files[key] = path


def collect_current_season(
    temporada: int = CURRENT_SEASON,
    rodadas: list[int] | None = None,
    token: str | None = None,
    ligas: list[str] | None = None,
) -> CurrentSeasonCollectionResult:
    """Collect live Cartola/Gato Mestre data and persist project datasets."""
    api = CartolaAPI(token=token)
    current_round = get_current_round(api)
    target_rounds = rodadas or list(range(1, current_round))
    result = CurrentSeasonCollectionResult(season=temporada, rounds=target_rounds)

    if not target_rounds:
        log.error("Nenhuma rodada para coletar.")
        return result

    players_file = DATA_DIR / f"jogadores_por_rodada_{temporada}.parquet"
    if players_file.exists():
        players = pd.read_parquet(players_file)
        rounds_to_collect = missing_rounds(players, target_rounds)
        if rounds_to_collect:
            log.info("Coletando %s rodadas novas.", len(rounds_to_collect))
            new_players = get_players_data(api, rounds_to_collect, temporada=temporada)
            players = pd.concat([players, new_players], ignore_index=True)
        else:
            log.info("Jogadores por rodada ja estao atualizados.")
    else:
        players = get_players_data(api, target_rounds, temporada=temporada)

    if not players.empty:
        players = deduplicate_by_key(
            players,
            ["temporada", "rodada", "atleta_id"],
            prefer_played=True,
        )
                
        _save_yearly_dataset(players, players_file, result, "jogadores_por_rodada")
        log.info(
            "Jogadores por rodada salvos: %s rodadas, %s atletas -> %s",
            players["rodada"].nunique(),
            players["atleta_id"].nunique(),
            players_file,
        )

    matches = preparar_partidas(api, target_rounds, temporada=temporada)
    _save_yearly_dataset(matches, DATA_DIR / f"partidas_{temporada}.parquet", result, "partidas")

    market = get_current_market(api)
    if not market.empty:
        market_file = DATA_DIR / "mercado_atual.parquet"
        market.to_parquet(market_file, index=False)
        result.files["mercado_atual"] = market_file
        log.info("Mercado atual salvo: %s atletas -> %s", len(market), market_file)

    if token:
        gato_api = GatoMestreAPI(token=token, temporada=temporada)
        odds_rounds = sorted(set(target_rounds + [current_round]))
        odds = get_odds(gato_api, api.clubes(), odds_rounds, temporada=temporada)
        _save_yearly_dataset(odds, DATA_DIR / f"odds_{temporada}.parquet", result, "odds")
    else:
        log.info("CARTOLA_TOKEN nao definido - odds nao coletadas.")

    user_means = get_cartola_users_mean(api, target_rounds)
    if not user_means.empty:
        means_file = DATA_DIR / "medias_cartoleiros.parquet"
        user_means.to_parquet(means_file, index=False)
        result.files["medias_cartoleiros"] = means_file

    league_brackets = get_league_brackets(api, ligas or DEFAULT_LIGAS)
    if not league_brackets.empty:
        brackets_file = DATA_DIR / "chaves_ligas.parquet"
        league_brackets.to_parquet(brackets_file, index=False)
        result.files["chaves_ligas"] = brackets_file

    return result
