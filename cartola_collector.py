"""
Cartola FC - Data Collector
============================
Coleta e persiste dados históricos da API do Cartola FC:
  - Pontuações por atleta/rodada  → data/historico.parquet
  - Mando de campo por rodada     → data/partidas.parquet

Uso:
    python cartola_collector.py --token SEU_TOKEN
    python cartola_collector.py --token SEU_TOKEN --rodadas 1 5 10
"""

import os
import re
import time
import json
import logging
import argparse
import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# CONFIGURAÇÕES COMPARTILHADAS
# ──────────────────────────────────────────────

BASE_URL = "https://api.cartola.globo.com"
GATOMESTRE_BASE = "https://api.gatomestre.globo.com"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

FORMATION = {
    1: 1,  # GOL
    2: 2,  # LAT
    3: 2,  # ZAG
    4: 3,  # MEI
    5: 3,  # ATA
    6: 1,  # TEC
}

TEMPORADA_ATUAL = 2026

BUDGET = 140.0

SCOUT_POINTS = {
    "G": 8.0, "A": 5.0, "FT": 3.5, "FD": 1.2, "FF": 0.8,
    "FS": 0.5, "PE": -0.3, "I": -0.1, "FC": -0.3, "GC": -3.0,
    "CV": -3.0, "CA": -1.0, "SG": 5.0, "DD": 3.0, "GS": -1.0,
    "DS": 1.2, "PP": -4.0, "DP": 7.0, "PC": 0.3, "RB": 1.5,
}

POSICAO_NOME = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}


def safe_filename(name) -> str:
    return re.sub(r'[<>:"/\\|?*]', '_', name)


# ──────────────────────────────────────────────
# API WRAPPER
# ──────────────────────────────────────────────

class CartolaAPI:
    """Wrapper para a API do Cartola FC com retry e cache em disco."""

    def __init__(self, token: str = None, cache_dir: Path = DATA_DIR / "cache"):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if token:
            self.session.headers["Authorization"] = token
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, path: str, use_cache: bool = True) -> dict:
        cache_file = self.cache_dir / safe_filename(path + ".json")
        if use_cache and cache_file.exists():
            return json.loads(cache_file.read_text())

        url = f"{BASE_URL}{path}"
        for attempt in range(3):
            try:
                r = self.session.get(url, timeout=15)
                r.raise_for_status()
                data = r.json()
                if use_cache:
                    cache_file.write_text(json.dumps(data))
                return data
            except requests.RequestException as e:
                log.warning(f"Tentativa {attempt+1} falhou para {path}: {e}")
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Falha ao acessar {path} após 3 tentativas")

    def market_status(self):
        return self._get("/mercado/status", use_cache=False)

    def atletas_mercado(self):
        return self._get("/atletas/mercado", use_cache=False)

    def atletas_pontuados(self, rodada: int = None):
        path = f"/atletas/pontuados/{rodada}" if rodada else "/atletas/pontuados"
        return self._get(path)

    def clubes(self):
        return self._get("/clubes")

    def rodadas(self):
        return self._get("/rodadas")

    def partidas(self, rodada: int = None):
        path = f"/partidas/{rodada}" if rodada else "/partidas"
        return self._get(path, use_cache=rodada is not None)
    
    def pos_rodada(self, rodada: int):
        path = f"/pos-rodada/destaques/{rodada}"
        return self._get(path, use_cache=rodada)
    
    def league(self, liga: int) -> dict:
        path = f"/auth/liga/{liga}?orderBy=rodada"
        return self._get(path, use_cache=True)


class GatoMestreAPI:
    """
    Wrapper para o endpoint de favoritos do Gato Mestre, que devolve as
    probabilidades (vitória / empate / derrota) por partida da rodada.

    Auth: requer Bearer token em $GATOMESTRE_TOKEN (token JWT do globo.com).
    A URL hardcoda o ano corrente — ajustar `temporada` ao virar a temporada.
    """

    def __init__(
        self,
        token: str,
        temporada: int = 2026,
        cache_dir: Path = DATA_DIR / "cache",
    ):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": token,
        })
        self.temporada = temporada
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def favoritos(self, rodada: int) -> dict:
        """
        Retorna odds da rodada `rodada`. O endpoint Gato Mestre indexa em
        rodada-1, ou seja, /rodadas/{rodada-1} retorna a rodada `rodada`.
        """
        index = rodada - 1
        cache_file = self.cache_dir / f"gato_favoritos_{self.temporada}_r{rodada}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        url = (
            f"{GATOMESTRE_BASE}/api/v2/equipes/{self.temporada}"
            f"/campeonato-brasileiro/favoritos/rodadas/{index}?a=true"
        )
        for attempt in range(3):
            try:
                r = self.session.get(url, timeout=15)
                r.raise_for_status()
                data = r.json()
                cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                return data
            except requests.RequestException as e:
                log.warning(f"Tentativa {attempt+1} falhou para favoritos r{rodada}: {e}")
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Falha ao acessar favoritos da rodada {rodada}")


# ──────────────────────────────────────────────
# COLETA
# ──────────────────────────────────────────────

def get_players_data(api: CartolaAPI, rodadas_alvo: list[int]) -> pd.DataFrame:
    """
    Get players historical data per round.
    Returns a DataFrame with an row per (player, round).
    """
    registros = []
    clubes_raw = api.clubes()
    clubes = {int(k): v["nome"] for k, v in clubes_raw.items()}

    for rodada in tqdm(rodadas_alvo, desc="Coletando rodadas"):
        try:
            data = api.atletas_pontuados(rodada)
        except Exception as e:
            log.warning(f"Rodada {rodada} indisponível: {e}")
            continue

        atletas = data.get("atletas", {})
        # /atletas/pontuados keys atletas by id in a dict but doesn't repeat
        # the id inside the record. Preserve the dict key as atleta_id.
        if isinstance(atletas, dict):
            atletas_iter = [(int(k), v) for k, v in atletas.items()]
        else:
            atletas_iter = [(a.get("atleta_id"), a) for a in atletas]

        for atleta_id, a in atletas_iter:
            scout = a.get("scout", {}) or {}
            registro = {
                "rodada":     rodada,
                "atleta_id":  atleta_id,
                "apelido":    a.get("apelido"),
                "posicao_id": a.get("posicao_id"),
                "clube_id":   a.get("clube_id"),
                "clube_nome": clubes.get(a.get("clube_id"), ""),
                "status_id":  a.get("status_id"),
                "pontos":     a.get("pontuacao", a.get("pontos_num", 0.0)),
                "preco":      a.get("preco_num", 0.0),
                "media":      a.get("media_num", 0.0),
                "jogos":      a.get("jogos_num", 0),
                "jogou":      a.get("entrou_em_campo", False),
                **{f"scout_{k}": v for k, v in scout.items()},
            }
            registros.append(registro)
        time.sleep(0.3)

    df = pd.DataFrame(registros)
    return df.sort_values(["atleta_id", "rodada"]).reset_index(drop=True)


def get_current_market(api: CartolaAPI) -> pd.DataFrame:
    """
    Snapshot do mercado vigente: preco, media, status, jogos por atleta.
    /atletas/pontuados não traz esses campos, então eles vêm daqui.
    """
    data = api.atletas_mercado()
    atletas = data.get("atletas", [])
    rows = [{
        "atleta_id":   a.get("atleta_id"),
        "apelido":     a.get("apelido"),
        "posicao_id":  a.get("posicao_id"),
        "clube_id":    a.get("clube_id"),
        "preco":       a.get("preco_num", 0.0),
        "media":       a.get("media_num", 0.0),
        "status_id":   a.get("status_id"),
        "jogos":       a.get("jogos_num", 0),
    } for a in atletas]
    return pd.DataFrame(rows)


def get_odds(
    gato_api: GatoMestreAPI,
    clubes_raw: dict,
    rodadas_alvo: list[int],
) -> pd.DataFrame:
    """
    Coleta probabilidades do Gato Mestre para cada rodada. Retorna DataFrame
    com uma linha POR TIME por partida (duas por jogo): home e away com as
    probs reorientadas para a perspectiva de cada lado.

    Colunas: rodada, clube_id, prob_win, prob_draw, prob_loss.
    """
    abbr_to_id = {}
    for cid, c in clubes_raw.items():
        abbr = (c.get("abreviacao") or "").upper()
        if abbr:
            abbr_to_id[abbr] = int(cid)

    rows = []
    for rodada in tqdm(rodadas_alvo, desc="Coletando odds"):
        # Gato Mestre não publica odds da rodada 1 — começa a partir da 2.
        if rodada < 2:
            continue
        try:
            data = gato_api.favoritos(rodada)
        except Exception as e:
            log.warning(f"Odds rodada {rodada} indisponíveis: {e}")
            continue

        for match in data.get("result", []) or []:
            home_abbr = (match.get("homeTeam", {}).get("abbr") or "").upper()
            away_abbr = (match.get("awayTeam", {}).get("abbr") or "").upper()
            home_id = abbr_to_id.get(home_abbr)
            away_id = abbr_to_id.get(away_abbr)
            if home_id is None or away_id is None:
                log.warning(
                    f"Rodada {rodada}: time sem mapeamento ({home_abbr} ou {away_abbr})"
                )
                continue
            odds = match.get("odds", {}) or {}
            win  = float(odds.get("win",  0)) / 100.0
            tie  = float(odds.get("tie",  0)) / 100.0
            loss = float(odds.get("loss", 0)) / 100.0
            rows.append({
                "rodada": rodada, "clube_id": home_id,
                "prob_win": win, "prob_draw": tie, "prob_loss": loss,
            })
            rows.append({
                "rodada": rodada, "clube_id": away_id,
                "prob_win": loss, "prob_draw": tie, "prob_loss": win,
            })
        time.sleep(0.3)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def carregar_dataset(
    temporadas_extras: tuple[int, ...] = (2022, 2023, 2024, 2025),
    temporada_atual: int = TEMPORADA_ATUAL,
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega histórico, partidas e odds da temporada atual e das passadas.
    Toda linha sai com coluna `temporada` preenchida (a temporada atual recebe
    a constante quando o parquet ainda não tem a coluna).

    Retorna (df_hist, df_partidas, df_odds), todos podem ser DataFrame vazio.
    """
    def _load(path: Path, temporada: int) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if not df.empty and "temporada" not in df.columns:
            df["temporada"] = temporada
        return df

    df_hist     = _load(data_dir / "historico.parquet",      temporada_atual)
    df_partidas = _load(data_dir / "partidas.parquet",       temporada_atual)
    df_odds     = _load(data_dir / "odds.parquet",           temporada_atual)

    for year in temporadas_extras:
        df_hist     = pd.concat([_load(data_dir / f"historico_{year}.parquet", year), df_hist], ignore_index=True)
        df_partidas = pd.concat([_load(data_dir / f"partidas_{year}.parquet",  year), df_partidas], ignore_index=True)
        df_odds     = pd.concat([_load(data_dir / f"odds_{year}.parquet",      year), df_odds], ignore_index=True)

    return df_hist, df_partidas, df_odds


def preparar_partidas(api: CartolaAPI, rodadas_alvo: list[int]) -> pd.DataFrame:
    """Extrai mando de campo (casa=1, fora=-1) e adversário por (rodada, clube_id)."""
    registros = []
    for rodada in tqdm(rodadas_alvo, desc="Coletando partidas"):
        try:
            data = api.partidas(rodada)
            for partida in data.get("partidas", []):
                casa = partida.get("clube_casa_id")
                fora = partida.get("clube_visitante_id")
                registros.append({"rodada": rodada, "clube_id": casa, "mando":  1, "clube_adversario_id": fora})
                registros.append({"rodada": rodada, "clube_id": fora, "mando": -1, "clube_adversario_id": casa})
        except Exception as e:
            log.warning(f"Partidas rodada {rodada}: {e}")
        time.sleep(0.2)

    return pd.DataFrame(registros) if registros else pd.DataFrame()


def get_cartola_users_mean(api: CartolaAPI, rodadas_alvo: list[int]) -> pd.DataFrame:
    medias_cartoleiros = []
    for rodada in rodadas_alvo:
        data = api.pos_rodada(rodada)
        media = data.get("media_pontos")
        medias_cartoleiros.append({"rodada": rodada, "media_cartoleiros": media})

    return  pd.DataFrame(medias_cartoleiros) if medias_cartoleiros else pd.DataFrame()

def get_league_brackets(api: CartolaAPI, ligas: str) -> pd.DataFrame:
    chaves_ligas = []
    for liga in ligas:
        chaves_mata_mata: dict = api.league(liga)["chaves_mata_mata"]
        for rodada, chaves in chaves_mata_mata.items():
            for chave in chaves:
                if chave["vencedor_id"] is None:
                    break

                pontos_vencedor = max(chave["time_mandante_pontuacao"], chave["time_visitante_pontuacao"])
                chaves_ligas.append({"rodada": rodada, "pontos": pontos_vencedor})

    return pd.DataFrame(chaves_ligas) if chaves_ligas else pd.DataFrame()


# ──────────────────────────────────────────────
# EXECUÇÃO
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cartola FC — Data Collector")
    parser.add_argument("--rodadas", type=int, nargs="+",   default=None,
                        help="Rodadas a coletar (padrão: todas até a rodada atual)")
    parser.add_argument("--temporada", type=int, nargs="+",   default=TEMPORADA_ATUAL,
                        help="Temporada a coletar (padrão: temporada atual)")
    args = parser.parse_args()
    token = os.environ.get("CARTOLA_TOKEN")    

    api = CartolaAPI(token=token)

    status = api.market_status()
    current_round = status.get("rodada_atual", status.get("rodada", {}).get("rodada_atual", 1))
    log.info(f"Rodada atual: {current_round}")

    target_rounds = args.rodadas or list(range(1, current_round))
    if not target_rounds:
        log.error("Nenhuma rodada para coletar (temporada ainda na rodada 1).")
        return

    # ── Players data per round ──
    players_per_round_file = DATA_DIR / "jogadores_por_rodada.parquet"
    if players_per_round_file.exists():
        df_players_per_round = pd.read_parquet(players_per_round_file)
        missing_rounds = [r for r in target_rounds if r not in df_players_per_round["rodada"].unique()]
        if missing_rounds:
            log.info(f"Coletando {len(missing_rounds)} rodadas novas...")
            new_df = get_players_data(api, missing_rounds)
            df_players_per_round = pd.concat([df_players_per_round, new_df], ignore_index=True)
        else:
            log.info("Dados de jogadores por rodada já atualizado, nada a coletar.")
    else:
        df_players_per_round = get_players_data(api, target_rounds)

    df_players_per_round.to_parquet(players_per_round_file, index=False)
    log.info(f"Dados de jogadores por rodada salvo: {df_players_per_round['rodada'].nunique()} rodadas, "
             f"{df_players_per_round['atleta_id'].nunique()} atletas → {players_per_round_file}")

    # ── Current market (price, mean, status_id, matches) ──
    df_current_market = get_current_market(api)
    if not df_current_market.empty:
        market_file = DATA_DIR / "mercado_atual.parquet"
        df_current_market.to_parquet(market_file, index=False)
        log.info(f"Mercado atual salvo: {len(df_current_market)} atletas → {market_file}")

    # ── Odds (Gato Mestre) ──
    if token:
        gato_api = GatoMestreAPI(token=token, temporada=args.temporada)
        # Inclui a rodada_atual (próxima a ser jogada) para uso na inferência ao vivo
        rodadas_para_odds = target_rounds + [current_round]
        df_odds = get_odds(gato_api, api.clubes(), sorted(set(rodadas_para_odds)))
        if not df_odds.empty:
            odds_file = DATA_DIR / "odds.parquet"
            df_odds.to_parquet(odds_file, index=False)
            log.info(f"Odds salvas: {df_odds['rodada'].nunique()} rodadas → {odds_file}")
    else:
        log.info("GATOMESTRE_TOKEN não definido — odds não coletadas.")

    # ──Cartola users average ──
    df_medias_cartoleiros = get_cartola_users_mean(api, target_rounds)
    if not df_medias_cartoleiros.empty:
        medias_cartoleiros_file = DATA_DIR / "medias_cartoleiros.parquet"
        df_medias_cartoleiros.to_parquet(medias_cartoleiros_file, index=False)
        log.info(f"Medias dos cartoleiros salvas: {df_medias_cartoleiros['rodada'].nunique()} rodadas → {medias_cartoleiros_file}")

    # ── League brackets ──
    LIGAS = ["1-mata-mata-brothers-do-graia-2026", "2o-mata-mata-brothers-do-graia"]
    df_chaves_ligas = get_league_brackets(api, LIGAS)
    if not df_chaves_ligas.empty:
        chaves_ligas_file = DATA_DIR / "chaves_ligas.parquet"
        df_chaves_ligas.to_parquet(chaves_ligas_file, index=False)
        log.info(f"Chaves das ligas salvas: {df_chaves_ligas['rodada'].nunique()} rodadas e {len(df_chaves_ligas)} chaves → {chaves_ligas_file}")


if __name__ == "__main__":
    main()
