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

import time
import json
import logging
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# CONFIGURAÇÕES COMPARTILHADAS
# ──────────────────────────────────────────────

BASE_URL = "https://api.cartola.globo.com"
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

BUDGET = 140.0

SCOUT_POINTS = {
    "G": 8.0, "A": 5.0, "FT": 3.5, "FD": 1.2, "FF": 0.8,
    "FS": 0.5, "PE": -0.3, "I": -0.1, "FC": -0.3, "GC": -3.0,
    "CV": -3.0, "CA": -1.0, "SG": 5.0, "DD": 3.0, "GS": -1.0,
    "DS": 1.2, "PP": -4.0, "DP": 7.0, "PC": 0.3, "RB": 1.5,
}

POSICAO_NOME = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}


# ──────────────────────────────────────────────
# API WRAPPER
# ──────────────────────────────────────────────

class CartolaAPI:
    """Wrapper para a API do Cartola FC com retry e cache em disco."""

    def __init__(self, token: str = None, cache_dir: Path = DATA_DIR / "cache"):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if token:
            self.session.headers["X-GLB-Token"] = token
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, path: str, use_cache: bool = True) -> dict:
        cache_file = self.cache_dir / (path.strip("/").replace("/", "_") + ".json")
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

    def mercado_status(self):
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


# ──────────────────────────────────────────────
# COLETA
# ──────────────────────────────────────────────

def coletar_historico(api: CartolaAPI, rodadas_alvo: list[int]) -> pd.DataFrame:
    """
    Coleta pontuações históricas rodada a rodada.
    Retorna DataFrame com uma linha por (atleta, rodada).
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
        if isinstance(atletas, dict):
            atletas = list(atletas.values())

        for a in atletas:
            scout = a.get("scout", {}) or {}
            registro = {
                "rodada":     rodada,
                "atleta_id":  a.get("atleta_id"),
                "apelido":    a.get("apelido"),
                "posicao_id": a.get("posicao_id"),
                "clube_id":   a.get("clube_id"),
                "clube_nome": clubes.get(a.get("clube_id"), ""),
                "status_id":  a.get("status_id"),
                "pontos":     a.get("pontuacao", a.get("pontos_num", 0.0)),
                "preco":      a.get("preco_num", 0.0),
                "media":      a.get("media_num", 0.0),
                "jogos":      a.get("jogos_num", 0),
                **{f"scout_{k}": v for k, v in scout.items()},
            }
            registros.append(registro)
        time.sleep(0.3)

    df = pd.DataFrame(registros)
    return df.sort_values(["atleta_id", "rodada"]).reset_index(drop=True)


def coletar_mercado_atual(api: CartolaAPI) -> pd.DataFrame:
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


def preparar_partidas(api: CartolaAPI, rodadas_alvo: list[int]) -> pd.DataFrame:
    """Extrai mando de campo (casa=1, fora=-1) por (rodada, clube_id)."""
    registros = []
    for rodada in tqdm(rodadas_alvo, desc="Coletando partidas"):
        try:
            data = api.partidas(rodada)
            for partida in data.get("partidas", []):
                registros.append({"rodada": rodada, "clube_id": partida.get("clube_casa_id"),      "mando":  1})
                registros.append({"rodada": rodada, "clube_id": partida.get("clube_visitante_id"), "mando": -1})
        except Exception as e:
            log.warning(f"Partidas rodada {rodada}: {e}")
        time.sleep(0.2)

    return pd.DataFrame(registros) if registros else pd.DataFrame()


# ──────────────────────────────────────────────
# EXECUÇÃO
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cartola FC — Data Collector")
    parser.add_argument("--token",   type=str, default=None, help="X-GLB-Token")
    parser.add_argument("--rodadas", type=int, nargs="+",   default=None,
                        help="Rodadas a coletar (padrão: todas até a rodada atual)")
    args = parser.parse_args()

    api = CartolaAPI(token=args.token)

    status = api.mercado_status()
    rodada_atual = status.get("rodada_atual", status.get("rodada", {}).get("rodada_atual", 1))
    log.info(f"Rodada atual: {rodada_atual}")

    rodadas_alvo = args.rodadas or list(range(1, rodada_atual))
    if not rodadas_alvo:
        log.error("Nenhuma rodada para coletar (temporada ainda na rodada 1).")
        return

    # ── Histórico ──
    hist_file = DATA_DIR / "historico.parquet"
    if hist_file.exists():
        df_hist = pd.read_parquet(hist_file)
        rodadas_faltando = [r for r in rodadas_alvo if r not in df_hist["rodada"].unique()]
        if rodadas_faltando:
            log.info(f"Coletando {len(rodadas_faltando)} rodadas novas...")
            df_novo = coletar_historico(api, rodadas_faltando)
            df_hist = pd.concat([df_hist, df_novo], ignore_index=True)
        else:
            log.info("Histórico já atualizado, nada a coletar.")
    else:
        df_hist = coletar_historico(api, rodadas_alvo)

    df_hist.to_parquet(hist_file, index=False)
    log.info(f"Histórico salvo: {df_hist['rodada'].nunique()} rodadas, "
             f"{df_hist['atleta_id'].nunique()} atletas → {hist_file}")

    # ── Partidas ──
    df_partidas = preparar_partidas(api, rodadas_alvo)
    if not df_partidas.empty:
        partidas_file = DATA_DIR / "partidas.parquet"
        df_partidas.to_parquet(partidas_file, index=False)
        log.info(f"Partidas salvas → {partidas_file}")


if __name__ == "__main__":
    main()
