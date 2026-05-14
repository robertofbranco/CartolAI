"""
Importador de temporadas anteriores
====================================
Baixa pontuações + scouts do repositório caRtola (henriquepgomide/caRtola)
e partidas + odds do endpoint Gato Mestre (com a temporada parametrizada),
salvando como parquets paralelos aos da temporada corrente:

    data/historico_{YEAR}.parquet
    data/partidas_{YEAR}.parquet
    data/odds_{YEAR}.parquet

Uso:
    # Token do Gato Mestre (renovar via globo.com — JWT é curto)
    export GATOMESTRE_TOKEN="Bearer eyJ..."
    python import_historic_data.py --year 2025
"""

import argparse
import logging
import os
from io import StringIO

import pandas as pd
import requests
from tqdm import tqdm

from cartola_collector import CartolaAPI, DATA_DIR, GatoMestreAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


CARTOLA_REPO_BASE = (
    "https://raw.githubusercontent.com/henriquepgomide/caRtola/master"
    "/data/01_raw/{year}/rodada-{rodada}.csv"
)

# caRtola CSV → schema do projeto.
CARTOLA_RENAME = {
    "atletas.rodada_id":            "rodada",
    "atletas.atleta_id":            "atleta_id",
    "atletas.apelido":              "apelido",
    "atletas.posicao_id":           "posicao_id",
    "atletas.clube_id":             "clube_id",
    "atletas.clube.id.full.name":   "clube_abreviacao",
    "atletas.status_id":            "status_id",
    "atletas.pontos_num":           "pontos",
    "atletas.preco_num":            "preco",
    "atletas.media_num":            "media",
    "atletas.jogos_num":            "jogos",
}

# Colunas de scout reconhecidas (subset que tem ponto no SCOUT_POINTS).
RECOGNIZED_SCOUTS = {
    "CA", "FC", "FF", "G", "I", "DS", "FS", "FD", "GS", "A",
    "FT", "CV", "DP", "SG", "PC", "PP", "GC",
}

# Nomes longos do CSV caRtola (2022–2024) → abbr de 3 letras usado pelo
# Gato Mestre. A partir de 2025 o caRtola já entrega códigos curtos, então
# este mapa só importa para temporadas antigas.
NOME_TO_ABBR = {
    "Flamengo": "FLA",
    "Botafogo": "BOT",
    "Corinthians": "COR",
    "Bahia": "BAH",
    "Fluminense": "FLU",
    "Vasco": "VAS",
    "Palmeiras": "PAL",
    "São Paulo": "SAO",
    "Santos": "SAN",
    "Bragantino": "RBB",
    "Atlético-MG": "CAM",
    "Cruzeiro": "CRU",
    "Grêmio": "GRE",
    "Internacional": "INT",
    "Juventude": "JUV",
    "Vitória": "VIT",
    "Criciúma": "CRI",
    "Goiás": "GOI",
    "Athlético-PR": "CAP",
    "Coritiba": "CFC",
    "América-MG": "AME",
    "Fortaleza": "FOR",
    "Atlético-GO": "ACG",
    "Cuiabá": "CUI",
    "Avaí": "AVA",
    "Ceará": "CEA",
}


def baixar_rodada(year: int, rodada: int) -> pd.DataFrame:
    url = CARTOLA_REPO_BASE.format(year=year, rodada=rodada)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def normalizar_rodada(df_raw: pd.DataFrame, year: int) -> pd.DataFrame:
    cols_existentes = {k: v for k, v in CARTOLA_RENAME.items() if k in df_raw.columns}
    out = df_raw[list(cols_existentes)].rename(columns=cols_existentes).copy()
    out["temporada"] = year

    for col in df_raw.columns:
        if col in RECOGNIZED_SCOUTS:
            out[f"scout_{col}"] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0)

    out["rodada"]     = pd.to_numeric(out["rodada"],     errors="coerce").astype("Int64")
    out["atleta_id"]  = pd.to_numeric(out["atleta_id"],  errors="coerce").astype("Int64")
    out["clube_id"]   = pd.to_numeric(out["clube_id"],   errors="coerce").astype("Int64")
    out["posicao_id"] = pd.to_numeric(out["posicao_id"], errors="coerce").astype("Int64")
    out["status_id"]  = pd.to_numeric(out["status_id"],  errors="coerce").astype("Int64")
    out["jogos"]      = pd.to_numeric(out["jogos"],      errors="coerce").astype("Int64")
    out["pontos"]     = pd.to_numeric(out["pontos"],     errors="coerce").astype(float)
    out["preco"]      = pd.to_numeric(out["preco"],      errors="coerce").astype(float)
    out["media"]      = pd.to_numeric(out["media"],      errors="coerce").astype(float)
    return out.dropna(subset=["atleta_id", "rodada", "clube_id"])


def importar_historico(year: int, rodadas: list[int]) -> pd.DataFrame:
    frames = []
    for rodada in tqdm(rodadas, desc=f"caRtola {year}"):
        try:
            raw = baixar_rodada(year, rodada)
            frames.append(normalizar_rodada(raw, year))
        except Exception as e:
            log.warning(f"Falha rodada {year}/{rodada}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["clube_nome"] = df["clube_abreviacao"]  # caRtola não traz nome longo
    return df.sort_values(["atleta_id", "rodada"]).reset_index(drop=True)


def importar_partidas_odds(
    gato_api: GatoMestreAPI,
    abbr_to_clube_id: dict[str, int],
    rodadas: list[int],
    temporada: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uma única chamada por rodada ao Gato Mestre devolve home/away + odds.
    Construímos partidas e odds em paralelo a partir do mesmo payload.
    """
    partidas_rows, odds_rows = [], []
    for rodada in tqdm(rodadas, desc=f"Gato Mestre {temporada}"):
        if rodada < 2:
            continue  # Gato Mestre não publica rodada 1
        try:
            data = gato_api.favoritos(rodada)
        except Exception as e:
            log.warning(f"Rodada {rodada}: {e}")
            continue

        for match in data.get("result", []) or []:
            home_abbr = (match.get("homeTeam", {}).get("abbr") or "").upper()
            away_abbr = (match.get("awayTeam", {}).get("abbr") or "").upper()
            home_id = abbr_to_clube_id.get(home_abbr)
            away_id = abbr_to_clube_id.get(away_abbr)
            if home_id is None or away_id is None:
                log.warning(f"Sem mapeamento para {home_abbr}/{away_abbr} (r{rodada})")
                continue

            partidas_rows.append({
                "rodada": rodada, "clube_id": home_id,
                "mando":  1, "clube_adversario_id": away_id,
            })
            partidas_rows.append({
                "rodada": rodada, "clube_id": away_id,
                "mando": -1, "clube_adversario_id": home_id,
            })

            odds = match.get("odds", {}) or {}
            win  = float(odds.get("win",  0)) / 100.0
            tie  = float(odds.get("tie",  0)) / 100.0
            loss = float(odds.get("loss", 0)) / 100.0
            odds_rows.append({
                "rodada": rodada, "clube_id": home_id,
                "prob_win": win, "prob_draw": tie, "prob_loss": loss,
            })
            odds_rows.append({
                "rodada": rodada, "clube_id": away_id,
                "prob_win": loss, "prob_draw": tie, "prob_loss": win,
            })

    df_partidas = pd.DataFrame(partidas_rows) if partidas_rows else pd.DataFrame()
    df_odds     = pd.DataFrame(odds_rows)     if odds_rows     else pd.DataFrame()
    if not df_partidas.empty:
        df_partidas["temporada"] = temporada
    if not df_odds.empty:
        df_odds["temporada"] = temporada
    return df_partidas, df_odds


def build_abbr_to_clube_id(data_dir, year: int) -> dict[str, int]:
    """
    Constrói abbr (3 letras, ex: 'FLA') → clube_id válido para a temporada
    `year`. Necessário porque o caRtola só padronizou os códigos curtos a
    partir de 2025, e o Gato Mestre só entrega o `abbr` de 3 letras.

    Prioridade (mais específico primeiro, para evitar colisões — /clubes
    devolve ~60 clubes históricos e abbrs como "AME" colidem entre clubes):
    1. historico_{year}.parquet (a temporada sendo importada).
    2. Outros historico_*.parquet existentes.
    3. /clubes da Cartola API (clubes da Série A atual).
    """
    abbr_to_id: dict[str, int] = {}

    def _ingest_parquet(parquet) -> None:
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
            elif raw in NOME_TO_ABBR:
                abbr_to_id.setdefault(NOME_TO_ABBR[raw], cid)

    primary = data_dir / f"historico_{year}.parquet"
    if primary.exists():
        _ingest_parquet(primary)

    for parquet in sorted(data_dir.glob("historico_*.parquet")):
        if parquet != primary:
            _ingest_parquet(parquet)

    try:
        for cid, c in (CartolaAPI().clubes() or {}).items():
            abbr = (c.get("abreviacao") or "").upper()
            if abbr:
                abbr_to_id.setdefault(abbr, int(cid))
    except Exception as e:
        log.warning(f"Falha ao buscar /clubes da Cartola API: {e}")

    return abbr_to_id


def main():
    parser = argparse.ArgumentParser(description="Importa dados de temporada anterior")
    parser.add_argument("--year",        type=int, default=2025)
    parser.add_argument("--gato-token",  type=str, default=None,
                        help="Bearer do Gato Mestre. Default: $GATOMESTRE_TOKEN")
    parser.add_argument("--rodadas",     type=int, nargs="+", default=None,
                        help="Subset de rodadas (default: 1..38)")    
    args = parser.parse_args()

    rodadas = args.rodadas or list(range(1, 39))
    gato_token = args.gato_token or os.environ.get("GATOMESTRE_TOKEN")

    log.info(f"Importando histórico de {args.year} do caRtola...")
    df_hist = importar_historico(args.year, rodadas)
    if df_hist.empty:
        log.error("Nenhum dado coletado.")
        return

    hist_path = DATA_DIR / f"historico_{args.year}.parquet"
    df_hist.to_parquet(hist_path, index=False)
    log.info(
        f"Histórico {args.year} salvo: {df_hist['rodada'].nunique()} rodadas, "
        f"{df_hist['atleta_id'].nunique()} atletas → {hist_path}"
    )

    abbr_to_id = build_abbr_to_clube_id(DATA_DIR, args.year)
    log.info(f"Mapa abbr→clube_id construído com {len(abbr_to_id)} clubes")

    if not gato_token:
        log.warning(
            "GATOMESTRE_TOKEN não definido — partidas e odds de "
            f"{args.year} não foram coletadas. Defina o token e rode novamente."
        )
        return

    gato_api = GatoMestreAPI(token=gato_token, temporada=args.year)
    df_partidas, df_odds = importar_partidas_odds(gato_api, abbr_to_id, rodadas, args.year)

    if not df_partidas.empty:
        partidas_path = DATA_DIR / f"partidas_{args.year}.parquet"
        df_partidas.to_parquet(partidas_path, index=False)
        log.info(f"Partidas {args.year} salvas → {partidas_path}")

    if not df_odds.empty:
        odds_path = DATA_DIR / f"odds_{args.year}.parquet"
        df_odds.to_parquet(odds_path, index=False)
        log.info(f"Odds {args.year} salvas → {odds_path}")


if __name__ == "__main__":
    main()