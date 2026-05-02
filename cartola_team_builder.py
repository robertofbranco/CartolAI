"""
Cartola FC - ML Team Builder
============================
Pipeline completo para:
  1. Coletar dados históricos da API do Cartola
  2. Engenharia de features (scouts, forma, casa/fora, etc.)
  3. Treinar modelo de predição de pontos por jogador
  4. Otimizar escalação via programação linear (maximizar pontos esperados)

Requisitos:
    pip install requests pandas numpy scikit-learn lightgbm pulp tqdm
"""

import os
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# ML
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# Otimização
import pulp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────

BASE_URL = "https://api.cartola.globo.com"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Formação padrão do Cartola (4-3-3 = esquema_id 3)
# 1 GOL + 2 LAT + 2 ZAG + 3 MEI + 3 ATA + 1 TEC = 12 jogadores
FORMATION = {
    1: 1,  # GOL
    2: 2,  # LAT
    3: 2,  # ZAG
    4: 3,  # MEI
    5: 3,  # ATA
    6: 1,  # TEC
}

BUDGET = 140.0  # cartoletas padrão

# Tabela de pontuação dos scouts
SCOUT_POINTS = {
    "G": 8.0, "A": 5.0, "FT": 3.5, "FD": 1.2, "FF": 0.8,
    "FS": 0.5, "PE": -0.3, "I": -0.1, "FC": -0.3, "GC": -3.0,
    "CV": -3.0, "CA": -1.0, "SG": 5.0, "DD": 3.0, "GS": -1.0,
    "DS": 1.2, "PP": -4.0, "DP": 7.0, "PC": 0.3, "RB": 1.5,
    "DP": 7.0,
}

POSICAO_NOME = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}

# ──────────────────────────────────────────────
# 1. COLETA DE DADOS
# ──────────────────────────────────────────────

class CartolaAPI:
    """Wrapper simples para a API do Cartola FC com retry e cache em disco."""

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
            # Calcula pontuação pelos scouts (verificação)
            pontos_scout = sum(SCOUT_POINTS.get(k, 0) * v for k, v in scout.items())
            registro = {
                "rodada": rodada,
                "atleta_id": a.get("atleta_id"),
                "apelido": a.get("apelido"),
                "posicao_id": a.get("posicao_id"),
                "clube_id": a.get("clube_id"),
                "clube_nome": clubes.get(a.get("clube_id"), ""),
                "status_id": a.get("status_id"),
                "pontos": a.get("pontuacao", a.get("pontos_num", 0.0)),
                "preco": a.get("preco_num", 0.0),
                "media": a.get("media_num", 0.0),
                "jogos": a.get("jogos_num", 0),
                **{f"scout_{k}": v for k, v in scout.items()},
            }
            registros.append(registro)
        time.sleep(0.3)  # respeitar rate limit

    df = pd.DataFrame(registros)
    df = df.sort_values(["atleta_id", "rodada"]).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# 2. ENGENHARIA DE FEATURES
# ──────────────────────────────────────────────

def construir_features(df: pd.DataFrame, partidas_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Gera features temporais e contextuais para cada (atleta, rodada).
    IMPORTANTE: Usa apenas dados do PASSADO (sem data leakage).
    """
    scout_cols = [c for c in df.columns if c.startswith("scout_")]
    df[scout_cols] = df[scout_cols].fillna(0)

    df = df.sort_values(["atleta_id", "rodada"]).copy()

    # ── Features de forma recente ──
    for janela in [3, 5, 10]:
        df[f"media_pts_{janela}r"] = (
            df.groupby("atleta_id")["pontos"]
            .transform(lambda x: x.shift(1).rolling(janela, min_periods=1).mean())
        )
        df[f"std_pts_{janela}r"] = (
            df.groupby("atleta_id")["pontos"]
            .transform(lambda x: x.shift(1).rolling(janela, min_periods=1).std().fillna(0))
        )

    # ── Última pontuação e tendência ──
    df["pts_ultima_rodada"] = df.groupby("atleta_id")["pontos"].shift(1)
    df["tendencia"] = df["media_pts_3r"] - df["media_pts_10r"]

    # ── Acumulados de scouts ──
    for col in ["scout_G", "scout_A", "scout_SG", "scout_GS", "scout_DD"]:
        if col in df.columns:
            df[f"acc_{col}"] = (
                df.groupby("atleta_id")[col]
                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
            )

    # ── Preço como proxy de qualidade ──
    df["preco_lag1"] = df.groupby("atleta_id")["preco"].shift(1)

    # ── Jogos consecutivos (regularidade) ──
    df["jogou"] = (df["pontos"] > 0).astype(int)
    df["regularidade_5r"] = (
        df.groupby("atleta_id")["jogou"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # ── Merge com mando de campo ──
    if partidas_df is not None and not partidas_df.empty:
        df = df.merge(
            partidas_df[["rodada", "clube_id", "mando"]],
            on=["rodada", "clube_id"],
            how="left",
        )
        df["mando"] = df["mando"].fillna(0)
    else:
        df["mando"] = 0  # neutro se não disponível

    # ── Label encoding de posição e clube ──
    df["posicao_enc"] = df["posicao_id"].astype(int)

    le_clube = LabelEncoder()
    df["clube_enc"] = le_clube.fit_transform(df["clube_id"].astype(str))

    return df


def preparar_partidas(api: CartolaAPI, rodadas_alvo: list[int]) -> pd.DataFrame:
    """
    Extrai mando de campo (casa=1, fora=-1) por (rodada, clube_id).
    """
    registros = []
    for rodada in tqdm(rodadas_alvo, desc="Coletando partidas"):
        try:
            data = api.partidas(rodada)
            for partida in data.get("partidas", []):
                registros.append({
                    "rodada": rodada,
                    "clube_id": partida.get("clube_casa_id"),
                    "mando": 1,
                })
                registros.append({
                    "rodada": rodada,
                    "clube_id": partida.get("clube_visitante_id"),
                    "mando": -1,
                })
        except Exception as e:
            log.warning(f"Partidas rodada {rodada}: {e}")
        time.sleep(0.2)

    return pd.DataFrame(registros) if registros else pd.DataFrame()


# ──────────────────────────────────────────────
# 3. MODELO DE PREDIÇÃO
# ──────────────────────────────────────────────

FEATURE_COLS = [
    "posicao_enc", "clube_enc", "preco_lag1",
    "media_pts_3r", "media_pts_5r", "media_pts_10r",
    "std_pts_3r", "std_pts_5r",
    "pts_ultima_rodada", "tendencia",
    "regularidade_5r", "mando",
    "acc_scout_G", "acc_scout_A", "acc_scout_SG",
    "acc_scout_GS", "acc_scout_DD",
]


def treinar_modelo(df: pd.DataFrame, rodada_corte: int):
    """
    Treina LightGBM com validação temporal.
    Usa rodadas < rodada_corte para treino.
    Retorna modelo treinado e MAE de validação.
    """
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    df_model = df[df["pontos"].notna()].copy()

    # Separação temporal (sem data leakage)
    treino = df_model[df_model["rodada"] < rodada_corte - 3]
    val    = df_model[(df_model["rodada"] >= rodada_corte - 3) &
                      (df_model["rodada"] < rodada_corte)]

    if treino.empty or val.empty:
        raise ValueError("Dados insuficientes para treino/validação.")

    X_treino = treino[feat_cols].fillna(0)
    y_treino = treino["pontos"]
    X_val    = val[feat_cols].fillna(0)
    y_val    = val["pontos"]

    params = {
        "objective": "regression_l1",   # MAE — robusto a outliers
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_treino, y_treino,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    log.info(f"Validação MAE: {mae:.3f} pts | Features: {len(feat_cols)}")

    return model, feat_cols, mae


# ──────────────────────────────────────────────
# 4. OTIMIZAÇÃO DA ESCALAÇÃO
# ──────────────────────────────────────────────

def otimizar_escalacao(
    df_mercado: pd.DataFrame,
    model: lgb.LGBMRegressor,
    feat_cols: list[str],
    budget: float = BUDGET,
    formation: dict = FORMATION,
    capitao_bonus: float = 2.0,
    max_por_clube: int = 5,
) -> pd.DataFrame:
    """
    Otimização inteira (ILP) para maximizar pontos esperados
    sujeito a restrições de: formação, orçamento e máximo por clube.

    Retorna DataFrame com o time escalado + coluna `capitao`.
    """
    df = df_mercado.copy()
    feat_cols_available = [c for c in feat_cols if c in df.columns]
    df["pts_pred"] = model.predict(df[feat_cols_available].fillna(0))
    df["pts_pred"] = df["pts_pred"].clip(lower=0)

    # Apenas jogadores disponíveis (status 2 = dúvida ainda entra; 7 = suspenso/machucado - excluir)
    df = df[~df["status_id"].isin([5, 7])].copy()
    df = df.reset_index(drop=True)

    n = len(df)
    prob = pulp.LpProblem("cartola_escalacao", pulp.LpMaximize)

    # Variáveis: x[i]=1 se jogador i escalado, cap[i]=1 se capitão
    x   = pulp.LpVariable.dicts("x",   range(n), cat="Binary")
    cap = pulp.LpVariable.dicts("cap", range(n), cat="Binary")

    # Objetivo: pontos esperados (capitão vale dobro)
    prob += pulp.lpSum(
        df.loc[i, "pts_pred"] * x[i] +
        df.loc[i, "pts_pred"] * (capitao_bonus - 1) * cap[i]
        for i in range(n)
    )

    # Restrição: orçamento
    prob += pulp.lpSum(df.loc[i, "preco"] * x[i] for i in range(n)) <= budget

    # Restrição: formação por posição
    total_jogadores = sum(formation.values())
    for pos_id, qtd in formation.items():
        jogadores_pos = [i for i in range(n) if df.loc[i, "posicao_id"] == pos_id]
        prob += pulp.lpSum(x[i] for i in jogadores_pos) == qtd

    # Restrição: exatamente 1 capitão (entre os escalados)
    prob += pulp.lpSum(cap[i] for i in range(n)) == 1
    for i in range(n):
        prob += cap[i] <= x[i]

    # Restrição: máximo de jogadores por clube
    for clube_id in df["clube_id"].unique():
        idx = df[df["clube_id"] == clube_id].index.tolist()
        prob += pulp.lpSum(x[i] for i in idx) <= max_por_clube

    # Resolver
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Otimização falhou: {pulp.LpStatus[status]}")

    escalados = [i for i in range(n) if pulp.value(x[i]) == 1]
    time_df = df.loc[escalados].copy()
    time_df["capitao"] = [pulp.value(cap[i]) == 1 for i in escalados]
    time_df["pts_capitao"] = time_df.apply(
        lambda r: r["pts_pred"] * capitao_bonus if r["capitao"] else r["pts_pred"], axis=1
    )

    return time_df.sort_values("posicao_id")


# ──────────────────────────────────────────────
# 5. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────

def imprimir_time(time_df: pd.DataFrame):
    """Exibe o time escalado de forma legível."""
    total_pred = time_df["pts_capitao"].sum()
    total_preco = time_df["preco"].sum()

    print("\n" + "="*60)
    print(f"{'CARTOLA FC - TIME OTIMIZADO':^60}")
    print("="*60)
    print(f"{'Pos':<6} {'Apelido':<22} {'Clube':<18} {'Preço':>6} {'Pts Pred':>8}")
    print("-"*60)

    for _, row in time_df.iterrows():
        pos = POSICAO_NOME.get(row["posicao_id"], "?")
        cap_flag = " ©" if row["capitao"] else ""
        print(f"{pos:<6} {row['apelido'][:20]+cap_flag:<22} "
              f"{row['clube_nome'][:16]:<18} "
              f"{row['preco']:>6.1f} {row['pts_capitao']:>8.2f}")

    print("-"*60)
    print(f"{'TOTAL':<46} {total_preco:>6.1f} {total_pred:>8.2f}")
    print(f"{'Budget restante: '}{BUDGET - total_preco:.1f} cartoletas")
    print("="*60)


def run_pipeline(
    token: str = None,
    rodadas_historico: list[int] = None,
    rodada_alvo: int = None,
    budget: float = BUDGET,
    formation: dict = FORMATION,
    salvar_csv: bool = True,
):
    """
    Executa o pipeline completo:
      coleta → features → treino → predição → otimização → exibe time
    """
    api = CartolaAPI(token=token)

    # Detectar rodada atual automaticamente
    status = api.mercado_status()
    rodada_atual = status.get("rodada_atual", status.get("rodada", {}).get("rodada_atual", 1))
    log.info(f"Rodada atual detectada: {rodada_atual}")

    if rodada_alvo is None:
        rodada_alvo = rodada_atual

    if rodadas_historico is None:
        # Usar todas as rodadas anteriores disponíveis
        rodadas_historico = list(range(1, rodada_alvo))

    if not rodadas_historico:
        raise ValueError("Nenhuma rodada histórica disponível (é a rodada 1?)")

    # ── Coleta histórica ──
    hist_file = DATA_DIR / "historico.parquet"
    if hist_file.exists():
        log.info("Carregando histórico em cache...")
        df_hist = pd.read_parquet(hist_file)
        rodadas_faltando = [r for r in rodadas_historico if r not in df_hist["rodada"].unique()]
        if rodadas_faltando:
            log.info(f"Coletando {len(rodadas_faltando)} rodadas novas...")
            df_novo = coletar_historico(api, rodadas_faltando)
            df_hist = pd.concat([df_hist, df_novo], ignore_index=True)
            df_hist.to_parquet(hist_file, index=False)
    else:
        df_hist = coletar_historico(api, rodadas_historico)
        df_hist.to_parquet(hist_file, index=False)

    # ── Partidas (mando de campo) ──
    df_partidas = preparar_partidas(api, rodadas_historico)

    # ── Features ──
    log.info("Construindo features...")
    df_feat = construir_features(df_hist, df_partidas)

    # ── Treino ──
    log.info(f"Treinando modelo para predizer rodada {rodada_alvo}...")
    model, feat_cols, mae = treinar_modelo(df_feat, rodada_corte=rodada_alvo)

    # ── Mercado da rodada alvo ──
    log.info("Buscando atletas disponíveis no mercado...")
    mercado_raw = api.atletas_mercado()
    df_mercado = pd.DataFrame(mercado_raw.get("atletas", []))

    clubes_raw = api.clubes()
    clubes_map = {int(k): v["nome"] for k, v in clubes_raw.items()}
    df_mercado["clube_nome"] = df_mercado["clube_id"].map(clubes_map).fillna("")
    df_mercado["rodada"] = rodada_alvo

    # Fazer merge de features históricas (última snapshot do jogador)
    ultima_feat = (
        df_feat[df_feat["rodada"] == df_feat["rodada"].max()]
        [["atleta_id"] + [c for c in feat_cols if c not in df_mercado.columns]]
        .drop_duplicates("atleta_id")
    )
    df_mercado = df_mercado.merge(ultima_feat, on="atleta_id", how="left")

    # Preencher colunas numéricas faltantes com a média da posição
    for col in feat_cols:
        if col in df_mercado.columns:
            df_mercado[col] = df_mercado[col].fillna(
                df_mercado.groupby("posicao_id")[col].transform("median")
            ).fillna(0)

    # ── Otimização ──
    log.info("Otimizando escalação...")
    time_df = otimizar_escalacao(df_mercado, model, feat_cols, budget, formation)

    # ── Output ──
    imprimir_time(time_df)

    if salvar_csv:
        out = DATA_DIR / f"time_rodada_{rodada_alvo}.csv"
        time_df.to_csv(out, index=False)
        log.info(f"Time salvo em {out}")

    return time_df, model


# ──────────────────────────────────────────────
# EXECUÇÃO
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cartola FC ML Team Builder")
    parser.add_argument("--token",   type=str,  default=None,  help="X-GLB-Token (opcional)")
    parser.add_argument("--rodada",  type=int,  default=None,  help="Rodada alvo (padrão: atual)")
    parser.add_argument("--budget",  type=float, default=140.0, help="Orçamento em cartoletas")
    parser.add_argument("--historico", type=int, nargs="+", default=None,
                        help="Lista de rodadas históricas (padrão: todas anteriores à alvo)")
    args = parser.parse_args()

    time_df, model = run_pipeline(
        token=args.token,
        rodada_alvo=args.rodada,
        budget=args.budget,
        rodadas_historico=args.historico,
    )
