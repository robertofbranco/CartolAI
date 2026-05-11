"""
Cartola FC - ML Team Builder
============================
Lê os dados coletados por cartola_collector.py e executa:
  1. Engenharia de features (scouts, forma, casa/fora, etc.)
  2. Treinamento de modelo LightGBM
  3. Otimização da escalação via programação linear (ILP)

Pré-requisito:
    python cartola_collector.py --token SEU_TOKEN

Uso:
    python cartola_team_builder.py --token SEU_TOKEN --rodada 15
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import pulp

from cartola_collector import (
    CartolaAPI,
    BASE_URL,
    DATA_DIR,
    BUDGET,
    FORMATION,
    POSICAO_NOME,
    SCOUT_POINTS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. ENGENHARIA DE FEATURES
# ──────────────────────────────────────────────

def construir_features(
    df: pd.DataFrame,
    partidas_df: pd.DataFrame = None,
    odds_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Gera features temporais e contextuais para cada (atleta, rodada).
    Usa apenas dados do PASSADO (sem data leakage).

    `odds_df` (opcional): DataFrame com colunas
    [rodada, clube_id, prob_win, prob_draw, prob_loss] do Gato Mestre.
    Odds são publicadas ANTES da partida, então alinhar (rodada, clube_id) com
    a linha de pontuação não vaza informação.
    """
    scout_cols = [c for c in df.columns if c.startswith("scout_")]
    df[scout_cols] = df[scout_cols].fillna(0)
    df = df.sort_values(["atleta_id", "rodada"]).copy()

    for janela in [3, 5, 10]:
        df[f"media_pts_{janela}r"] = (
            df.groupby("atleta_id")["pontos"]
            .transform(lambda x: x.shift(1).rolling(janela, min_periods=1).mean())
        )
        df[f"std_pts_{janela}r"] = (
            df.groupby("atleta_id")["pontos"]
            .transform(lambda x: x.shift(1).rolling(janela, min_periods=1).std().fillna(0))
        )

    df["pts_ultima_rodada"] = df.groupby("atleta_id")["pontos"].shift(1)
    df["tendencia"] = df["media_pts_3r"] - df["media_pts_10r"]

    for col in ["scout_G", "scout_A", "scout_SG", "scout_GS", "scout_DD"]:
        if col in df.columns:
            df[f"acc_{col}"] = (
                df.groupby("atleta_id")[col]
                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
            )

    df["preco_lag1"] = df.groupby("atleta_id")["preco"].shift(1)

    df["jogou"] = (df["pontos"] > 0).astype(int)
    df["regularidade_5r"] = (
        df.groupby("atleta_id")["jogou"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    if partidas_df is not None and not partidas_df.empty:
        df = df.merge(
            partidas_df[["rodada", "clube_id", "mando"]],
            on=["rodada", "clube_id"],
            how="left",
        )
        df["mando"] = df["mando"].fillna(0)
    else:
        df["mando"] = 0

    odds_cols = ["prob_win", "prob_draw", "prob_loss"]
    if odds_df is not None and not odds_df.empty:
        df = df.merge(
            odds_df[["rodada", "clube_id"] + odds_cols],
            on=["rodada", "clube_id"],
            how="left",
        )
        for col, default in zip(odds_cols, [0.33, 0.34, 0.33]):
            df[col] = df[col].fillna(default)
    else:
        df["prob_win"], df["prob_draw"], df["prob_loss"] = 0.33, 0.34, 0.33

    df["posicao_enc"] = df["posicao_id"].astype(int)
    df["clube_enc"] = LabelEncoder().fit_transform(df["clube_id"].astype(str))

    return df


# ──────────────────────────────────────────────
# 2. MODELO DE PREDIÇÃO
# ──────────────────────────────────────────────

FEATURE_COLS = [
    "posicao_enc", "clube_enc", "preco_lag1",
    "media_pts_3r", "media_pts_5r", "media_pts_10r",
    "std_pts_3r", "std_pts_5r",
    "pts_ultima_rodada", "tendencia",
    "regularidade_5r", "mando",
    "acc_scout_G", "acc_scout_A", "acc_scout_SG",
    "acc_scout_GS", "acc_scout_DD",
    "prob_win", "prob_draw", "prob_loss",
]


def treinar_modelo(df: pd.DataFrame, rodada_corte: int):
    """
    Treina LightGBM com validação temporal (sem data leakage).
    Retorna (modelo, feature_cols).
    """
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    df_model = df[df["pontos"].notna()].copy()

    treino = df_model[df_model["rodada"] < rodada_corte - 3]
    val    = df_model[(df_model["rodada"] >= rodada_corte - 3) &
                      (df_model["rodada"] < rodada_corte)]

    if treino.empty or val.empty:
        raise ValueError("Dados insuficientes para treino/validação.")

    X_treino, y_treino = treino[feat_cols].fillna(0), treino["pontos"]
    X_val,    y_val    = val[feat_cols].fillna(0),    val["pontos"]

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_treino, y_treino,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    return model, feat_cols


# ──────────────────────────────────────────────
# 3. OTIMIZAÇÃO DA ESCALAÇÃO
# ──────────────────────────────────────────────

def otimizar_escalacao(
    df_mercado: pd.DataFrame,
    model: lgb.LGBMRegressor,
    feat_cols: list[str],
    budget: float = BUDGET,
    formation: dict = FORMATION,
    capitao_bonus: float = 1.5,
    max_por_clube: int = 5,
    alpha_pred: float = 0.3,
    capitao_posicoes: tuple = (4, 5),
) -> pd.DataFrame:
    """
    ILP para maximizar pontos esperados respeitando formação, budget e limite por clube.

    `alpha_pred` blends model prediction with the season-average prior:
        score = alpha * model_pred + (1 - alpha) * media_num
    Set 1.0 to trust the model fully, 0.0 to fall back to media_num only.

    `capitao_posicoes` restricts who can wear the C — defaults to MEI/ATA
    """
    df = df_mercado.copy()
    raw_pred = model.predict(df[[c for c in feat_cols if c in df.columns]].fillna(0))
    raw_pred = np.clip(raw_pred, 0, None)
    media = df["media"].fillna(0).to_numpy() if "media" in df.columns else np.zeros(len(df))
    df["pts_pred"] = alpha_pred * raw_pred + (1.0 - alpha_pred) * media
    df = df[~df["status_id"].isin([5, 7])].reset_index(drop=True)

    n = len(df)
    cap_ok = {i for i in range(n) if df.loc[i, "posicao_id"] in capitao_posicoes}

    prob = pulp.LpProblem("cartola_escalacao", pulp.LpMaximize)
    x   = pulp.LpVariable.dicts("x",   range(n), cat="Binary")
    cap = pulp.LpVariable.dicts("cap", range(n), cat="Binary")

    prob += pulp.lpSum(
        df.loc[i, "pts_pred"] * x[i] + df.loc[i, "pts_pred"] * (capitao_bonus - 1) * cap[i]
        for i in range(n)
    )
    prob += pulp.lpSum(df.loc[i, "preco"] * x[i] for i in range(n)) <= budget

    for pos_id, qtd in formation.items():
        idx = [i for i in range(n) if df.loc[i, "posicao_id"] == pos_id]
        prob += pulp.lpSum(x[i] for i in idx) == qtd

    prob += pulp.lpSum(cap[i] for i in range(n)) == 1
    for i in range(n):
        prob += cap[i] <= x[i]
        if i not in cap_ok:
            prob += cap[i] == 0

    for clube_id in df["clube_id"].unique():
        idx = df[df["clube_id"] == clube_id].index.tolist()
        prob += pulp.lpSum(x[i] for i in idx) <= max_por_clube

    status = pulp.PULP_CBC_CMD(msg=False).solve(prob)
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
# 4. DISPLAY
# ──────────────────────────────────────────────

def imprimir_time(time_df: pd.DataFrame):
    total_pred  = time_df["pts_capitao"].sum()
    total_preco = time_df["preco"].sum()

    print("\n" + "="*60)
    print(f"{'CARTOLA FC - TIME OTIMIZADO':^60}")
    print("="*60)
    print(f"{'Pos':<6} {'Apelido':<22} {'Clube':<18} {'Preço':>6} {'Pts Pred':>8}")
    print("-"*60)
    for _, row in time_df.iterrows():
        pos      = POSICAO_NOME.get(row["posicao_id"], "?")
        cap_flag = " ©" if row["capitao"] else ""
        print(f"{pos:<6} {row['apelido'][:20]+cap_flag:<22} "
              f"{row['clube_nome'][:16]:<18} "
              f"{row['preco']:>6.1f} {row['pts_capitao']:>8.2f}")
    print("-"*60)
    print(f"{'TOTAL':<46} {total_preco:>6.1f} {total_pred:>8.2f}")
    print(f"Budget restante: {BUDGET - total_preco:.1f} cartoletas")
    print("="*60)


# ──────────────────────────────────────────────
# 5. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────

def run_pipeline(
    token: str = None,
    rodada_alvo: int = None,
    budget: float = BUDGET,
    formation: dict = FORMATION,
    salvar_csv: bool = True,
):
    """
    Carrega historico.parquet (coletado pelo cartola_collector.py),
    treina o modelo e otimiza a escalação para a rodada alvo.
    """
    hist_file = DATA_DIR / "historico.parquet"
    if not hist_file.exists():
        raise FileNotFoundError(
            "historico.parquet não encontrado. "
            "Execute cartola_collector.py primeiro."
        )

    df_hist = pd.read_parquet(hist_file)
    log.info(f"Histórico carregado: {df_hist['rodada'].nunique()} rodadas, "
             f"{df_hist['atleta_id'].nunique()} atletas")

    partidas_file = DATA_DIR / "partidas.parquet"
    df_partidas = pd.read_parquet(partidas_file) if partidas_file.exists() else pd.DataFrame()

    odds_file = DATA_DIR / "odds.parquet"
    df_odds = pd.read_parquet(odds_file) if odds_file.exists() else pd.DataFrame()

    # Detectar rodada alvo via API ou último dado disponível
    if rodada_alvo is None:
        api = CartolaAPI(token=token)
        try:
            status = api.mercado_status()
            rodada_alvo = status.get("rodada_atual", status.get("rodada", {}).get("rodada_atual", 1))
        except Exception:
            rodada_alvo = int(df_hist["rodada"].max()) + 1
            log.warning(f"API indisponível — usando rodada inferida: {rodada_alvo}")
    log.info(f"Rodada alvo: {rodada_alvo}")

    log.info("Construindo features...")
    df_feat = construir_features(df_hist, df_partidas, df_odds)

    log.info(f"Treinando modelo para rodada {rodada_alvo}...")
    model, feat_cols = treinar_modelo(df_feat, rodada_corte=rodada_alvo)

    # Buscar mercado ao vivo
    api = CartolaAPI(token=token)
    log.info("Buscando atletas disponíveis no mercado...")
    mercado_raw  = api.atletas_mercado()
    df_mercado   = pd.DataFrame(mercado_raw.get("atletas", []))
    clubes_map   = {int(k): v["nome"] for k, v in api.clubes().items()}
    df_mercado["clube_nome"] = df_mercado["clube_id"].map(clubes_map).fillna("")
    df_mercado["rodada"]     = rodada_alvo

    odds_cols = ["prob_win", "prob_draw", "prob_loss"]
    feat_cols_no_odds = [c for c in feat_cols if c not in odds_cols]
    ultima_feat = (
        df_feat[df_feat["rodada"] == df_feat["rodada"].max()]
        [["atleta_id"] + [c for c in feat_cols_no_odds if c not in df_mercado.columns]]
        .drop_duplicates("atleta_id")
    )
    df_mercado = df_mercado.merge(ultima_feat, on="atleta_id", how="left")

    if not df_odds.empty:
        odds_alvo = df_odds[df_odds["rodada"] == rodada_alvo][["clube_id"] + odds_cols]
        df_mercado = df_mercado.merge(odds_alvo, on="clube_id", how="left")
    for col, default in zip(odds_cols, [0.33, 0.34, 0.33]):
        if col not in df_mercado.columns:
            df_mercado[col] = default
        else:
            df_mercado[col] = df_mercado[col].fillna(default)

    for col in feat_cols:
        if col in df_mercado.columns:
            df_mercado[col] = df_mercado[col].fillna(
                df_mercado.groupby("posicao_id")[col].transform("median")
            ).fillna(0)

    log.info("Otimizando escalação...")
    time_df = otimizar_escalacao(df_mercado, model, feat_cols, budget, formation)

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
    parser = argparse.ArgumentParser(description="Cartola FC ML Team Builder")
    parser.add_argument("--token",  type=str,   default=None,
                        help="X-GLB-Token (Bearer ...). Default: $CARTOLA_TOKEN")
    parser.add_argument("--rodada", type=int,   default=None,  help="Rodada alvo (padrão: atual)")
    parser.add_argument("--budget", type=float, default=160.0, help="Orçamento em cartoletas")
    args = parser.parse_args()
    token = args.token or os.environ.get("CARTOLA_TOKEN")
    run_pipeline(token=token, rodada_alvo=args.rodada, budget=args.budget)
