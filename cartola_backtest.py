"""
Cartola FC - Backtesting Engine
================================
Simula o pipeline rodada a rodada em dados históricos para medir:
  - MAE de predição por rodada
  - Pontuação real do time otimizado vs. baseline (melhor time possível a posteriori)
  - Eficiência de budget
  - Ranking estimado (percentil) em relação à média do mercado

Uso:
    python cartola_backtest.py --inicio 5 --fim 20 --budget 140

Requisitos: mesmo requirements.txt do pipeline principal
"""

import logging
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from cartola_team_builder import (
    construir_features,
    treinar_modelo,
    otimizar_escalacao,
    POSICAO_NOME,
    FORMATION,
    BUDGET,
    DATA_DIR,
    FEATURE_COLS,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# ESTRUTURA DE RESULTADO
# ──────────────────────────────────────────────

@dataclass
class ResultadoRodada:
    rodada: int
    mae_predicao: float                  # erro médio de predição (pontos)
    pts_modelo: float                    # pontos reais do time escolhido pelo modelo
    pts_baseline_media: float            # pontos do time escolhido pela média histórica simples
    pts_teto: float                      # pontos do melhor time possível (oracle, a posteriori)
    eficiencia: float                    # pts_modelo / pts_teto  (0–1)
    budget_usado: float                  # cartoletas gastas
    capitao: str                         # apelido do capitão escolhido
    capitao_pts_reais: float             # pontos reais do capitão
    time_escalado: pd.DataFrame = field(repr=False)


# ──────────────────────────────────────────────
# ORACLE (teto teórico)
# ──────────────────────────────────────────────

def calcular_teto(df_rodada_real: pd.DataFrame, budget: float, formation: dict) -> float:
    """
    Monta o melhor time possível com pontuação REAL da rodada (oracle).
    Usado como denominador da eficiência.
    """
    import pulp

    df = df_rodada_real.copy()
    df = df[df["pontos"] >= 0].reset_index(drop=True)
    n = len(df)

    prob = pulp.LpProblem("teto", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(n), cat="Binary")
    cap = pulp.LpVariable.dicts("cap", range(n), cat="Binary")

    prob += pulp.lpSum(
        df.loc[i, "pontos"] * x[i] + df.loc[i, "pontos"] * cap[i]
        for i in range(n)
    )
    prob += pulp.lpSum(df.loc[i, "preco"] * x[i] for i in range(n)) <= budget

    for pos_id, qtd in formation.items():
        idx = [i for i in range(n) if df.loc[i, "posicao_id"] == pos_id]
        prob += pulp.lpSum(x[i] for i in idx) == qtd

    prob += pulp.lpSum(cap[i] for i in range(n)) == 1
    for i in range(n):
        prob += cap[i] <= x[i]

    for clube_id in df["clube_id"].unique():
        idx = df[df["clube_id"] == clube_id].index.tolist()
        prob += pulp.lpSum(x[i] for i in idx) <= 5

    pulp.PULP_CBC_CMD(msg=False).solve(prob)

    escalados = [i for i in range(n) if pulp.value(x[i]) == 1]
    cap_idx = [i for i in range(n) if pulp.value(cap[i]) == 1]

    pts = sum(df.loc[i, "pontos"] for i in escalados)
    pts += df.loc[cap_idx[0], "pontos"] if cap_idx else 0
    return pts


def calcular_baseline_media(df_mercado: pd.DataFrame, budget: float, formation: dict) -> float:
    """
    Time montado apenas com `media_num` (média histórica do Cartola) — sem ML.
    Serve como baseline simples para comparação.
    """
    import pulp

    df = df_mercado.copy().reset_index(drop=True)
    n = len(df)

    prob = pulp.LpProblem("baseline", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(n), cat="Binary")
    cap = pulp.LpVariable.dicts("cap", range(n), cat="Binary")

    prob += pulp.lpSum(
        df.loc[i, "media"] * x[i] + df.loc[i, "media"] * cap[i]
        for i in range(n)
    )
    prob += pulp.lpSum(df.loc[i, "preco"] * x[i] for i in range(n)) <= budget

    for pos_id, qtd in formation.items():
        idx = [i for i in range(n) if df.loc[i, "posicao_id"] == pos_id]
        prob += pulp.lpSum(x[i] for i in idx) == qtd

    prob += pulp.lpSum(cap[i] for i in range(n)) == 1
    for i in range(n):
        prob += cap[i] <= x[i]

    for clube_id in df["clube_id"].unique():
        idx = df[df["clube_id"] == clube_id].index.tolist()
        prob += pulp.lpSum(x[i] for i in idx) <= 5

    pulp.PULP_CBC_CMD(msg=False).solve(prob)
    escalados = [i for i in range(n) if pulp.value(x[i]) == 1]
    cap_idx = [i for i in range(n) if pulp.value(cap[i]) == 1]

    pts = sum(df.loc[i, "media"] for i in escalados)
    pts += df.loc[cap_idx[0], "media"] if cap_idx else 0
    return pts


# ──────────────────────────────────────────────
# ENGINE DE BACKTESTING
# ──────────────────────────────────────────────

def rodar_backtest(
    df_hist: pd.DataFrame,
    df_partidas: pd.DataFrame,
    rodada_inicio: int,
    rodada_fim: int,
    budget: float = BUDGET,
    formation: dict = FORMATION,
    min_rodadas_treino: int = 5,
) -> list[ResultadoRodada]:
    """
    Para cada rodada no intervalo [rodada_inicio, rodada_fim]:
      1. Treina o modelo com tudo que veio ANTES dessa rodada
      2. Usa o mercado daquela rodada para montar o time
      3. Compara com os pontos REAIS da rodada (que o modelo nunca viu)
    """
    resultados = []

    for rodada_alvo in range(rodada_inicio, rodada_fim + 1):
        log.info(f"── Backtesting rodada {rodada_alvo} ──")

        # Dados disponíveis até esta rodada (sem ver o futuro)
        df_treino_base = df_hist[df_hist["rodada"] < rodada_alvo].copy()

        if df_treino_base["rodada"].nunique() < min_rodadas_treino:
            log.warning(f"Rodada {rodada_alvo}: dados insuficientes, pulando.")
            continue

        # Features com dados até a rodada anterior
        partidas_hist = df_partidas[df_partidas["rodada"] < rodada_alvo] if not df_partidas.empty else pd.DataFrame()
        df_feat = construir_features(df_treino_base, partidas_hist)

        # Treinar modelo
        try:
            model, feat_cols, mae = treinar_modelo(df_feat, rodada_corte=rodada_alvo)
        except ValueError as e:
            log.warning(f"Rodada {rodada_alvo}: {e}")
            continue

        # Simular mercado: snapshot dos jogadores na rodada alvo
        # (usamos os dados daquela rodada como proxy de mercado)
        df_rodada_real = df_hist[df_hist["rodada"] == rodada_alvo].copy()
        if df_rodada_real.empty:
            log.warning(f"Rodada {rodada_alvo}: sem dados reais, pulando.")
            continue

        # Enriquecer com features da rodada anterior (o que o modelo veria ao vivo)
        ultima_feat = (
            df_feat[df_feat["rodada"] == df_feat["rodada"].max()]
            [["atleta_id"] + [c for c in feat_cols if c not in df_rodada_real.columns]]
            .drop_duplicates("atleta_id")
        )
        df_mercado_sim = df_rodada_real.merge(ultima_feat, on="atleta_id", how="left")

        for col in feat_cols:
            if col in df_mercado_sim.columns:
                df_mercado_sim[col] = df_mercado_sim[col].fillna(
                    df_mercado_sim.groupby("posicao_id")[col].transform("median")
                ).fillna(0)

        # Montar time com o modelo
        try:
            time_modelo = otimizar_escalacao(df_mercado_sim, model, feat_cols, budget, formation)
        except Exception as e:
            log.warning(f"Rodada {rodada_alvo}: otimização falhou — {e}")
            continue

        # Pontuação REAL dos jogadores escolhidos pelo modelo
        reais = df_rodada_real.set_index("atleta_id")["pontos"].to_dict()
        pts_reais_lista = [reais.get(aid, 0) for aid in time_modelo["atleta_id"]]
        cap_idx = time_modelo["capitao"].values
        pts_modelo = sum(
            p * 2 if c else p
            for p, c in zip(pts_reais_lista, cap_idx)
        )

        # Capitão
        cap_row = time_modelo[time_modelo["capitao"]].iloc[0]
        capitao_pts_reais = reais.get(cap_row["atleta_id"], 0)

        # Teto (oracle)
        pts_teto = calcular_teto(df_rodada_real, budget, formation)

        # Baseline (média histórica simples)
        pts_baseline = calcular_baseline_media(df_mercado_sim, budget, formation)

        eficiencia = pts_modelo / pts_teto if pts_teto > 0 else 0

        resultado = ResultadoRodada(
            rodada=rodada_alvo,
            mae_predicao=mae,
            pts_modelo=pts_modelo,
            pts_baseline_media=pts_baseline,
            pts_teto=pts_teto,
            eficiencia=eficiencia,
            budget_usado=time_modelo["preco"].sum(),
            capitao=cap_row["apelido"],
            capitao_pts_reais=capitao_pts_reais,
            time_escalado=time_modelo,
        )
        resultados.append(resultado)
        log.info(
            f"  Modelo: {pts_modelo:.1f} | Baseline: {pts_baseline:.1f} | "
            f"Teto: {pts_teto:.1f} | Eficiência: {eficiencia:.1%} | MAE: {mae:.2f}"
        )

    return resultados


# ──────────────────────────────────────────────
# RELATÓRIO
# ──────────────────────────────────────────────

def gerar_relatorio(resultados: list[ResultadoRodada], output_dir: Path = DATA_DIR):
    """Gera DataFrame resumo + gráficos do backtesting."""

    df = pd.DataFrame([{
        "rodada":            r.rodada,
        "mae_predicao":      r.mae_predicao,
        "pts_modelo":        r.pts_modelo,
        "pts_baseline":      r.pts_baseline_media,
        "pts_teto":          r.pts_teto,
        "eficiencia":        r.eficiencia,
        "budget_usado":      r.budget_usado,
        "ganhou_baseline":   r.pts_modelo > r.pts_baseline_media,
        "capitao":           r.capitao,
        "capitao_pts_reais": r.capitao_pts_reais,
    } for r in resultados])

    # ── Sumário no terminal ──
    print("\n" + "="*70)
    print(f"{'BACKTEST SUMMARY':^70}")
    print("="*70)
    print(f"  Rodadas testadas:        {len(df)}")
    print(f"  Pts modelo  (média):     {df['pts_modelo'].mean():.2f}  ±{df['pts_modelo'].std():.2f}")
    print(f"  Pts baseline (média):    {df['pts_baseline'].mean():.2f}  ±{df['pts_baseline'].std():.2f}")
    print(f"  Pts teto    (média):     {df['pts_teto'].mean():.2f}")
    print(f"  Eficiência  (média):     {df['eficiencia'].mean():.1%}")
    print(f"  Bateu baseline:          {df['ganhou_baseline'].sum()}/{len(df)} rodadas "
          f"({df['ganhou_baseline'].mean():.0%})")
    print(f"  MAE médio:               {df['mae_predicao'].mean():.3f} pts")
    print(f"  Melhor rodada:           R{df.loc[df['pts_modelo'].idxmax(), 'rodada']} "
          f"({df['pts_modelo'].max():.1f} pts)")
    print(f"  Pior rodada:             R{df.loc[df['pts_modelo'].idxmin(), 'rodada']} "
          f"({df['pts_modelo'].min():.1f} pts)")
    print("="*70)

    # ── Gráficos ──
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Cartola FC — Backtesting Report", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    rodadas = df["rodada"].values

    # 1. Pontuação por rodada
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(rodadas, df["pts_teto"],     "--", color="gold",   label="Teto (oracle)", alpha=0.7)
    ax1.plot(rodadas, df["pts_modelo"],   "-o", color="#2196F3", label="Modelo ML",    linewidth=2)
    ax1.plot(rodadas, df["pts_baseline"], "-s", color="#FF7043", label="Baseline (média)", linewidth=1.5, alpha=0.8)
    ax1.fill_between(rodadas, df["pts_baseline"], df["pts_modelo"],
                     where=df["pts_modelo"] >= df["pts_baseline"],
                     alpha=0.15, color="#2196F3", label="Ganhou baseline")
    ax1.fill_between(rodadas, df["pts_baseline"], df["pts_modelo"],
                     where=df["pts_modelo"] < df["pts_baseline"],
                     alpha=0.15, color="#FF7043", label="Perdeu baseline")
    ax1.set_title("Pontuação Real por Rodada")
    ax1.set_xlabel("Rodada")
    ax1.set_ylabel("Pontos")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Eficiência (%)
    ax2 = fig.add_subplot(gs[0, 2])
    colors = ["#2196F3" if e >= 0.7 else "#FF7043" for e in df["eficiencia"]]
    ax2.bar(rodadas, df["eficiencia"] * 100, color=colors, alpha=0.8)
    ax2.axhline(df["eficiencia"].mean() * 100, color="black", linestyle="--",
                linewidth=1, label=f"Média {df['eficiencia'].mean():.0%}")
    ax2.set_title("Eficiência vs. Teto")
    ax2.set_xlabel("Rodada")
    ax2.set_ylabel("% do teto alcançado")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Ganho acumulado sobre o baseline
    ganho_acumulado = (df["pts_modelo"] - df["pts_baseline"]).cumsum()
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(rodadas, ganho_acumulado, "-o", color="#4CAF50", linewidth=2)
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.fill_between(rodadas, 0, ganho_acumulado,
                     where=ganho_acumulado >= 0, alpha=0.2, color="#4CAF50")
    ax3.fill_between(rodadas, 0, ganho_acumulado,
                     where=ganho_acumulado < 0, alpha=0.2, color="#F44336")
    ax3.set_title("Ganho Acumulado vs. Baseline (pontos)")
    ax3.set_xlabel("Rodada")
    ax3.set_ylabel("Δ pontos acumulados")
    ax3.grid(True, alpha=0.3)

    # 4. MAE por rodada
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(rodadas, df["mae_predicao"], "-o", color="#9C27B0", linewidth=2)
    ax4.axhline(df["mae_predicao"].mean(), color="black", linestyle="--",
                linewidth=1, label=f"Média {df['mae_predicao'].mean():.2f}")
    ax4.set_title("MAE de Predição por Rodada")
    ax4.set_xlabel("Rodada")
    ax4.set_ylabel("MAE (pontos)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.savefig(output_dir / "backtest_report.png", dpi=150, bbox_inches="tight")
    log.info(f"Gráfico salvo em {output_dir / 'backtest_report.png'}")

    # ── Salvar CSV ──
    df.to_csv(output_dir / "backtest_resultados.csv", index=False)
    log.info(f"Resultados salvos em {output_dir / 'backtest_resultados.csv'}")

    return df


# ──────────────────────────────────────────────
# EXECUÇÃO
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cartola FC — Backtesting Engine")
    parser.add_argument("--inicio",  type=int, default=6,    help="Primeira rodada a testar (mín. 6)")
    parser.add_argument("--fim",     type=int, default=20,   help="Última rodada a testar")
    parser.add_argument("--budget",  type=float, default=140.0)
    args = parser.parse_args()

    # Carregar histórico salvo pelo pipeline principal
    hist_file = DATA_DIR / "historico.parquet"
    if not hist_file.exists():
        raise FileNotFoundError(
            "historico.parquet não encontrado. "
            "Execute cartola_team_builder.py primeiro para coletar os dados."
        )

    df_hist = pd.read_parquet(hist_file)
    log.info(f"Histórico carregado: {df_hist['rodada'].nunique()} rodadas, "
             f"{df_hist['atleta_id'].nunique()} atletas únicos")

    # Partidas (mando de campo) — se existir
    partidas_file = DATA_DIR / "partidas.parquet"
    df_partidas = pd.read_parquet(partidas_file) if partidas_file.exists() else pd.DataFrame()

    # Rodar backtest
    resultados = rodar_backtest(
        df_hist=df_hist,
        df_partidas=df_partidas,
        rodada_inicio=args.inicio,
        rodada_fim=args.fim,
        budget=args.budget,
    )

    if not resultados:
        log.error("Nenhum resultado gerado. Verifique os dados históricos.")
        return

    # Relatório
    gerar_relatorio(resultados)


if __name__ == "__main__":
    main()
