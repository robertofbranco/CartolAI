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
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass, field

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
    pts_modelo: float                    # pontos reais do time escolhido pelo modelo
    pts_media_geral: float         # media de pontos de todos os times do cartola
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
        df.loc[i, "pontos"] * x[i] + 0.5 * df.loc[i, "pontos"] * cap[i]
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
    pts += 0.5 * df.loc[cap_idx[0], "pontos"] if cap_idx else 0
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
    df_mercado_atual: pd.DataFrame = None,
    df_media_cartoleiros: pd.DataFrame = None
) -> list[ResultadoRodada]:
    """
    Para cada rodada no intervalo [rodada_inicio, rodada_fim]:
      1. Treina o modelo com tudo que veio ANTES dessa rodada
      2. Usa o mercado daquela rodada para montar o time
      3. Compara com os pontos REAIS da rodada (que o modelo nunca viu)
    """
    resultados = []

    # Lookup: clube_id -> nome (do histórico, que já mapeia ambos)
    clubes_lookup = (
        df_hist.dropna(subset=["clube_id", "clube_nome"])
        .drop_duplicates("clube_id")
        .set_index("clube_id")["clube_nome"]
        .to_dict()
    )

    # Lookup: (rodada, clube_id) -> clube_adversario_id. Coluna pode não existir em
    # parquets antigos — quando ausente, o adversário cai para "?" no log.
    if not df_partidas.empty and "clube_adversario_id" in df_partidas.columns:
        adv_lookup = (
            df_partidas.dropna(subset=["clube_adversario_id"])
            .drop_duplicates(["rodada", "clube_id"])
            .set_index(["rodada", "clube_id"])["clube_adversario_id"]
            .to_dict()
        )
    else:
        adv_lookup = {}

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

        # /atletas/pontuados não retorna preco/media — vêm de mercado_atual.
        # Usamos o snapshot atual como proxy (preços não variam drasticamente intra-temporada).
        # Importante: NÃO importar status_id daqui — ele reflete "agora", não a rodada
        # alvo (entre rodadas, todos os técnicos ficam com status_id=7 e quebram o ILP).
        if df_mercado_atual is not None and not df_mercado_atual.empty:
            df_rodada_real = df_rodada_real.drop(
                columns=["preco", "media"], errors="ignore"
            ).merge(
                df_mercado_atual[["atleta_id", "preco", "media"]],
                on="atleta_id", how="left",
            )
            df_rodada_real = df_rodada_real[df_rodada_real["preco"].notna()].reset_index(drop=True)
            df_rodada_real["media"] = df_rodada_real["media"].fillna(0.0)
            if df_rodada_real.empty:
                log.warning(f"Rodada {rodada_alvo}: nenhum atleta cruzou com mercado_atual, pulando.")
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
            p * 1.5 if c else p
            for p, c in zip(pts_reais_lista, cap_idx)
        )

        # Capitão
        cap_row = time_modelo[time_modelo["capitao"]].iloc[0]
        capitao_pts_reais = reais.get(cap_row["atleta_id"], 0)

        # Anotar posição (string), adversário e pontos reais no lineup para o log/CSV
        time_modelo["posicao"] = time_modelo["posicao_id"].map(POSICAO_NOME)
        time_modelo["adversario"] = (
            time_modelo["clube_id"]
            .map(lambda cid: adv_lookup.get((rodada_alvo, cid)))
            .map(clubes_lookup)
            .fillna("?")
        )
        time_modelo["pontos_real"] = time_modelo["atleta_id"].map(reais).fillna(0.0)

        log.info(f"Time escalado R{rodada_alvo}:")
        for _, p in time_modelo.iterrows():
            cap_flag = " (C)" if p["capitao"] else ""
            apelido = (p["apelido"] or "")[:20]
            clube = (p.get("clube_nome") or "")[:16]
            adv = (p["adversario"] or "")[:16]
            log.info(
                f"  {p['posicao']:<3} {apelido:<20} {clube:<16} vs {adv:<16} "
                f"avg={p['media']:>5.2f} preco={p['preco']:>5.1f} "
                f"pts={p['pontos_real']:>5.1f}{cap_flag}"
            )

        # Teto (oracle)
        pts_teto = calcular_teto(df_rodada_real, budget, formation)
        
        pts_media_geral = df_media_cartoleiros.loc[df_media_cartoleiros["rodada"] == rodada_alvo, "media_cartoleiros"].iloc[0]

        eficiencia = pts_modelo / pts_teto if pts_teto > 0 else 0

        resultado = ResultadoRodada(
            rodada=rodada_alvo,            
            pts_modelo=pts_modelo,
            pts_media_geral=pts_media_geral,
            pts_teto=pts_teto,
            eficiencia=eficiencia,
            budget_usado=time_modelo["preco"].sum(),
            capitao=cap_row["apelido"],
            capitao_pts_reais=capitao_pts_reais,
            time_escalado=time_modelo,
        )
        resultados.append(resultado)
        log.info(
            f"  Modelo: {pts_modelo:.1f} | Media Geral: {pts_media_geral:.1f} | "
            f"Teto: {pts_teto:.1f} | Eficiência: {eficiencia:.1%}"
        )

    return resultados


# ──────────────────────────────────────────────
# RELATÓRIO
# ──────────────────────────────────────────────

def gerar_relatorio(resultados: list[ResultadoRodada], output_dir: Path = DATA_DIR):
    """Gera DataFrame resumo + gráficos do backtesting."""

    df = pd.DataFrame([{
        "rodada":             r.rodada,        
        "pts_modelo":         r.pts_modelo,
        "pts_media_geral":    r.pts_media_geral,
        "pts_teto":           r.pts_teto,
        "eficiencia":         r.eficiencia,
        "budget_usado":       r.budget_usado,
        "ganhou_media_geral": r.pts_modelo > r.pts_media_geral,
        "capitao":            r.capitao,
        "capitao_pts_reais":  r.capitao_pts_reais,
    } for r in resultados])

    # ── Sumário no terminal ──
    print("\n" + "="*70)
    print(f"{'BACKTEST SUMMARY':^70}")
    print("="*70)
    print(f"  Rodadas testadas:        {len(df)}")
    print(f"  Pts modelo  (média):     {df['pts_modelo'].mean():.2f}  ±{df['pts_modelo'].std():.2f}")
    print(f"  Pts media geral:         {df['pts_media_geral'].mean():.2f}  ±{df['pts_media_geral'].std():.2f}")
    print(f"  Pts teto    (média):     {df['pts_teto'].mean():.2f}")
    print(f"  Eficiência  (média):     {df['eficiencia'].mean():.1%}")
    print(f"  Bateu media geral:       {df['ganhou_media_geral'].sum()}/{len(df)} rodadas "
          f"({df['ganhou_media_geral'].mean():.0%})")    
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
    ax1.plot(rodadas, df["pts_media_geral"], "-s", color="#FF7043", label="Media geral", linewidth=1.5, alpha=0.8)
    ax1.fill_between(rodadas, df["pts_media_geral"], df["pts_modelo"],
                     where=df["pts_modelo"] >= df["pts_media_geral"],
                     alpha=0.15, color="#2196F3", label="Ganhou media geral")
    ax1.fill_between(rodadas, df["pts_media_geral"], df["pts_modelo"],
                     where=df["pts_modelo"] < df["pts_media_geral"],
                     alpha=0.15, color="#FF7043", label="Perdeu media geral")
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
    ganho_acumulado = (df["pts_modelo"] - df["pts_media_geral"]).cumsum()
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(rodadas, ganho_acumulado, "-o", color="#4CAF50", linewidth=2)
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.fill_between(rodadas, 0, ganho_acumulado,
                     where=ganho_acumulado >= 0, alpha=0.2, color="#4CAF50")
    ax3.fill_between(rodadas, 0, ganho_acumulado,
                     where=ganho_acumulado < 0, alpha=0.2, color="#F44336")
    ax3.set_title("Ganho Acumulado vs. Media geral (pontos)")
    ax3.set_xlabel("Rodada")
    ax3.set_ylabel("Δ pontos acumulados")
    ax3.grid(True, alpha=0.3)    

    plt.savefig(output_dir / "backtest" / "backtest_report.png", dpi=150, bbox_inches="tight")
    log.info(f"Gráfico salvo em {output_dir / "backtest" / 'backtest_report.png'}")

    # ── Salvar CSV ──
    df.to_csv(output_dir / "backtest" / "backtest_resultados.csv", index=False)
    log.info(f"Resultados salvos em {output_dir / "backtest" / 'backtest_resultados.csv'}")

    # ── Lineups consolidados (uma linha por jogador escalado por rodada) ──
    lineup_cols = ["posicao", "apelido", "clube_nome", "adversario", "capitao", "media", "preco", "pontos_real"]
    lineup_frames = []
    for r in resultados:
        if r.time_escalado is None or r.time_escalado.empty:
            continue
        lineup = r.time_escalado.reindex(columns=lineup_cols).copy()
        lineup.insert(0, "rodada", r.rodada)
        lineup_frames.append(lineup)

    if lineup_frames:
        lineup_df = pd.concat(lineup_frames, ignore_index=True).rename(columns={"clube_nome": "clube"})
        lineup_path = output_dir / "backtest" / "backtest_times.csv"
        lineup_df.to_csv(lineup_path, index=False)
        log.info(f"Lineups salvos em {lineup_path}")

    return df


# ──────────────────────────────────────────────
# EXECUÇÃO
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cartola FC — Backtesting Engine")
    parser.add_argument("--inicio",  type=int, default=8,    help="Primeira rodada a testar (mín. 6)")
    parser.add_argument("--fim",     type=int, default=14,   help="Última rodada a testar")
    parser.add_argument("--budget",  type=float, default=150.0)
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

    # Mercado atual (preco, media, status_id) — pré-requisito para budget realista
    mercado_file = DATA_DIR / "mercado_atual.parquet"
    if not mercado_file.exists():
        raise FileNotFoundError(
            "mercado_atual.parquet não encontrado. "
            "Execute cartola_collector.py para gerar o snapshot."
        )
    df_mercado_atual = pd.read_parquet(mercado_file)
    log.info(f"Mercado atual carregado: {len(df_mercado_atual)} atletas")

    media_cartoleiros_file = DATA_DIR / "medias_cartoleiros.parquet"
    df_media_cartoleiros = pd.read_parquet(media_cartoleiros_file)
    log.info(f"Medias dos cartoleiros carregadas: {len(df_media_cartoleiros)} medias")

    # Rodar backtest
    resultados = rodar_backtest(
        df_hist=df_hist,
        df_partidas=df_partidas,
        rodada_inicio=args.inicio,
        rodada_fim=args.fim,
        budget=args.budget,
        df_mercado_atual=df_mercado_atual,
        df_media_cartoleiros=df_media_cartoleiros
    )

    if not resultados:
        log.error("Nenhum resultado gerado. Verifique os dados históricos.")
        return

    # Relatório
    gerar_relatorio(resultados)


if __name__ == "__main__":
    main()
