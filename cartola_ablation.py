"""
Cartola FC — Feature Ablation
==============================
Roda backtests com subconjuntos crescentes de features para ver quais ajudam
e quais machucam no regime de poucos dados (~13 rodadas).

Cada grupo é CUMULATIVO: adiciona às features do grupo anterior. O último grupo
liga o multiplier explícito de matchup no otimizador.

Uso:
    python cartola_ablation.py --inicio 8 --fim 14 --budget 150
"""

import argparse
import logging
from dataclasses import dataclass, field

import pandas as pd

import cartola_team_builder as ctb
from cartola_team_builder import DATA_DIR, POSITION_FEATURES
from cartola_backtest import rodar_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


PLAYER_BASE = [
    "posicao_enc", "preco_lag1",
    "media_pts_3r", "media_pts_5r", "media_pts_10r",
    "std_pts_3r", "std_pts_5r",
    "pts_ultima_rodada", "tendencia",
    "regularidade_5r",
]
SCOUTS = ["acc_scout_G", "acc_scout_A", "acc_scout_SG", "acc_scout_GS", "acc_scout_DD"]
CLUBE_ENC = ["clube_enc"]
MANDO = ["mando"]
OPPO_GLOBAL = ["oppo_def_avg5"]
OPPO_GRANULAR = ["oppo_def_pos_avg5", "clube_at_mando_avg5"]
ODDS = ["prob_win", "prob_draw", "prob_loss"]

# Verificador / Avaliador / per-position usam features novas. Trazemos os
# nomes pra cá pra compor as ablações.
PLAYER_HOME_AWAY = ["media_casa_5r", "media_fora_5r"]
MEDIA_EFETIVA = ["media_efetiva"]
EXTRA_SCOUTS = ["acc_scout_FF", "acc_scout_FD", "acc_scout_FT",
                "acc_scout_FS", "acc_scout_shots"]


@dataclass
class Ablation:
    name: str
    feature_cols: list[str]
    matchup_weight: float = 0.0
    apply_round_verifier: bool = False
    position_features: dict | None = None

ABLATIONS: list[Ablation] = [    
    Ablation("01_+scouts",          PLAYER_BASE + SCOUTS),
    Ablation("02_+mando",           PLAYER_BASE + SCOUTS + MANDO),    
    Ablation("03_+oppo_global",     PLAYER_BASE + SCOUTS + MANDO + ODDS + OPPO_GLOBAL),
    Ablation(
        "04_+player_evaluator",
        PLAYER_BASE + SCOUTS + MANDO + ODDS + OPPO_GLOBAL + PLAYER_HOME_AWAY + MEDIA_EFETIVA,
        apply_round_verifier=True,
    ),
    Ablation(
        "05_+per_pos_ATA_MEI",
        PLAYER_BASE + SCOUTS + MANDO + ODDS + OPPO_GLOBAL + PLAYER_HOME_AWAY + MEDIA_EFETIVA + EXTRA_SCOUTS,
        apply_round_verifier=True,
        position_features=POSITION_FEATURES,
    ),
]


def main():
    parser = argparse.ArgumentParser(description="Cartola FC — Feature Ablation")
    parser.add_argument("--inicio", type=int, default=8)
    parser.add_argument("--fim",    type=int, default=14)
    parser.add_argument("--budget", type=float, default=150.0)
    parser.add_argument("--temporada", type=int, default=2026)
    args = parser.parse_args()

    from cartola_collector import TEMPORADA_ATUAL, carregar_dataset
    df_hist, df_partidas, df_odds = carregar_dataset()

    if "temporada" in df_hist.columns:
        if args.temporada not in df_hist["temporada"].unique():
            raise ValueError(
                f"Temporada {args.temporada} não está em df_hist. "
                f"Disponíveis: {sorted(df_hist['temporada'].unique())}"
            )
        df_hist = df_hist[df_hist["temporada"] <= args.temporada].copy()
        if "temporada" in df_partidas.columns:
            df_partidas = df_partidas[df_partidas["temporada"] <= args.temporada].copy()
        if df_odds is not None and not df_odds.empty and "temporada" in df_odds.columns:
            df_odds = df_odds[df_odds["temporada"] <= args.temporada].copy()
        log.info(f"Ablation restrita a dados ≤ temporada {args.temporada}")

    df_mercado_atual = None
    if args.temporada == TEMPORADA_ATUAL:
        df_mercado_atual = pd.read_parquet(DATA_DIR / "mercado_atual.parquet")
    else:
        log.info(
            f"Temporada-alvo {args.temporada} ≠ corrente ({TEMPORADA_ATUAL}); "
            f"ignorando mercado_atual.parquet (preco/media virão de historico_{args.temporada}.parquet)"
        )

    original_cols = list(ctb.FEATURE_COLS)
    summary_rows = []
    try:
        for ab in ABLATIONS:
            log.info(
                f"\n=== {ab.name} | {len(ab.feature_cols)} features | "
                f"matchup_weight={ab.matchup_weight} | "
                f"verifier={ab.apply_round_verifier} | "
                f"per_pos={list((ab.position_features or {}).keys())} ==="
            )
            ctb.FEATURE_COLS = ab.feature_cols
            optimizer_kwargs = {"matchup_weight": ab.matchup_weight}
            treinar_kwargs = {
                "apply_round_verifier": ab.apply_round_verifier,
                "position_features":    ab.position_features,
            }
            resultados = rodar_backtest(
                df_hist=df_hist,
                df_partidas=df_partidas,
                rodada_inicio=args.inicio,
                rodada_fim=args.fim,
                temporada=args.temporada,
                budget=args.budget,
                df_mercado_atual=df_mercado_atual,
                df_odds=df_odds,
                optimizer_kwargs=optimizer_kwargs,
                treinar_kwargs=treinar_kwargs,
                usar_modelo_ml=True,
            )
            if not resultados:
                log.warning(f"{ab.name}: sem resultados")
                continue

            df = pd.DataFrame([{
                "pts_modelo":   r.pts_modelo,
                "pts_baseline": r.pts_baseline_media,
                "pts_teto":     r.pts_teto,
                "eficiencia":   r.eficiencia,
                "mae":          r.mae_predicao,
            } for r in resultados])

            row = {
                "ablation":          ab.name,
                "n_features":        len(ab.feature_cols),
                "matchup_weight":    ab.matchup_weight,
                "round_verifier":    ab.apply_round_verifier,
                "per_pos":           ",".join(str(p) for p in (ab.position_features or {}).keys()) or "-",
                "pts_modelo_mean":   df["pts_modelo"].mean(),
                "pts_modelo_std":    df["pts_modelo"].std(),
                "pts_baseline_mean": df["pts_baseline"].mean(),
                "delta_vs_baseline": df["pts_modelo"].mean() - df["pts_baseline"].mean(),
                "eficiencia_mean":   df["eficiencia"].mean(),
                "ganhou_baseline":   (df["pts_modelo"] > df["pts_baseline"]).mean(),
                "mae_mean":          df["mae"].mean(),
            }
            summary_rows.append(row)
            log.info(
                f"{ab.name}: pts_modelo={row['pts_modelo_mean']:.1f} "
                f"(baseline={row['pts_baseline_mean']:.1f}, "
                f"Δ={row['delta_vs_baseline']:+.1f}), "
                f"efic={row['eficiencia_mean']:.1%}, MAE={row['mae_mean']:.2f}"
            )
    finally:
        ctb.FEATURE_COLS = original_cols

    if not summary_rows:
        log.error("Nenhum resultado coletado.")
        return

    summary_df = pd.DataFrame(summary_rows)
    out_dir = DATA_DIR / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ablation_summary.csv"
    summary_df.to_csv(out, index=False)

    print("\n" + "=" * 110)
    print(f"{'ABLATION SUMMARY':^110}")
    print("=" * 110)
    cols_view = [
        "ablation", "n_features", "round_verifier", "per_pos",
        "pts_modelo_mean", "pts_modelo_std",
        "pts_baseline_mean", "delta_vs_baseline",
        "eficiencia_mean", "ganhou_baseline", "mae_mean",
    ]
    print(summary_df[cols_view].to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
