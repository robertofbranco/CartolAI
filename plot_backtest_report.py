"""Generate Cartola backtest report plots from saved or in-memory results."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import pandas as pd

from cartola_data.config import DATA_DIR, POSICAO_NOME

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


log = logging.getLogger(__name__)

RENAN_FLICK_POINTS_BY_ROUND = {
    6: 62.4,
    7: 63.77,
    8: 51.96,
    9: 46.38,
    10: 54.15,
    11: 87.46,
    12: 50.65,
    13: 119.39,
    14: 48.34,
    15: 58.30,
    16: 103.84,
    17: 74.82,
    18: 121.75,
    19: 111.20,
    20: 88.81,
    21: 71.74,
    22: 97.56,
    23: 73.16,
    24: 95.17,
    25: 60.99,
    26: 69.53,
    27: 50.52
}


CHAVES_LIGAS_OUTPUT_COLUMNS = [
    "rodada",
    "chaves_ligas_max_pontos",
    "chaves_ligas_media_pontos",
    "chaves_ligas_mediana_participantes",
    "chaves_ligas_p75_participantes",
    "chaves_ligas_p90_participantes",
    "chaves_ligas_p95_participantes",
    "chaves_ligas_qtd_participantes",
]
CHAVES_LIGAS_PARTICIPANT_SCORE_COLUMNS = [
    ("time_mandante_pontuacao", "mandante"),
    ("time_visitante_pontuacao", "visitante"),
]


def load_medias_cartoleiros(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load Cartola users' average score by round, when available."""
    medias_file = Path(data_dir) / "medias_cartoleiros.parquet"
    columns = ["rodada", "media_cartoleiros"]
    if not medias_file.exists():
        log.warning("Arquivo nao encontrado: %s", medias_file)
        return pd.DataFrame(columns=columns)

    try:
        medias_df = pd.read_parquet(medias_file)
    except Exception as exc:
        log.warning("Falha ao carregar %s: %s", medias_file, exc)
        return pd.DataFrame(columns=columns)

    if not set(columns).issubset(medias_df.columns):
        log.warning("%s nao contem as colunas esperadas: %s", medias_file, columns)
        return pd.DataFrame(columns=columns)

    return (
        medias_df[columns]
        .dropna(subset=["rodada", "media_cartoleiros"])
        .drop_duplicates("rodada")
        .sort_values("rodada")
    )


def _empty_chaves_ligas() -> pd.DataFrame:
    return pd.DataFrame(columns=CHAVES_LIGAS_OUTPUT_COLUMNS)


def _participant_scores_from_chaves(chaves_df: pd.DataFrame) -> pd.DataFrame:
    id_columns = [column for column in ["liga", "rodada", "chave_index"] if column in chaves_df.columns]
    score_columns = [column for column, _ in CHAVES_LIGAS_PARTICIPANT_SCORE_COLUMNS]
    if set(score_columns).issubset(chaves_df.columns):
        participant_scores = chaves_df[id_columns + score_columns].melt(
            id_vars=id_columns,
            value_vars=score_columns,
            var_name="lado",
            value_name="pontos_participante",
        )
        side_names = dict(CHAVES_LIGAS_PARTICIPANT_SCORE_COLUMNS)
        participant_scores["lado"] = participant_scores["lado"].map(side_names)
    elif "pontos" in chaves_df.columns:
        participant_scores = chaves_df[id_columns + ["pontos"]].rename(
            columns={"pontos": "pontos_participante"}
        )
        participant_scores["lado"] = "vencedor"
    else:
        return pd.DataFrame(columns=id_columns + ["lado", "pontos_participante"])

    participant_scores["rodada"] = pd.to_numeric(participant_scores["rodada"], errors="coerce")
    participant_scores["pontos_participante"] = pd.to_numeric(
        participant_scores["pontos_participante"],
        errors="coerce",
    )
    return participant_scores.dropna(subset=["rodada", "pontos_participante"])


def load_chaves_ligas_participant_scores(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load one playoff participant score per row, when available."""
    chaves_file = Path(data_dir) / "chaves_ligas.parquet"
    columns = ["rodada", "pontos_participante", "lado"]
    if not chaves_file.exists():
        log.warning("Arquivo nao encontrado: %s", chaves_file)
        return pd.DataFrame(columns=columns)

    try:
        chaves_df = pd.read_parquet(chaves_file)
    except Exception as exc:
        log.warning("Falha ao carregar %s: %s", chaves_file, exc)
        return pd.DataFrame(columns=columns)

    if "rodada" not in chaves_df.columns:
        log.warning("%s nao contem a coluna esperada: rodada", chaves_file)
        return pd.DataFrame(columns=columns)

    participant_scores = _participant_scores_from_chaves(chaves_df)
    if participant_scores.empty:
        return pd.DataFrame(columns=columns)

    if set(["time_mandante_pontuacao", "time_visitante_pontuacao"]).isdisjoint(chaves_df.columns):
        log.warning(
            "%s contem apenas pontos dos vencedores; benchmark de participantes sera aproximado.",
            chaves_file,
        )

    return participant_scores.sort_values(["rodada", "pontos_participante"]).reset_index(drop=True)


def load_chaves_ligas(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load league bracket winner and participant benchmarks by round, when available."""
    chaves_file = Path(data_dir) / "chaves_ligas.parquet"
    source_columns = ["rodada", "pontos"]
    if not chaves_file.exists():
        log.warning("Arquivo nao encontrado: %s", chaves_file)
        return _empty_chaves_ligas()

    try:
        chaves_df = pd.read_parquet(chaves_file)
    except Exception as exc:
        log.warning("Falha ao carregar %s: %s", chaves_file, exc)
        return _empty_chaves_ligas()

    if not set(source_columns).issubset(chaves_df.columns):
        log.warning("%s nao contem as colunas esperadas: %s", chaves_file, source_columns)
        return _empty_chaves_ligas()

    chaves_df = chaves_df.copy()
    chaves_df["rodada"] = pd.to_numeric(chaves_df["rodada"], errors="coerce")
    chaves_df["pontos"] = pd.to_numeric(chaves_df["pontos"], errors="coerce")
    chaves_df = chaves_df.dropna(subset=["rodada", "pontos"])

    if chaves_df.empty:
        return _empty_chaves_ligas()

    winner_stats = (
        chaves_df.groupby("rodada", as_index=False)["pontos"]
        .agg(
            chaves_ligas_max_pontos="max",
            chaves_ligas_media_pontos="mean",
        )
    )
    score_columns = [column for column, _ in CHAVES_LIGAS_PARTICIPANT_SCORE_COLUMNS]
    participant_scores = (
        _participant_scores_from_chaves(chaves_df)
        if set(score_columns).issubset(chaves_df.columns)
        else pd.DataFrame()
    )
    if participant_scores.empty:
        return winner_stats.reindex(columns=CHAVES_LIGAS_OUTPUT_COLUMNS).sort_values("rodada")

    participant_stats = (
        participant_scores.groupby("rodada")["pontos_participante"]
        .agg(
            chaves_ligas_mediana_participantes="median",
            chaves_ligas_p75_participantes=lambda values: values.quantile(0.75),
            chaves_ligas_p90_participantes=lambda values: values.quantile(0.90),
            chaves_ligas_p95_participantes=lambda values: values.quantile(0.95),
            chaves_ligas_qtd_participantes="count",
        )
        .reset_index()
    )
    return (
        winner_stats.merge(participant_stats, on="rodada", how="left")
        .reindex(columns=CHAVES_LIGAS_OUTPUT_COLUMNS)
        .sort_values("rodada")
    )


def _percentile_rank(score: float, scores: pd.Series) -> float:
    scores = pd.to_numeric(scores, errors="coerce").dropna()
    if scores.empty or pd.isna(score):
        return float("nan")
    return float((scores <= score).mean() * 100)


def annotate_point_values(ax, x_values, y_values, color, offset) -> None:
    for x_value, y_value in zip(x_values, y_values):
        ax.annotate(
            f"{y_value:.1f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=offset,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=color,
            zorder=10,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.8,
                "alpha": 0.92,
            },
        )


def use_integer_x_ticks(ax) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_backtest_report(
    resultados_df: pd.DataFrame,
    mae_posicao_df: pd.DataFrame | None,
    output_path: str | Path,
    data_dir: str | Path = DATA_DIR,
) -> Path:
    """Generate the backtest report image and return its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = resultados_df.copy()
    mae_posicao_df = (
        mae_posicao_df.copy()
        if mae_posicao_df is not None
        else pd.DataFrame(columns=["rodada", "posicao", "mae_predicao"])
    )

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Cartola FC - Backtesting Report", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[1.4, 1.4, 1.0],
        hspace=0.5,
        wspace=0.35,
    )

    rodadas = df["rodada"].values
    medias_cartoleiros = load_medias_cartoleiros(data_dir)
    medias_cartoleiros_plot = df[["rodada"]].merge(
        medias_cartoleiros,
        on="rodada",
        how="left",
    ).dropna(subset=["media_cartoleiros"])
    renan_flick = pd.DataFrame(
        {
            "rodada": list(RENAN_FLICK_POINTS_BY_ROUND.keys()),
            "renan_flick_pontos": list(RENAN_FLICK_POINTS_BY_ROUND.values()),
        }
    )
    renan_flick_plot = df[["rodada"]].merge(
        renan_flick,
        on="rodada",
        how="left",
    ).dropna(subset=["renan_flick_pontos"])

    ax1 = fig.add_subplot(gs[:2, :2])
    ax1.plot(
        rodadas,
        df["pts_teto"],
        "--o",
        color="gold",
        label="Teto (oracle)",
        alpha=0.75,
        linewidth=2,
        markersize=5,
    )
    ax1.plot(
        rodadas,
        df["pts_modelo"],
        "-o",
        color="#2196F3",
        label="Modelo ML",
        linewidth=2,
        markersize=5,
    )
    if not medias_cartoleiros_plot.empty:
        ax1.plot(
            medias_cartoleiros_plot["rodada"],
            medias_cartoleiros_plot["media_cartoleiros"],
            "-o",
            color="#2E7D32",
            label="Media Cartoleiros",
            linewidth=2,
            markersize=5,
            alpha=0.9,
        )
    if not renan_flick_plot.empty:
        ax1.plot(
            renan_flick_plot["rodada"],
            renan_flick_plot["renan_flick_pontos"],
            "-D",
            color="#7B1FA2",
            label="Renan Flick",
            linewidth=2,
            markersize=5,
            alpha=0.9,
        )
    annotate_point_values(ax1, rodadas, df["pts_teto"], "#8A6D00", (0, 12))
    annotate_point_values(ax1, rodadas, df["pts_modelo"], "#0D47A1", (0, -18))
    ax1.set_title("Pontuacao Real por Rodada")
    if not renan_flick_plot.empty:
        annotate_point_values(
            ax1,
            renan_flick_plot["rodada"],
            renan_flick_plot["renan_flick_pontos"],
            "#4A148C",
            (18, 12),
        )
    if not medias_cartoleiros_plot.empty:
        annotate_point_values(
            ax1,
            medias_cartoleiros_plot["rodada"],
            medias_cartoleiros_plot["media_cartoleiros"],
            "#2E7D32",
            (18, -18),
        )
    
    ax1.set_xlabel("Rodada")    
    ax1.set_ylabel("Pontos")
    use_integer_x_ticks(ax1)
    ax1.margins(y=0.15)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    colors = ["#2196F3" if e >= 0.7 else "#FF7043" for e in df["eficiencia"]]
    ax2.bar(rodadas, df["eficiencia"] * 100, color=colors, alpha=0.8)
    ax2.axhline(
        df["eficiencia"].mean() * 100,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Media {df['eficiencia'].mean():.0%}",
    )
    ax2.set_title("Eficiencia vs. Teto")
    ax2.set_xlabel("Rodada")
    ax2.set_ylabel("% do teto alcancado")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(rodadas, df["mae_predicao"], "-o", color="#9C27B0", linewidth=2)
    ax4.axhline(
        df["mae_predicao"].mean(),
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Media {df['mae_predicao'].mean():.2f}",
    )
    ax4.set_title("MAE de Predicao por Rodada")
    ax4.set_xlabel("Rodada")
    ax4.set_ylabel("MAE (pontos)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, :])
    if not mae_posicao_df.empty:
        posicao_order = list(POSICAO_NOME.values())
        for posicao in posicao_order:
            position_mae = mae_posicao_df[mae_posicao_df["posicao"] == posicao]
            if position_mae.empty:
                continue
            ax5.plot(
                position_mae["rodada"],
                position_mae["mae_predicao"],
                "-o",
                label=posicao,
                linewidth=1.8,
                markersize=4,
            )
        ax5.set_title("MAE por Posicao")
        ax5.set_xlabel("Rodada")
        ax5.set_ylabel("MAE (pontos)")
        ax5.legend(fontsize=8, ncol=min(6, max(1, mae_posicao_df["posicao"].nunique())))
        ax5.grid(True, alpha=0.3)
    else:
        ax5.set_axis_off()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Grafico salvo em %s", output_path)
    return output_path


def plot_playoff_benchmark(
    resultados_df: pd.DataFrame,
    output_path: str | Path,
    data_dir: str | Path = DATA_DIR,
) -> Path | None:
    """Generate a playoff benchmark plot comparing model score to bracket percentiles."""
    required_columns = ["rodada", "pts_modelo"]
    if not set(required_columns).issubset(resultados_df.columns):
        log.warning("Resultados nao contem as colunas esperadas: %s", required_columns)
        return None

    participant_scores = load_chaves_ligas_participant_scores(data_dir)
    if participant_scores.empty:
        log.warning("Sem pontuacoes de chaves para gerar benchmark de playoffs.")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_scores = resultados_df[required_columns].copy()
    model_scores["rodada"] = pd.to_numeric(model_scores["rodada"], errors="coerce")
    model_scores["pts_modelo"] = pd.to_numeric(model_scores["pts_modelo"], errors="coerce")
    model_scores = model_scores.dropna(subset=required_columns)

    score_stats = (
        participant_scores.groupby("rodada")["pontos_participante"]
        .agg(
            mediana="median",
            p75=lambda values: values.quantile(0.75),
            p90=lambda values: values.quantile(0.90),
            p95=lambda values: values.quantile(0.95),
            maximo="max",
            participantes="count",
        )
        .reset_index()
    )
    plot_df = model_scores.merge(score_stats, on="rodada", how="inner").sort_values("rodada")
    if plot_df.empty:
        log.warning("Sem rodadas em comum entre backtest e chaves de playoffs.")
        return None

    scores_by_round = {
        rodada: round_scores["pontos_participante"]
        for rodada, round_scores in participant_scores.groupby("rodada")
    }
    plot_df["percentil_modelo"] = plot_df.apply(
        lambda row: _percentile_rank(
            row["pts_modelo"],
            scores_by_round.get(row["rodada"], pd.Series(dtype=float)),
        ),
        axis=1,
    )

    winner_stats = load_chaves_ligas(data_dir)[["rodada", "chaves_ligas_media_pontos"]]
    plot_df = plot_df.merge(winner_stats, on="rodada", how="left")

    sides = set(participant_scores["lado"].dropna().unique())
    scope_label = (
        "todos os participantes das chaves"
        if sides - {"vencedor"}
        else "vencedores das chaves (arquivo antigo)"
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.35},
    )
    fig.suptitle("Benchmark de Playoffs", fontsize=14, fontweight="bold")

    x_values = plot_df["rodada"].astype(float).to_numpy()
    ax1.fill_between(
        x_values,
        plot_df["mediana"].astype(float).to_numpy(),
        plot_df["p90"].astype(float).to_numpy(),
        color="#FFB74D",
        alpha=0.18,
        label="Faixa P50-P90",
    )
    ax1.plot(
        x_values,
        plot_df["pts_modelo"],
        "-o",
        color="#1565C0",
        linewidth=2.5,
        markersize=6,
        label="Modelo ML",
    )
    ax1.plot(x_values, plot_df["mediana"], "-o", color="#455A64", label="Mediana", linewidth=2)
    ax1.plot(x_values, plot_df["p90"], "-s", color="#EF6C00", label="P90", linewidth=2)
    ax1.plot(x_values, plot_df["p95"], "-^", color="#C62828", label="P95", linewidth=2)
    ax1.plot(x_values, plot_df["maximo"], "--", color="#212121", label="Vencedor/max", linewidth=1.8)
    winners_plot = plot_df.dropna(subset=["chaves_ligas_media_pontos"])
    if not winners_plot.empty:
        ax1.plot(
            winners_plot["rodada"],
            winners_plot["chaves_ligas_media_pontos"],
            ":",
            color="#6A1B9A",
            label="Media vencedores",
            linewidth=2,
        )
    annotate_point_values(ax1, x_values, plot_df["pts_modelo"], "#0D47A1", (0, -18))
    ax1.set_title(f"Pontuacao por rodada vs. {scope_label}")
    ax1.set_xlabel("Rodada")
    ax1.set_ylabel("Pontos")
    use_integer_x_ticks(ax1)
    ax1.margins(y=0.15)
    ax1.legend(fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    bar_colors = [
        "#2E7D32" if percentile >= 75 else "#F9A825" if percentile >= 50 else "#D84315"
        for percentile in plot_df["percentil_modelo"]
    ]
    ax2.bar(x_values, plot_df["percentil_modelo"], color=bar_colors, alpha=0.85)
    for threshold, color in [(50, "#616161"), (75, "#33691E"), (90, "#E65100"), (95, "#B71C1C")]:
        ax2.axhline(threshold, color=color, linestyle="--", linewidth=1, alpha=0.75)
        ax2.text(x_values[-1] + 0.15, threshold, f"P{threshold}", color=color, va="center", fontsize=8)
    annotate_point_values(ax2, x_values, plot_df["percentil_modelo"], "#263238", (0, 10))
    ax2.set_title("Percentil do modelo na rodada")
    ax2.set_xlabel("Rodada")
    ax2.set_ylabel("% participantes batidos")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Benchmark de playoffs salvo em %s", output_path)
    return output_path


def plot_backtest_report_from_folder(
    output_folder: str | Path,
    output_path: str | Path | None = None,
    benchmark_output_path: str | Path | None = None,
    data_dir: str | Path = DATA_DIR,
) -> Path:
    """Generate a report image from CSVs in a backtest output folder."""
    output_folder = Path(output_folder)
    resultados_path = output_folder / "backtest_resultados.csv"
    mae_posicao_path = output_folder / "backtest_mae_por_posicao.csv"

    if not resultados_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {resultados_path}")

    resultados_df = pd.read_csv(resultados_path)
    if mae_posicao_path.exists():
        mae_posicao_df = pd.read_csv(mae_posicao_path)
    else:
        log.warning("Arquivo nao encontrado: %s", mae_posicao_path)
        mae_posicao_df = pd.DataFrame(columns=["rodada", "posicao", "mae_predicao"])

    output_path = Path(output_path) if output_path is not None else output_folder / "backtest_report.png"
    report_path = plot_backtest_report(resultados_df, mae_posicao_df, output_path, data_dir=data_dir)
    benchmark_output_path = (
        Path(benchmark_output_path)
        if benchmark_output_path is not None
        else output_folder / "backtest_playoff_benchmark.png"
    )
    plot_playoff_benchmark(resultados_df, benchmark_output_path, data_dir=data_dir)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Cartola backtest report plot")
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("results"),
        help="Pasta com backtest_resultados.csv e backtest_mae_por_posicao.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Caminho da imagem gerada; padrao: <output-folder>/backtest_report.png",
    )
    parser.add_argument(
        "--benchmark-output-path",
        type=Path,
        default=None,
        help=(
            "Caminho do benchmark de playoffs; padrao: "
            "<output-folder>/backtest_playoff_benchmark.png"
        ),
    )    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    plot_backtest_report_from_folder(
        output_folder=args.output_folder,
        output_path=args.output_path,
        benchmark_output_path=args.benchmark_output_path,
        data_dir=DATA_DIR,
    )


if __name__ == "__main__":
    main()
