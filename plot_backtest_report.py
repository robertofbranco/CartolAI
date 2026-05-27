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


log = logging.getLogger(__name__)

RENAN_FLICK_POINTS_BY_ROUND = {
    10: 54.15,
    11: 87.46,
    12: 50.65,
    13: 119.39,
    14: 48.34,
    15: 58.30,
    16: 103.84,
    17: 74.82,
}


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


def load_chaves_ligas(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load league bracket max and average winning score by round, when available."""
    chaves_file = Path(data_dir) / "chaves_ligas.parquet"
    source_columns = ["rodada", "pontos"]
    output_columns = [
        "rodada",
        "chaves_ligas_max_pontos",
        "chaves_ligas_media_pontos",
    ]
    if not chaves_file.exists():
        log.warning("Arquivo nao encontrado: %s", chaves_file)
        return pd.DataFrame(columns=output_columns)

    try:
        chaves_df = pd.read_parquet(chaves_file)
    except Exception as exc:
        log.warning("Falha ao carregar %s: %s", chaves_file, exc)
        return pd.DataFrame(columns=output_columns)

    if not set(source_columns).issubset(chaves_df.columns):
        log.warning("%s nao contem as colunas esperadas: %s", chaves_file, source_columns)
        return pd.DataFrame(columns=output_columns)

    chaves_df = chaves_df[source_columns].copy()
    chaves_df["rodada"] = pd.to_numeric(chaves_df["rodada"], errors="coerce")
    chaves_df["pontos"] = pd.to_numeric(chaves_df["pontos"], errors="coerce")
    chaves_df = chaves_df.dropna(subset=["rodada", "pontos"])

    if chaves_df.empty:
        return pd.DataFrame(columns=output_columns)

    return (
        chaves_df.groupby("rodada", as_index=False)["pontos"]
        .agg(
            chaves_ligas_max_pontos="max",
            chaves_ligas_media_pontos="mean",
        )
        .sort_values("rodada")
    )


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
    chaves_ligas = load_chaves_ligas(data_dir)
    chaves_ligas_plot = df[["rodada"]].merge(
        chaves_ligas[["rodada", "chaves_ligas_media_pontos"]],
        on="rodada",
        how="left",
    ).dropna(subset=["chaves_ligas_media_pontos"])
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
    if not chaves_ligas_plot.empty:
        ax1.plot(
            chaves_ligas_plot["rodada"],
            chaves_ligas_plot["chaves_ligas_media_pontos"],
            "-s",
            color="#E64A19",
            label="Media vencedores das chaves",
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
    if not chaves_ligas_plot.empty:
        annotate_point_values(
            ax1,
            chaves_ligas_plot["rodada"],
            chaves_ligas_plot["chaves_ligas_media_pontos"],
            "#BF360C",
            (18, -18),
        )
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


def plot_backtest_report_from_folder(
    output_folder: str | Path,
    output_path: str | Path | None = None,
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
    return plot_backtest_report(resultados_df, mae_posicao_df, output_path, data_dir=data_dir)


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
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Pasta com medias_cartoleiros.parquet e chaves_ligas.parquet",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    plot_backtest_report_from_folder(
        output_folder=args.output_folder,
        output_path=args.output_path,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
