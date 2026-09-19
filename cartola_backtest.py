"""
Cartola FC - Backtesting Engine
================================
Simula o pipeline rodada a rodada em dados históricos para medir:
  - MAE de predição por rodada
  - Jogadores selecionados
  - Pontuacao alcancada  

Uso:
    python cartola_backtest.py --inicio 5 --fim 20 --budget 140

Requisitos: mesmo requirements.txt do pipeline principal
"""

import logging
import warnings
import argparse
import sys
import pandas as pd

from cartola_data.config import CAPTAIN_BONUS, CURRENT_SEASON, DATA_DIR, TUNING
from cartola_data.datasets import read_datasets
from feature_engineering import build_features
from pathlib import Path
from dataclasses import dataclass, field
from plot_backtest_report import (
    load_chaves_ligas as _load_chaves_ligas,
    load_medias_cartoleiros as _load_medias_cartoleiros,
    plot_backtest_report,
    plot_playoff_benchmark,
)

from cartola_team_builder import (
    assign_captain,
    apply_reserve_substitutions,
    build_target_round_market_features,
    build_team,
    lineup_output_table,
    score_with_captain_bonus,
    CAPTAIN_COL
)
from cartola_model_training import (
    DEFAULT_MODEL_STRATEGY,
    available_model_strategies,
    mean_mae_from_models,
    train_models_by_position,
)

from cartola_data.config import (
    CAPTAIN_BONUS,    
    DATA_DIR,
    FORMATION,
    POSICAO_NOME,    
    TUNING
)

CSV_PLAYERS_COLUMNS = [
    "rodada",
    "atleta_id",
    "apelido",
    "posicao",
    "clube",
    "clube adversario",
    "mando",
    "pontos_com_bonus",
    "pontos",
    "pontos_previstos",    
    "capitao",
    "reserva",
    "reserva_de_luxo",
    "substituiu_apelido"
]


def configure_console_output() -> None:
    """Avoid UnicodeEncodeError on Windows consoles with legacy encodings."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


configure_console_output()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
MARKET_VALUE_COLS = ["preco", "media"]

# ──────────────────────────────────────────────
# ESTRUTURA DE RESULTADO
# ──────────────────────────────────────────────

@dataclass
class ResultadoRodada:
    rodada: int
    mae_predicao: float                  # erro médio de predição (pontos)
    pts_modelo: float                    # pontos reais do time escolhido pelo modelo
    pts_teto: float                      # pontos do melhor time possível (oracle, a posteriori)
    eficiencia: float                    # pts_modelo / pts_teto  (0–1)
    budget_usado: float                  # cartoletas gastas
    time_escalado: pd.DataFrame = field(repr=False)
    mae_por_posicao: dict[str, float] = field(default_factory=dict)


def build_clubes_lookup(df_players_per_round: pd.DataFrame) -> dict:
    """Return clube_id -> best available club name, preferring full names."""
    if not {"clube_id", "clube_nome"}.issubset(df_players_per_round.columns):
        return {}

    clubes_df = df_players_per_round.dropna(subset=["clube_id", "clube_nome"])
    clubes_lookup = {}

    for clube_id, clube_rows in clubes_df.groupby("clube_id"):
        nomes = [
            str(nome).strip()
            for nome in clube_rows["clube_nome"].dropna().unique()
            if str(nome).strip()
        ]
        if not nomes:
            continue

        clubes_lookup[clube_id] = max(nomes, key=len)

    return clubes_lookup


def load_medias_cartoleiros() -> pd.DataFrame:
    """Backward-compatible wrapper for report plot overlay data."""
    return _load_medias_cartoleiros(DATA_DIR)


def load_chaves_ligas() -> pd.DataFrame:
    """Backward-compatible wrapper for report plot overlay data."""
    return _load_chaves_ligas(DATA_DIR)


def mae_por_posicao_from_models(models_by_position: dict) -> dict[str, float]:
    """Return validation MAE by position name from trained position models."""
    mae_por_posicao = {}
    for posicao_id, model_info in models_by_position.items():
        posicao = POSICAO_NOME.get(int(posicao_id), str(posicao_id))
        mae_por_posicao[posicao] = float(model_info["mae"])
    return mae_por_posicao


# ──────────────────────────────────────────────
# ORACLE (teto teórico)
# ──────────────────────────────────────────────

def calcular_teto(df_rodada_real: pd.DataFrame, formation: dict) -> float:
    """
    Monta o melhor time possível com pontuação REAL da rodada (oracle).
    Usado como denominador da eficiência.
    """
    df = df_rodada_real.sort_values("pontos", ascending=False)

    selected_players = []

    for position, n_players in formation.items():        
        position_pool = df[df["posicao_id"] == position].copy()
        chosen = []

        for _, player in position_pool.iterrows():            
            if len(chosen) >= n_players:
                break

            chosen.append(player)

        selected_players.extend(chosen)

    team_df = pd.DataFrame(selected_players)
    team_df = team_df.sort_values(["posicao_id", "pontos"], ascending=[True, False])
    team_df = assign_captain(team_df, score_column="pontos")

    return score_with_captain_bonus(team_df, points_column="pontos")


def latest_features_before_round(
    df_feat: pd.DataFrame,
    season: int,
    rodada_alvo: int,
    feature_cols: list[str],
    market_columns: pd.Index,
) -> pd.DataFrame:
    """Return the latest available pre-round feature row for each athlete."""
    columns = ["atleta_id"] + [col for col in feature_cols if col not in market_columns]
    previous_features = df_feat.loc[
        (df_feat["temporada"] < season)
        | (
            (df_feat["temporada"] == season)
            & (df_feat["rodada"] < rodada_alvo)
        )
    ].copy()
    previous_features["season_sort"] = (previous_features["temporada"] == season).astype(int)
    sort_columns = ["season_sort", "temporada", "rodada"]

    if previous_features.empty:
        return pd.DataFrame(columns=columns)

    return (
        previous_features
        .sort_values(sort_columns)
        [columns]
        .drop_duplicates("atleta_id", keep="last")
    )


def latest_market_values_before_round(
    df_players_per_round: pd.DataFrame,
    season: int,
    rodada_alvo: int,
) -> pd.DataFrame:
    """Return latest nonzero market values before the simulated round."""
    required_cols = ["atleta_id", "posicao_id"] + MARKET_VALUE_COLS
    available_cols = [col for col in required_cols if col in df_players_per_round.columns]
    history = df_players_per_round.loc[
        (df_players_per_round["temporada"] < season)
        | (
            (df_players_per_round["temporada"] == season)
            & (df_players_per_round["rodada"] < rodada_alvo)
        ),
        available_cols + ["temporada", "rodada"],
    ].copy()

    if history.empty:
        return pd.DataFrame(columns=required_cols)

    history = history[history["preco"].fillna(0) > 0]
    if history.empty:
        return pd.DataFrame(columns=required_cols)

    history["season_sort"] = (history["temporada"] == season).astype(int)
    return (
        history
        .sort_values(["season_sort", "temporada", "rodada"])
        [available_cols]
        .drop_duplicates("atleta_id", keep="last")
    )


def current_market_values() -> pd.DataFrame:
    market_file = DATA_DIR / "mercado_atual.parquet"
    if not market_file.exists():
        return pd.DataFrame(columns=["atleta_id"] + MARKET_VALUE_COLS)

    try:
        market_df = pd.read_parquet(market_file)
    except Exception as exc:
        log.warning(f"Falha ao carregar {market_file}: {exc}")
        return pd.DataFrame(columns=["atleta_id"] + MARKET_VALUE_COLS)

    columns = [col for col in ["atleta_id"] + MARKET_VALUE_COLS if col in market_df.columns]
    return market_df[columns].drop_duplicates("atleta_id")


def fill_missing_market_values(
    df_market: pd.DataFrame,
    df_players_per_round: pd.DataFrame,
    season: int,
    rodada_alvo: int,
) -> pd.DataFrame:
    """Replace zero price/media placeholders in the simulated market."""
    df_market = df_market.copy()
    missing_market_mask = df_market["preco"].fillna(0) <= 0
    if not missing_market_mask.any():
        return df_market

    missing_before = int(missing_market_mask.sum())

    historical_values = latest_market_values_before_round(
        df_players_per_round=df_players_per_round,
        season=season,
        rodada_alvo=rodada_alvo,
    )
    if not historical_values.empty:
        df_market = df_market.merge(
            historical_values[["atleta_id"] + MARKET_VALUE_COLS],
            on="atleta_id",
            how="left",
            suffixes=("", "_hist"),
        )
        for col in MARKET_VALUE_COLS:
            fallback_col = f"{col}_hist"
            if fallback_col in df_market.columns:
                df_market.loc[missing_market_mask, col] = df_market.loc[
                    missing_market_mask, col
                ].where(
                    df_market.loc[missing_market_mask, fallback_col].isna(),
                    df_market.loc[missing_market_mask, fallback_col],
                )
                df_market = df_market.drop(columns=fallback_col)

    missing_market_mask = df_market["preco"].fillna(0) <= 0
    market_values = current_market_values()
    if missing_market_mask.any() and not market_values.empty:
        df_market = df_market.merge(
            market_values,
            on="atleta_id",
            how="left",
            suffixes=("", "_current"),
        )
        for col in MARKET_VALUE_COLS:
            fallback_col = f"{col}_current"
            if fallback_col in df_market.columns:
                df_market.loc[missing_market_mask, col] = df_market.loc[
                    missing_market_mask, col
                ].where(
                    df_market.loc[missing_market_mask, fallback_col].isna(),
                    df_market.loc[missing_market_mask, fallback_col],
                )
                df_market = df_market.drop(columns=fallback_col)

    missing_market_mask = df_market["preco"].fillna(0) <= 0
    if missing_market_mask.any():
        historical_nonzero = df_players_per_round[df_players_per_round["preco"].fillna(0) > 0]
        position_medians = historical_nonzero.groupby("posicao_id")[MARKET_VALUE_COLS].median()
        for col in MARKET_VALUE_COLS:
            df_market.loc[missing_market_mask, col] = df_market.loc[
                missing_market_mask, "posicao_id"
            ].map(position_medians[col])

    for col in MARKET_VALUE_COLS:
        df_market[col] = pd.to_numeric(df_market[col], errors="coerce").fillna(0)

    missing_after = int((df_market["preco"].fillna(0) <= 0).sum())
    if missing_after != missing_before:
        log.info(
            f"Valores de mercado reparados: {missing_before - missing_after}/"
            f"{missing_before} atletas com preco zerado."
        )

    return df_market


# ──────────────────────────────────────────────
# ENGINE DE BACKTESTING
# ──────────────────────────────────────────────

def rodar_backtest(
    rodada_inicio: int,
    rodada_fim: int,
    formation: dict = FORMATION,
    season: int = 2026,
    tuning: dict = TUNING,
    strategy: str = DEFAULT_MODEL_STRATEGY,
    output_folder: str | Path | None = RESULTS_DIR,
) -> list[ResultadoRodada]:
    """
    Para cada rodada no intervalo [rodada_inicio, rodada_fim]:
      1. Treina o modelo com tudo que veio ANTES dessa rodada
      2. Usa o mercado daquela rodada para montar o time
      3. Compara com os pontos REAIS da rodada (que o modelo nunca viu)
    """
    output_dir = Path(output_folder) if output_folder is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    df_players_per_round, df_matches, df_odds = read_datasets()

    resultados = []

    # Lookup: clube_id -> nome (prefere nome completo quando o historico tambem
    # tem abreviacoes como FLA/PAL/VAS para temporadas mais recentes).
    clubes_lookup = build_clubes_lookup(df_players_per_round)

    # Lookup: (rodada, clube_id) -> clube_adversario_id. Coluna pode não existir em
    # parquets antigos — quando ausente, o adversário cai para "?" no log.
    if not df_matches.empty and "clube_adversario_id" in df_matches.columns:
        adv_keys = ["temporada", "rodada", "clube_id"]
        adv_lookup = (
            df_matches.dropna(subset=["clube_adversario_id"])
            .drop_duplicates(adv_keys)
            .set_index(adv_keys)["clube_adversario_id"]
            .to_dict()
        )
    else:
        adv_lookup = {}

    for rodada_alvo in range(rodada_inicio, rodada_fim + 1):
        log.info(f"── Backtesting rodada {rodada_alvo} ──")

        # Dados disponíveis até esta rodada (sem ver o futuro)
        df_players_per_round_training = df_players_per_round.loc[
            (df_players_per_round["temporada"] < season)
            | (
                (df_players_per_round["temporada"] == season)
                & (df_players_per_round["rodada"] < rodada_alvo)
            )
        ].copy()

        df_matches_training = df_matches.loc[
            (df_matches["temporada"] < season)
            | (
                (df_matches["temporada"] == season)
                & (df_matches["rodada"] < rodada_alvo)
            )
        ].copy()

        df_odds_training = df_odds.loc[
            (df_odds["temporada"] < season)
            | (
                (df_odds["temporada"] == season)
                & (df_odds["rodada"] < rodada_alvo)
            )
        ].copy()
        
        df_feat = build_features(df_players_per_round_training, df_matches_training, df_odds_training)

        try:
            models_by_pos = train_models_by_position(
                df_feat,
                round_limit=rodada_alvo,
                season=season,
                tuning=tuning,
                strategy=strategy,
            )
        except ValueError as e:
            log.warning(f"Rodada {rodada_alvo}: {e}")
            continue
        mae = mean_mae_from_models(models_by_pos)
        mae_por_posicao = mae_por_posicao_from_models(models_by_pos)

        # Simular mercado: snapshot dos jogadores na rodada alvo
        # (usamos os dados daquela rodada como proxy de mercado)
        df_rodada_real = df_players_per_round.loc[(df_players_per_round["temporada"] == season)
                                                  & (df_players_per_round["rodada"] == rodada_alvo)].copy()
        if df_rodada_real.empty:
            log.warning(f"Rodada {rodada_alvo}: sem dados reais, pulando.")
            continue

        # Simulate the market row available at lock time: player market data
        # from the target round plus lagged features through the previous round
        # and target-round fixture/odds context.
        df_mercado_sim = build_target_round_market_features(
            df_players_per_round=df_players_per_round,
            df_matches=df_matches,
            df_odds=df_odds,
            df_market=df_rodada_real,
            season=season,
            rodada_alvo=rodada_alvo,
        )

        # Montar time com o modelo
        try:
            time_modelo = build_team(
                df_mercado_sim,
                models_by_pos,
                formation,
                include_reserves=True,
            )
            time_modelo = apply_reserve_substitutions(time_modelo, df_rodada_real)
        except Exception as e:
            log.warning(f"Rodada {rodada_alvo}: otimização falhou — {e}")
            continue

        # Pontuação REAL dos jogadores escolhidos pelo modelo
        reais = df_rodada_real.set_index("atleta_id")["pontos"].to_dict()
        time_modelo["pontos_real"] = time_modelo["atleta_id"].map(reais).fillna(0.0)
        time_modelo["multiplicador_capitao"] = 1.0
        if CAPTAIN_COL in time_modelo.columns:
            captain_mask = time_modelo[CAPTAIN_COL].fillna(False).astype(bool)
            time_modelo.loc[captain_mask, "multiplicador_capitao"] = CAPTAIN_BONUS
        time_modelo["pontos_com_bonus"] = (
            time_modelo["pontos_real"] * time_modelo["multiplicador_capitao"]
        )
        pts_modelo = score_with_captain_bonus(time_modelo, points_column="pontos_real")

        # Anotar posição (string), adversário e pontos reais no lineup para o log/CSV
        time_modelo["posicao"] = time_modelo["posicao_id"].map(POSICAO_NOME)
        time_modelo["clube"] = (
            time_modelo["clube_id"]
            .map(clubes_lookup)
            .fillna(time_modelo.get("clube_nome", ""))
            .fillna("?")
        )
        time_modelo["adversario"] = (
            time_modelo["clube_id"]
            .map(
                lambda cid: adv_lookup.get((season, rodada_alvo, cid))
                if "temporada" in df_matches.columns
                else adv_lookup.get((rodada_alvo, cid))
            )
            .map(clubes_lookup)
            .fillna("?")
        )
        log.info(f"Time escalado R{rodada_alvo}:")
        for _, p in time_modelo.iterrows():
            apelido = (p["apelido"] or "")[:20]
            clube = (p.get("clube") or "")[:16]
            adv = (p["adversario"] or "")[:16]
            reserva = " RES" if bool(p.get("reserva", False)) else ""
            luxo = " LUX" if bool(p.get("reserva_de_luxo", False)) else ""
            capitao = " CAP" if bool(p.get(CAPTAIN_COL, False)) else ""
            log.info(
                f"  {p['posicao']:<3} {apelido:<20} {clube:<16} vs {adv:<16}"
                f"{reserva:<4}{luxo:<4}{capitao:<4} "
                f"avg={p['media']:>5.2f} preco={p['preco']:>5.1f} "
                f"pts={p['pontos_real']:>5.1f} final={p['pontos_com_bonus']:>5.1f}"
            )

        # Teto (oracle)
        pts_teto = calcular_teto(df_rodada_real, formation)

        eficiencia = pts_modelo / pts_teto if pts_teto > 0 else 0

        resultado = ResultadoRodada(
            rodada=rodada_alvo,
            mae_predicao=mae,
            pts_modelo=pts_modelo,
            pts_teto=pts_teto,
            eficiencia=eficiencia,
            budget_usado=time_modelo["preco"].sum(),
            time_escalado=time_modelo,
            mae_por_posicao=mae_por_posicao,
        )
        resultados.append(resultado)
        log.info(
            f"  Modelo: {pts_modelo:.1f} |  "
            f"Teto: {pts_teto:.1f} | Eficiência: {eficiencia:.1%} | MAE: {mae:.2f}"
        )

    if output_dir is not None:
        if resultados:
            gerar_relatorio(resultados, output_dir=output_dir)
        else:
            log.warning(f"Nenhum resultado gerado; relatorio nao foi salvo em {output_dir}.")

    return resultados


# ──────────────────────────────────────────────
# RELATÓRIO
# ──────────────────────────────────────────────

def gerar_relatorio(resultados: list[ResultadoRodada], output_dir: str | Path = RESULTS_DIR):
    """Gera DataFrame resumo + gráficos do backtesting."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([{
        "rodada":            r.rodada,
        "mae_predicao":      r.mae_predicao,
        "pts_modelo":        r.pts_modelo,
        "pts_teto":          r.pts_teto,
        "eficiencia":        r.eficiencia,
        "budget_usado":      r.budget_usado
    } for r in resultados])

    mae_posicao_df = pd.DataFrame([
        {
            "rodada": r.rodada,
            "posicao": posicao,
            "mae_predicao": mae_posicao,
        }
        for r in resultados
        for posicao, mae_posicao in r.mae_por_posicao.items()
    ])

    # ── Sumário no terminal ──
    print("\n" + "="*70)
    print(f"{'BACKTEST SUMMARY':^70}")
    print("="*70)
    print(f"  Rodadas testadas:        {len(df)}")
    print(f"  Pts modelo  (média):     {df['pts_modelo'].mean():.2f}  ±{df['pts_modelo'].std():.2f}")
    print(f"  Pts teto    (média):     {df['pts_teto'].mean():.2f}")
    print(f"  Eficiência  (média):     {df['eficiencia'].mean():.1%}")
    print(f"  MAE médio:               {df['mae_predicao'].mean():.3f} pts")
    print(f"  Melhor rodada:           R{df.loc[df['pts_modelo'].idxmax(), 'rodada']} "
          f"({df['pts_modelo'].max():.1f} pts)")
    print(f"  Pior rodada:             R{df.loc[df['pts_modelo'].idxmin(), 'rodada']} "
          f"({df['pts_modelo'].min():.1f} pts)")
    print("="*70)

    # ── Gráficos ──
    plot_backtest_report(
        df,
        mae_posicao_df,
        output_dir / "backtest_report.png",
        data_dir=DATA_DIR,
    )
    plot_playoff_benchmark(
        df,
        output_dir / "backtest_playoff_benchmark.png",
        data_dir=DATA_DIR,
    )

    # ── Salvar CSV ──
    df.to_csv(output_dir / "backtest_resultados.csv", index=False)
    log.info(f"Resultados salvos em {output_dir / 'backtest_resultados.csv'}")

    mae_posicao_path = output_dir / "backtest_mae_por_posicao.csv"
    mae_posicao_df.to_csv(mae_posicao_path, index=False)
    log.info(f"MAE por posição salvo em {mae_posicao_path}")

    selected_players_rows = []
    for resultado in resultados:
        time_escalado = resultado.time_escalado.copy()
        if time_escalado.empty:
            continue

        time_escalado["rodada"] = resultado.rodada
        lineup_table = lineup_output_table(time_escalado)
        if "pontos_real" in time_escalado.columns:
            lineup_table["pontos"] = time_escalado["pontos_real"].to_numpy()
        elif "pontos" in time_escalado.columns:
            lineup_table["pontos"] = time_escalado["pontos"].to_numpy()
        if "pontos_com_bonus" in time_escalado.columns:
            lineup_table["pontos_com_bonus"] = time_escalado["pontos_com_bonus"].to_numpy()
        for column in ["substituiu_atleta_id", "substituiu_apelido"]:
            if column in time_escalado.columns:
                lineup_table[column] = time_escalado[column].to_numpy()
        lineup_table["_posicao_id"] = time_escalado["posicao_id"].to_numpy()
        selected_players_rows.append(lineup_table)

    selected_players_columns = CSV_PLAYERS_COLUMNS
    if selected_players_rows:
        selected_players_df = pd.concat(selected_players_rows, ignore_index=True)
        selected_players_df = selected_players_df.sort_values(
            ["rodada", "_posicao_id"],
            ascending=[True, True],
        )
        for column in selected_players_columns:
            if column not in selected_players_df.columns:
                selected_players_df[column] = ""
        selected_players_df = selected_players_df[selected_players_columns]
    else:
        selected_players_df = pd.DataFrame(columns=selected_players_columns)

    selected_players_path = output_dir / "backtest_time_escalado.csv"
    selected_players_df.to_csv(selected_players_path, index=False)
    log.info(f"Time escalado salvo em {selected_players_path}")

    return df


# ──────────────────────────────────────────────
# EXECUÇÃO
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cartola FC — Backtesting Engine")
    parser.add_argument("--temporada",  type=int, default=CURRENT_SEASON,    help="Primeira rodada a testar (mín. 6)")
    parser.add_argument("--inicio",     type=int, default=15,    help="Primeira rodada a testar (mín. 6)")
    parser.add_argument("--fim",        type=int, default=27,   help="Última rodada a testar")
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=RESULTS_DIR,
        help="Pasta onde salvar os arquivos gerados pelo backtest",
    )
    parser.add_argument(
        "--model-strategy",
        choices=available_model_strategies(),
        default=DEFAULT_MODEL_STRATEGY,
        help="Metodo de ML usado para treinar os modelos por posicao",
    )
    #parser.add_argument("--budget",  type=float, default=140.0)
    args = parser.parse_args()    

    # Rodar backtest
    resultados = rodar_backtest(
        season=args.temporada,
        rodada_inicio=args.inicio,
        rodada_fim=args.fim,
        strategy=args.model_strategy,
        output_folder=args.output_folder,
    )

    if not resultados:
        log.error("Nenhum resultado gerado. Verifique os dados históricos.")
        return


if __name__ == "__main__":
    main()
