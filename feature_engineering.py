import logging
import pandas as pd

from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = [
    "posicao_enc", "clube_enc", # "preco_lag1",
    "media_pts_5r", "media_pts_3r", "media_pts_10r",
    "std_pts_5r", "std_pts_3r",
    "regularidade_5r", #"pts_ultima_rodada",
    "mando",
    "scout_G_5r", "scout_A_5r", "scout_SG_5r",
    "scout_GS_5r", "scout_DE_5r",
    "prob_win", "prob_draw", "prob_loss"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_features(
    players_per_round: pd.DataFrame,
    partidas_df: pd.DataFrame | None = None,
    odds_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Generates temporal and contextual features for each (player, round)
    Uses only past data.
    """
    scout_cols = [c for c in players_per_round.columns if c.startswith("scout_")]
    players_per_round = players_per_round.copy()
    players_per_round[scout_cols] = players_per_round[scout_cols].fillna(0)
    features_df = players_per_round.copy()
    features_df = features_df.sort_values(["atleta_id", "rodada"]).copy()

    for window in [3, 5, 10]:
        features_df[f"media_pts_{window}r"] = (
            features_df.groupby("atleta_id")["pontos"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        features_df[f"std_pts_{window}r"] = (
            features_df.groupby("atleta_id")["pontos"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
        )

    features_df["pts_ultima_rodada"] = features_df.groupby("atleta_id")["pontos"].shift(1)

    for col in [feat for feat in FEATURE_COLS if feat.startswith('scout')]:
        if col in features_df.columns:
            features_df[f"{col}_5r"] = (
                features_df.groupby("atleta_id")[col]
                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            )
    
    features_df["regularidade_5r"] = (
        features_df.groupby("atleta_id")["jogou"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    features_df["preco_lag1"] = features_df.groupby("atleta_id")["preco"].shift(1)

    if partidas_df is not None and not partidas_df.empty:
        match_keys = ["temporada", "rodada", "clube_id"]
        features_df = features_df.merge(
            partidas_df[match_keys + ["mando"]].drop_duplicates(match_keys),
            on=match_keys,
            how="left",
        )
        features_df["mando"] = features_df["mando"].fillna(0)
    else:
        features_df["mando"] = 0

    odds_cols = ["prob_win", "prob_draw", "prob_loss"]
    available_odds_cols = [col for col in odds_cols if odds_df is not None and col in odds_df.columns]
    if odds_df is not None and not odds_df.empty and available_odds_cols:
        odds_keys = [col for col in ["temporada", "rodada", "clube_id"] if col in features_df.columns and col in odds_df.columns]
        if "clube_id" in odds_keys:
            features_df = features_df.merge(
                odds_df[odds_keys + available_odds_cols].drop_duplicates(odds_keys),
                on=odds_keys,
                how="left",
            )

    features_df[["prob_win", "prob_draw", "prob_loss"]] = (
        features_df[["prob_win", "prob_draw", "prob_loss"]]
        .fillna({"prob_win": 1/3, "prob_draw": 1/3, "prob_loss": 1/3})
    )

    features_df["posicao_enc"] = features_df["posicao_id"].astype(int)
    features_df["clube_enc"] = LabelEncoder().fit_transform(features_df["clube_id"].astype(str))

    return features_df
