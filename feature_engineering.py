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
    "scout_GS_5r", "scout_DE_5r"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_features(players_per_round: pd.DataFrame, partidas_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generates temporal and contextual features for each (player, round)
    Uses only past data.
    """
    scout_cols = [c for c in players_per_round.columns if c.startswith("scout_")]
    players_per_round[scout_cols] = players_per_round[scout_cols].fillna(0)
    players_per_round = players_per_round.sort_values(["atleta_id", "rodada"]).copy()

    for window in [3, 5, 10]:
        players_per_round[f"media_pts_{window}r"] = (
            players_per_round.groupby("atleta_id")["pontos"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        players_per_round[f"std_pts_{window}r"] = (
            players_per_round.groupby("atleta_id")["pontos"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
        )

    players_per_round["pts_ultima_rodada"] = players_per_round.groupby("atleta_id")["pontos"].shift(1)

    for col in ["scout_G", "scout_A", "scout_SG", "scout_GS", "scout_DE"]:
        if col in players_per_round.columns:
            players_per_round[f"{col}_5r"] = (
                players_per_round.groupby("atleta_id")[col]
                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            )
    
    players_per_round["regularidade_5r"] = (
        players_per_round.groupby("atleta_id")["jogou"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    players_per_round["preco_lag1"] = players_per_round.groupby("atleta_id")["preco"].shift(1)

    if partidas_df is not None and not partidas_df.empty:
        match_keys = ["temporada", "rodada", "clube_id"]
        players_per_round = players_per_round.merge(
            partidas_df[match_keys + ["mando"]].drop_duplicates(match_keys),
            on=match_keys,
            how="left",
        )
        players_per_round["mando"] = players_per_round["mando"].fillna(0)
    else:
        players_per_round["mando"] = 0

    players_per_round["posicao_enc"] = players_per_round["posicao_id"].astype(int)
    players_per_round["clube_enc"] = LabelEncoder().fit_transform(players_per_round["clube_id"].astype(str))

    return players_per_round
