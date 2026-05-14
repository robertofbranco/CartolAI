import logging
import pandas as pd

from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = [
    "posicao_enc", "clube_enc", # "preco_lag1",
    "media_pts_5r", #"media_pts_3r", "media_pts_10r",
    "std_pts_5r", #"std_pts_3r",
    "regularidade_5r", #"pts_ultima_rodada",
    "mando",
    "scout_G_5r", "scout_A_5r", "scout_SG_5r",
    "scout_GS_5r", "scout_DD_5r",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, partidas_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generates temporal and contextual features for each (player, round)
    Uses only past data.
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

    for col in ["scout_G", "scout_A", "scout_SG", "scout_GS", "scout_DD"]:
        if col in df.columns:
            df[f"acc_{col}"] = (
                df.groupby("atleta_id")[col]
                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            )

    df["jogou"] = (df["pontos"] > 0).astype(int)
    df["regularidade_5r"] = (
        df.groupby("atleta_id")["jogou"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    df["preco_lag1"] = df.groupby("atleta_id")["preco"].shift(1)

    if partidas_df is not None and not partidas_df.empty:
        df = df.merge(
            partidas_df[["rodada", "clube_id", "mando"]],
            on=["rodada", "clube_id"],
            how="left",
        )
        df["mando"] = df["mando"].fillna(0)
    else:
        df["mando"] = 0

    df["posicao_enc"] = df["posicao_id"].astype(int)
    df["clube_enc"] = LabelEncoder().fit_transform(df["clube_id"].astype(str))

    return df
