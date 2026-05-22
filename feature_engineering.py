import logging
import pandas as pd

from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = [
    #"clube_enc", 
    #"preco_lag1",
    "media_pts_5r", "media_pts_3r", "media_pts_10r",
    "avg_pts_casa_5r", "avg_pts_fora_5r",
    "std_pts_5r", "std_pts_3r",
    "regularidade_5r", "pts_ultima_rodada",
    "mando",
    "scout_G_5r", "scout_A_5r", "scout_SG_5r",
    "scout_GS_5r", "scout_DE_5r", "scout_FD_5r",
    "scout_FF_5r", "scout_FT_5r",
    "prob_win", "prob_draw", "prob_loss",
    "gols_feitos_clube_5r", "gols_sofridos_clube_5r",
    "gols_feitos_adv_5r", "gols_sofridos_adv_5r",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _add_home_away_points_averages(
    features_df: pd.DataFrame,
    temporal_group_cols: list[str],
    window: int = 5,
) -> pd.DataFrame:
    """
    Add each player's prior home/away point averages.

    For every row, both columns describe what was known before that round:
    the average of the player's previous `window` home matches and previous
    `window` away matches.
    """
    features_df = features_df.copy()
    casa_col = f"avg_pts_casa_{window}r"
    fora_col = f"avg_pts_fora_{window}r"
    features_df[casa_col] = float("nan")
    features_df[fora_col] = float("nan")

    for _, group in features_df.groupby(temporal_group_cols, sort=False):
        home_points = []
        away_points = []

        for index, row in group.sort_values("rodada").iterrows():
            if home_points:
                features_df.at[index, casa_col] = sum(home_points[-window:]) / min(
                    len(home_points),
                    window,
                )
            if away_points:
                features_df.at[index, fora_col] = sum(away_points[-window:]) / min(
                    len(away_points),
                    window,
                )

            points = row.get("pontos")
            if pd.isna(points):
                continue

            if row.get("mando") == 1:
                home_points.append(float(points))
            elif row.get("mando") == -1:
                away_points.append(float(points))

    return features_df


def build_features(
    players_per_round: pd.DataFrame,
    partidas_df: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generates temporal and contextual features for each (player, round)
    Uses only past data.
    """
    scout_cols = [c for c in players_per_round.columns if c.startswith("scout_")]
    players_per_round = players_per_round.copy()
    players_per_round[scout_cols] = players_per_round[scout_cols].fillna(0)
    features_df = players_per_round.copy()
    temporal_group_cols = ["temporada", "atleta_id"] if "temporada" in features_df.columns else ["atleta_id"]
    sort_cols = temporal_group_cols + ["rodada"]
    features_df = features_df.sort_values(sort_cols).copy()

    match_keys = ["temporada", "rodada", "clube_id"]
    if "mando" in features_df.columns:
        features_df = features_df.drop(columns="mando")
    features_df = features_df.merge(
        partidas_df[match_keys + ["mando"]].drop_duplicates(match_keys),
        on=match_keys,
        how="left",
    )
    features_df["mando"] = features_df["mando"].fillna(0)

    for window in [3, 5, 10]:
        features_df[f"media_pts_{window}r"] = (
            features_df.groupby(temporal_group_cols)["pontos"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        features_df[f"std_pts_{window}r"] = (
            features_df.groupby(temporal_group_cols)["pontos"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
        )

    features_df = _add_home_away_points_averages(features_df, temporal_group_cols)

    features_df["pts_ultima_rodada"] = features_df.groupby(temporal_group_cols)["pontos"].shift(1)

    for col in ["scout_G", "scout_A", "scout_SG", "scout_GS", "scout_DE", "scout_FD", "scout_FF", "scout_FT"]:
        if col in features_df.columns:
            round_col = f"{col}_round"

            features_df[round_col] = (
                features_df.groupby(["temporada", "atleta_id"])[col]
                .diff()
            )
            is_first_player_season_row = features_df.groupby(["temporada", "atleta_id"]).cumcount() == 0
            features_df.loc[is_first_player_season_row, round_col] = features_df.loc[
                is_first_player_season_row,
                col,
            ]

            features_df[f"{col}_5r"] = (
                features_df.groupby(["temporada", "atleta_id"])[round_col]
                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            )
    
    features_df["regularidade_5r"] = (
        features_df.groupby(temporal_group_cols)["jogou"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    features_df["preco_lag1"] = features_df.groupby(temporal_group_cols)["preco"].shift(1)
    
    club_match_goal_cols = ["gols_feitos_clube", "gols_sofridos_clube"]
    club_group_cols = ["temporada", "clube_id"]
    club_goals_df = (
        partidas_df[match_keys + club_match_goal_cols]
        .drop_duplicates(match_keys)
        .sort_values(club_group_cols + ["rodada"])
        .copy()
    )

    for col in club_match_goal_cols:
        club_goals_df[f"{col}_5r"] = (
            club_goals_df.groupby(club_group_cols)[col]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )

    features_df = features_df.merge(
        club_goals_df[match_keys + ["gols_feitos_clube_5r", "gols_sofridos_clube_5r"]],
        on=match_keys,
        how="left",
    )
        
    opponent_goal_features = (
        partidas_df[match_keys + ["clube_adversario_id"]]
        .drop_duplicates(match_keys)
        .merge(
            club_goals_df[
                match_keys + ["gols_feitos_clube_5r", "gols_sofridos_clube_5r"]
            ],
            left_on=["temporada", "rodada", "clube_adversario_id"],
            right_on=["temporada", "rodada", "clube_id"],
            how="left",
            suffixes=("", "_adv_stats"),
        )
        .rename(
            columns={
                "gols_feitos_clube_5r": "gols_feitos_adv_5r",
                "gols_sofridos_clube_5r": "gols_sofridos_adv_5r",
            }
        )
    )

    features_df = features_df.merge(
        opponent_goal_features[
            match_keys + ["gols_feitos_adv_5r", "gols_sofridos_adv_5r"]
        ],
        on=match_keys,
        how="left",
    )

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

    for col in odds_cols:
        if col not in features_df.columns:
            features_df[col] = float("nan")

    features_df[odds_cols] = features_df[odds_cols].fillna(
        {"prob_win": 1 / 3, "prob_draw": 1 / 3, "prob_loss": 1 / 3}
    )
    
    #features_df["clube_enc"] = LabelEncoder().fit_transform(features_df["clube_id"].astype(str))

    return features_df
