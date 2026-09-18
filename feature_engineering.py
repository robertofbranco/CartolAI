import logging
import pandas as pd

from sklearn.preprocessing import LabelEncoder

FEATURE_SCOUTS = [
    "scout_G", "scout_A", "scout_FS",
    "scout_FD", "scout_FF", "scout_FT",
    "scout_DS", "scout_FC", "scout_CA",
    "scout_GS", "scout_DE", "scout_SG",
    "scout_V"
]

GOL_FEATURE_COLS = [
    "clube_enc", "clube_adv_enc",
    "preco_lag1", #"pts_ultima_rodada",
    "regularidade_5r",
    "std_pts_3r", "std_pts_5r",
    "media_pts_3r", "media_pts_5r", "media_pts_10r",
    "avg_pts_mando_3r", "avg_pts_mando_5r",
    "prob_draw", "prob_loss",
    "mando",
    "gols_sofridos_clube_mando_3r", "gols_feitos_adv_mando_3r",
    "FD_avd_3r", "FD_adv_5r", # Attack Vol
    # Scouts
    "scout_GS_5r", "scout_CA_5r", "scout_FC_5r",
    "scout_DE_5r", "scout_SG_5r", "scout_G_5r",
]

TEC_FEATURE_COLS = [
    "clube_enc",
    "clube_adv_enc",
    "media_lag",
    "mando",
    "prob_win", "prob_draw",
    # How the player performed in the last matches
    "preco_lag1",
    "pts_ultima_rodada",
    "std_pts_3r", "std_pts_5r", "std_pts_10r",
    "media_pts_3r", "media_pts_5r", "media_pts_10r",
    "media_pts_delta_3_5r", "media_pts_delta_3_10r", "media_pts_delta_5_10r",
    "regularidade_5r",
    # How the player performs at home vs away
    "avg_pts_casa_3r", "avg_pts_fora_3r",
    "avg_pts_casa_5r", "avg_pts_fora_5r",
    # How the club performed in the last matches    
    "gols_feitos_clube_3r", "gols_sofridos_clube_3r",
    "gols_feitos_adv_3r", "gols_sofridos_adv_3r",
    "gols_feitos_clube_5r", "gols_sofridos_clube_5r",
    "gols_feitos_adv_5r", "gols_sofridos_adv_5r",
    "pontos_conquistados_clube_3r", "pontos_cedidos_adv_3r",
    "pontos_conquistados_clube_5r", "pontos_cedidos_adv_5r",
    # How the club performs away vs at home
    "gols_feitos_clube_mando_5r", "gols_sofridos_clube_mando_5r",
    "gols_feitos_adv_mando_5r", "gols_sofridos_adv_mando_5r",
    # Scouts
    "scout_V_5r",
]

FEATURE_COLS = [
    "clube_enc", "clube_adv_enc",
    "media_lag", "regularidade_5r",
    "prob_win", "prob_draw", "prob_loss",

    # How the player performed in the last matches
    "preco_lag1",
    "pts_ultima_rodada",
    "std_pts_3r", "std_pts_5r", #"std_pts_10r",
    "media_pts_3r", "media_pts_5r", "media_pts_10r",
    "media_pts_delta_3_5r", "media_pts_delta_3_10r", #"media_pts_delta_5_10r",    

    # How the player performs at home vs away
    "avg_pts_mando_3r", "avg_pts_mando_5r",

    # How the club performed in the last matches
    "gols_feitos_clube_3r", "gols_sofridos_clube_3r",
    "gols_feitos_adv_3r", "gols_sofridos_adv_3r",
    "gols_feitos_clube_5r", "gols_sofridos_clube_5r",
    "gols_feitos_adv_5r", "gols_sofridos_adv_5r",
    "pontos_conquistados_clube_3r", "pontos_cedidos_adv_3r",
    "pontos_conquistados_clube_5r", "pontos_cedidos_adv_5r",
    
    # How the club performs away vs at home
    "mando",
    "gols_feitos_clube_mando_3r", "gols_sofridos_clube_mando_3r",
    "gols_feitos_adv_mando_3r", "gols_sofridos_adv_mando_3r",
    "gols_feitos_clube_mando_5r", "gols_sofridos_clube_mando_5r",
    "gols_feitos_adv_mando_5r", "gols_sofridos_adv_mando_5r",

    # Scouts
    "scout_G_5r", "scout_A_5r", "scout_FS_5r",
    "scout_FD_5r", "scout_FF_5r", "scout_FT_5r",    
    "scout_DS_5r", "scout_FC_5r", "scout_CA_5r",
    "scout_GS_5r", "scout_DE_5r", "scout_SG_5r",
]

WINDOWS = (3, 5, 10)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _add_mando_points_averages(
    features_df: pd.DataFrame,    
    temporal_group_cols: list[str],
    window: int = 5,
) -> pd.DataFrame:
    """
    Adds the average of the player's previous `window` mando matches.
    """
    features_df = features_df.copy()
    mando_col = f"avg_pts_mando_{window}r"
    features_df[mando_col] = float("nan")

    features_df[mando_col] = (
        features_df.groupby(temporal_group_cols + ["mando"], sort=False)["pontos"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    return features_df


def _shift_post_round_market_columns(
    features_df: pd.DataFrame,
    temporal_group_cols: list[str],
) -> pd.DataFrame:
    """Use only market values that were known before the row's round."""
    post_round_cols = ["preco", "media"]
    available_cols = [col for col in post_round_cols if col in features_df.columns]
    if not available_cols:
        return features_df

    features_df = features_df.copy()
    features_df[available_cols] = (
        features_df.groupby(temporal_group_cols)[available_cols].shift(1)
    )
    return features_df


def _add_rolling_goal_features(
    features_df: pd.DataFrame,
    partidas_df: pd.DataFrame,
    match_keys: list[str],
    window: int,
) -> pd.DataFrame:
    """Add club and opponent goal averages from previous club matches."""
    club_match_goal_cols = ["gols_feitos_clube", "gols_sofridos_clube"]
    club_goal_feature_cols = [
        f"gols_feitos_clube_{window}r",
        f"gols_sofridos_clube_{window}r",
    ]
    club_mando_goal_feature_cols = [
        f"gols_feitos_clube_mando_{window}r",
        f"gols_sofridos_clube_mando_{window}r",
    ]
    opponent_goal_feature_cols = [
        f"gols_feitos_adv_{window}r",
        f"gols_sofridos_adv_{window}r",
    ]
    opponent_mando_goal_feature_cols = [
        f"gols_feitos_adv_mando_{window}r",
        f"gols_sofridos_adv_mando_{window}r",
    ]

    has_mando = "mando" in partidas_df.columns
    club_group_cols = ["temporada", "clube_id"]
    club_goals_cols = match_keys + ["mando"] + club_match_goal_cols

    club_goals_df = (
        partidas_df[club_goals_cols]
        .drop_duplicates(match_keys)
        .sort_values(club_group_cols + ["rodada"])
        .copy()
    )

    for col in club_match_goal_cols:
        club_goals_df[f"{col}_{window}r"] = (
            club_goals_df.groupby(club_group_cols)[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

        mando_col = f"{col.replace('_clube', '_clube_mando')}_{window}r"
        if has_mando:
            club_goals_df[mando_col] = (
                club_goals_df.groupby(club_group_cols + ["mando"])[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
        else:
            club_goals_df[mando_col] = float("nan")

    features_df = features_df.merge(
        club_goals_df[match_keys + club_goal_feature_cols + club_mando_goal_feature_cols],
        on=match_keys,
        how="left",
    )

    opponent_goal_features = (
        partidas_df[match_keys + ["clube_adversario_id"]]
        .drop_duplicates(match_keys)
        .merge(
            club_goals_df[match_keys + club_goal_feature_cols + club_mando_goal_feature_cols],
            left_on=["temporada", "rodada", "clube_adversario_id"],
            right_on=["temporada", "rodada", "clube_id"],
            how="left",
            suffixes=("", "_adv_stats"),
        )
        .rename(
            columns={
                f"gols_feitos_clube_{window}r": f"gols_feitos_adv_{window}r",
                f"gols_sofridos_clube_{window}r": f"gols_sofridos_adv_{window}r",
                f"gols_feitos_clube_mando_{window}r": f"gols_feitos_adv_mando_{window}r",
                f"gols_sofridos_clube_mando_{window}r": f"gols_sofridos_adv_mando_{window}r",
            }
        )
    )

    return features_df.merge(
        opponent_goal_features[match_keys + opponent_goal_feature_cols + opponent_mando_goal_feature_cols],
        on=match_keys,
        how="left",
    )


def _add_rolling_club_points_features(
    features_df: pd.DataFrame,
    partidas_df: pd.DataFrame,
    match_keys: list[str],
    window: int,
) -> pd.DataFrame:
    """Add rolling fantasy-point totals for each club and opponent."""
    club_points_col = "pontos_conquistados_clube"
    club_conceded_col = "pontos_cedidos_clube"
    club_points_feature_cols = [
        f"pontos_conquistados_clube_{window}r",
        f"pontos_cedidos_clube_{window}r",
    ]
    club_mando_points_feature_cols = [
        f"pontos_conquistados_clube_mando_{window}r",
        f"pontos_cedidos_clube_mando_{window}r",
    ]
    team_points_feature_cols = [
        f"pontos_conquistados_clube_mando_{window}r",
        f"pontos_conquistados_clube_{window}r",
    ]
    adv_conceded_points_feature_cols = [
        f"pontos_cedidos_adv_mando_{window}r",
        f"pontos_cedidos_adv_{window}r",
    ]
    required_cols = match_keys + ["mando", "clube_adversario_id"]
    missing_cols = [col for col in required_cols if col not in partidas_df.columns]
    if missing_cols or "pontos" not in features_df.columns:
        features_df = features_df.copy()
        for col in team_points_feature_cols + adv_conceded_points_feature_cols:
            features_df[col] = float("nan")
        return features_df

    has_mando = "mando" in partidas_df.columns
    club_group_cols = ["temporada", "clube_id"]
    round_cols = ["temporada", "rodada"]
    club_round_points = (
        features_df[match_keys + ["pontos"]]
        .groupby(match_keys, as_index=False)["pontos"]
        .sum(min_count=1)
        .rename(columns={"pontos": club_points_col})
    )
    club_points_df = (
        partidas_df[required_cols]
        .drop_duplicates(match_keys)
        .sort_values(club_group_cols + ["rodada"])
        .copy()
    )
    club_points_df = club_points_df.merge(
        club_round_points,
        on=match_keys,
        how="left",
    )
    club_points_df = club_points_df.merge(
        club_round_points,
        left_on=round_cols + ["clube_adversario_id"],
        right_on=round_cols + ["clube_id"],
        how="left",
        suffixes=("", "_adv"),
    )
    # Points conceded are the fantasy points scored by this club's opponent.
    club_points_df = club_points_df.rename(
        columns={f"{club_points_col}_adv": club_conceded_col}
    )
    if "clube_id_adv" in club_points_df.columns:
        club_points_df = club_points_df.drop(columns="clube_id_adv")
    club_points_df = club_points_df.sort_values(club_group_cols + ["rodada"])

    for col in [club_points_col, club_conceded_col]:
        club_points_df[f"{col}_{window}r"] = (
            club_points_df.groupby(club_group_cols)[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

        mando_col = f"{col.replace('_clube', '_clube_mando')}_{window}r"
        if has_mando:
            club_points_df[mando_col] = (
                club_points_df.groupby(club_group_cols + ["mando"])[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
        else:
            club_points_df[mando_col] = float("nan")

    features_df = features_df.merge(
        club_points_df[match_keys + team_points_feature_cols],
        on=match_keys,
        how="left",
    )

    opponent_points_features = (
        partidas_df[match_keys + ["clube_adversario_id"]]
        .drop_duplicates(match_keys)
        .merge(
            club_points_df[match_keys + club_points_feature_cols + club_mando_points_feature_cols],
            left_on=["temporada", "rodada", "clube_adversario_id"],
            right_on=["temporada", "rodada", "clube_id"],
            how="left",
            suffixes=("", "_adv_stats"),
        )
        .rename(
            columns={
                f"pontos_cedidos_clube_{window}r": f"pontos_cedidos_adv_{window}r",
                f"pontos_cedidos_clube_mando_{window}r": f"pontos_cedidos_adv_mando_{window}r",
            }
        )
    )

    return features_df.merge(
        opponent_points_features[match_keys + adv_conceded_points_feature_cols],
        on=match_keys,
        how="left",
    )


def _add_rolling_opponent_fd_features(
    features_df: pd.DataFrame,
    partidas_df: pd.DataFrame,
    match_keys: list[str],
) -> pd.DataFrame:
    """Add the opponent's prior average shots saved (FD) per match.

    ``scout_FD`` is cumulative at player level, so it is first converted to a
    per-round count and then summed for each club.  The feature intentionally
    keeps the scout count rather than Cartola points (1.2 points per FD).
    """
    feature_cols = ["FD_avd_3r", "FD_adv_5r"]
    required_match_cols = match_keys + ["clube_adversario_id"]
    if "scout_FD" not in features_df.columns or any(
        col not in partidas_df.columns for col in required_match_cols
    ):
        features_df = features_df.copy()
        for col in feature_cols:
            features_df[col] = float("nan")
        return features_df

    features_df = features_df.copy()
    player_group_cols = ["temporada", "atleta_id"]
    features_df["_scout_FD_round"] = (
        features_df.groupby(player_group_cols)["scout_FD"].diff()
    )
    first_player_season_row = features_df.groupby(player_group_cols).cumcount() == 0
    features_df.loc[first_player_season_row, "_scout_FD_round"] = features_df.loc[
        first_player_season_row, "scout_FD"
    ]

    club_fd_df = (
        features_df[match_keys + ["_scout_FD_round"]]
        .groupby(match_keys, as_index=False)["_scout_FD_round"]
        .sum(min_count=1)
        .sort_values(["temporada", "clube_id", "rodada"])
    )
    for window in (3, 5):
        club_fd_df[f"_FD_clube_{window}r"] = (
            club_fd_df.groupby(["temporada", "clube_id"])["_scout_FD_round"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    opponent_fd_df = (
        partidas_df[required_match_cols]
        .drop_duplicates(match_keys)
        .merge(
            club_fd_df[
                match_keys + ["_FD_clube_3r", "_FD_clube_5r"]
            ].rename(columns={"clube_id": "clube_adversario_id"}),
            on=["temporada", "rodada", "clube_adversario_id"],
            how="left",
        )
        .rename(
            columns={"_FD_clube_3r": "FD_avd_3r", "_FD_clube_5r": "FD_adv_5r"}
        )
    )
    return features_df.merge(opponent_fd_df[match_keys + feature_cols], on=match_keys, how="left")


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
    temporal_group_cols = ["temporada", "atleta_id"]
    sort_cols = temporal_group_cols + ["rodada"]
    features_df = features_df.sort_values(sort_cols).copy()
    features_df = _shift_post_round_market_columns(features_df, temporal_group_cols)

    match_keys = ["temporada", "rodada", "clube_id"]
    match_context_cols = [
        col for col in ["mando", "clube_adversario_id"] if col in partidas_df.columns
    ]
    for col in match_context_cols:
        if col in features_df.columns:
            features_df = features_df.drop(columns=col)
    features_df = features_df.merge(
        partidas_df[match_keys + match_context_cols].drop_duplicates(match_keys),
        on=match_keys,
        how="left",
    )
    if "mando" not in features_df.columns:
        features_df["mando"] = 0
    features_df["mando"] = features_df["mando"].fillna(0)

    for window in WINDOWS:
        features_df[f"media_pts_{window}r"] = (
            features_df.groupby(temporal_group_cols)["pontos"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        features_df[f"std_pts_{window}r"] = (
            features_df.groupby(temporal_group_cols)["pontos"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
        )

        features_df = _add_mando_points_averages(features_df, temporal_group_cols, window)

        features_df = _add_rolling_goal_features(
            features_df,
            partidas_df,
            match_keys,
            window,
        )
        features_df = _add_rolling_club_points_features(
            features_df,
            partidas_df,
            match_keys,
            window,
        )

    features_df = _add_rolling_opponent_fd_features(
        features_df,
        partidas_df,
        match_keys,
    )

    for window in WINDOWS:
        for col in FEATURE_SCOUTS:
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

                features_df[f"{col}_{window}r"] = (
                    features_df.groupby(["temporada", "atleta_id"])[round_col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )

    features_df["media_pts_delta_3_5r"] = (
        features_df["media_pts_3r"] - features_df["media_pts_5r"]
    )
    features_df["media_pts_delta_3_10r"] = (
        features_df["media_pts_3r"] - features_df["media_pts_10r"]
    )
    features_df["media_pts_delta_5_10r"] = (
        features_df["media_pts_5r"] - features_df["media_pts_10r"]
    )

    features_df["pts_ultima_rodada"] = features_df.groupby(temporal_group_cols)["pontos"].shift(1)

    features_df["regularidade_5r"] = (
        features_df.groupby(temporal_group_cols)["jogou"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
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
    
    # preco and media were already shifted by _shift_post_round_market_columns().
    features_df["preco_lag1"] = (
        features_df["preco"] if "preco" in features_df.columns else float("nan")
    )
    features_df["media_lag"] = (
        features_df["media"] if "media" in features_df.columns else float("nan")
    )

    features_df["clube_enc"] = LabelEncoder().fit_transform(features_df["clube_id"].astype(str))
    features_df["clube_adv_enc"] = LabelEncoder().fit_transform(
        features_df["clube_adversario_id"].astype(str)
    )

    return features_df
