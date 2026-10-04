import logging
import os
import argparse
import pandas as pd
from dotenv import load_dotenv

from cartola_data.current import get_current_round
from feature_engineering import build_features
from cartola_model_training import (
    DEFAULT_MODEL_STRATEGY,
    available_model_strategies,
    feature_cols_from_models,
    mean_mae_from_models,
    model_feature_matrix,
    train_models_by_position,
)

from cartola_data.config import (
    CAPTAIN_BONUS,
    CAPTAIN_POS,
    CURRENT_SEASON,
    DATA_DIR,
    FORMATION,
    POSICAO_NOME,
    STATUS,
    ODDS_FILTER,
    TUNING,
)

from cartola_data.api import CartolaAPI
from cartola_data.datasets import read_datasets

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ODDS_COLS = ["prob_win", "prob_draw", "prob_loss"]
POS_THRESHOLD = [1, 2, 3, 4, 5, 6]
TEC_POSITION_ID = 6
LUXURY_RESERVE_POSITIONS = (4, 5)
ATTACK_CANDIDATE_LIMIT = 5
DEFAULT_ATTACK_SELECTION_POLICY = {
    4: {"risk": 0, "stable": 1, "default": 2},
    5: {"risk": 1, "stable": 0, "default": 2},
}
CAPTAIN_POSITION_SELECTION_POLICY = {"risk": 1, "stable": 1, "default": 2}
CAPTAIN_COL = "capitao"
LINEUP_OUTPUT_COLUMNS = [
    "rodada",
    "atleta_id",
    "apelido",
    "posicao",
    "clube",
    "clube adversario",
    "mando",
    "pontos_previstos",
    "reserva",
    "reserva_de_luxo",
    "capitao",
]


def merge_target_round_odds(
    market_df: pd.DataFrame,
    odds_df: pd.DataFrame | None,
    season: int,
    rodada_alvo: int,
) -> pd.DataFrame:
    """Attach match odds for the round being predicted."""
    market_df = market_df.copy()
    if odds_df is None or odds_df.empty:
        return market_df

    odds_keys = ["temporada", "rodada", "clube_id"]
    if not set(odds_keys + ODDS_COLS).issubset(odds_df.columns):
        return market_df

    round_odds = odds_df.loc[
        (odds_df["temporada"] == season)
        & (odds_df["rodada"] == rodada_alvo),
        odds_keys + ODDS_COLS,
    ].drop_duplicates(odds_keys)
    if round_odds.empty:
        for col in ODDS_COLS:
            if col in market_df.columns:
                market_df[col] = pd.NA
        return market_df

    for col in ODDS_COLS:
        if col in market_df.columns:
            market_df = market_df.drop(columns=col)

    market_df = market_df.merge(
        round_odds,
        on=odds_keys,
        how="left",
    )

    return market_df


def build_target_round_market_features(
    df_players_per_round: pd.DataFrame,
    df_matches: pd.DataFrame | None,
    df_odds: pd.DataFrame | None,
    df_market: pd.DataFrame,
    season: int,
    rodada_alvo: int,
) -> pd.DataFrame:
    """Build a target-round row using only information available at lock time."""
    if df_matches is None:
        df_matches = pd.DataFrame(
            columns=[
                "temporada",
                "rodada",
                "clube_id",
                "mando",
                "clube_adversario_id",
                "gols_feitos_clube",
                "gols_sofridos_clube",
            ]
        )
    if df_odds is None:
        df_odds = pd.DataFrame(
            columns=["temporada", "rodada", "clube_id", *ODDS_COLS]
        )

    history_players = df_players_per_round.loc[
        (df_players_per_round["temporada"] < season)
        | (
            (df_players_per_round["temporada"] == season)
            & (df_players_per_round["rodada"] < rodada_alvo)
        )
    ].copy()

    target_market = df_market.copy()
    target_market["temporada"] = season
    target_market["rodada"] = rodada_alvo

    result_cols = ["pontos", "jogou", "entrou_em_campo"] + [
        col for col in target_market.columns if col.startswith("scout_")
    ]
    for col in result_cols:
        if col in target_market.columns:
            target_market[col] = pd.NA

    players_context = pd.concat(
        [history_players, target_market],
        ignore_index=True,
        sort=False,
    )

    matches_context = df_matches.loc[
        (df_matches["temporada"] < season)
        | (
            (df_matches["temporada"] == season)
            & (df_matches["rodada"] <= rodada_alvo)
        )
    ].copy()
    target_match_mask = (
        (matches_context["temporada"] == season)
        & (matches_context["rodada"] == rodada_alvo)
    )
    for col in ["gols_feitos_clube", "gols_sofridos_clube"]:
        if col in matches_context.columns:
            matches_context.loc[target_match_mask, col] = pd.NA

    odds_context = df_odds.loc[
        (df_odds["temporada"] < season)
        | (
            (df_odds["temporada"] == season)
            & (df_odds["rodada"] <= rodada_alvo)
        )
    ].copy()

    market_features = build_features(players_context, matches_context, odds_context)
    market_features = market_features.loc[
        (market_features["temporada"] == season)
        & (market_features["rodada"] == rodada_alvo)
    ].copy()

    market_features = merge_target_round_odds(
        market_features,
        df_odds,
        season,
        rodada_alvo,
    )

    if set(ODDS_COLS).issubset(market_features.columns):
        missing_odds = market_features[ODDS_COLS].isna().any(axis=1)
        if missing_odds.any():
            missing_clubs = sorted(
                market_features.loc[missing_odds, "clube_id"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            log.warning(
                "Rodada %s: odds ausentes para %s atletas em clubes %s; "
                "usando fallback neutro.",
                rodada_alvo,
                int(missing_odds.sum()),
                missing_clubs,
            )
            market_features.loc[missing_odds, ODDS_COLS] = market_features.loc[
                missing_odds,
                ODDS_COLS,
            ].fillna(1 / 3)

    return market_features


def merge_target_round_match_context(
    market_df: pd.DataFrame,
    matches_df: pd.DataFrame | None,
    season: int,
    rodada_alvo: int,
    clubes_lookup: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Attach target-round home/away and opponent information."""
    market_df = market_df.copy()
    if matches_df is None or matches_df.empty:
        return market_df

    match_cols = ["temporada", "rodada", "clube_id", "mando", "clube_adversario_id"]
    if not set(match_cols).issubset(matches_df.columns):
        return market_df

    match_context = (
        matches_df.loc[
            (matches_df["temporada"] == season)
            & (matches_df["rodada"] == rodada_alvo),
            match_cols,
        ]
        .drop_duplicates(["temporada", "rodada", "clube_id"])
        .copy()
    )
    if match_context.empty:
        return market_df

    for col in ["mando", "clube_adversario_id", "adversario"]:
        if col in market_df.columns:
            market_df = market_df.drop(columns=col)

    market_df = market_df.merge(
        match_context,
        on=["temporada", "rodada", "clube_id"],
        how="left",
    )

    if clubes_lookup:
        market_df["adversario"] = (
            market_df["clube_adversario_id"]
            .map(clubes_lookup)
            .fillna("")
        )

    return market_df


def refresh_opponent_encoding_from_features(
    market_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Update target-round opponent encoding using the training feature mapping."""
    required_cols = {"clube_adversario_id", "clube_adv_enc"}
    if (
        "clube_adversario_id" not in market_df.columns
        or not required_cols.issubset(features_df.columns)
    ):
        return market_df

    encoded_opponents = (
        features_df[["clube_adversario_id", "clube_adv_enc"]]
        .dropna(subset=["clube_adversario_id", "clube_adv_enc"])
        .copy()
    )
    if encoded_opponents.empty:
        return market_df

    encoded_opponents["encoding_key"] = pd.to_numeric(
        encoded_opponents["clube_adversario_id"],
        errors="coerce",
    ).astype("Int64").astype(str)
    encoding_lookup = (
        encoded_opponents
        .drop_duplicates("encoding_key", keep="last")
        .set_index("encoding_key")["clube_adv_enc"]
    )

    market_df = market_df.copy()
    target_key = pd.to_numeric(
        market_df["clube_adversario_id"],
        errors="coerce",
    ).astype("Int64").astype(str)
    target_encoding = target_key.map(encoding_lookup)
    has_target_opponent = market_df["clube_adversario_id"].notna()
    if "clube_adv_enc" in market_df.columns:
        market_df.loc[has_target_opponent, "clube_adv_enc"] = target_encoding[
            has_target_opponent
        ]
    else:
        market_df["clube_adv_enc"] = target_encoding

    return market_df


def apply_odds_filter(
    position_pool: pd.DataFrame,
    min_prob_win: float,
    max_prob_loss: float,
) -> pd.DataFrame:
    """Avoid selecting high-risk TEC picks when safer candidates exist."""
    if not set(["prob_win", "prob_loss"]).issubset(position_pool.columns):
        return position_pool

    eligible = position_pool[
        (position_pool["prob_win"] >= min_prob_win)
        & (position_pool["prob_loss"] <= max_prob_loss)
    ]
    if eligible.empty:
        return position_pool

    return eligible


def _points_lookup(
    team_df: pd.DataFrame,
    play_status_df: pd.DataFrame | None = None,
    points_column: str = "pontos",
) -> dict:
    score_source = play_status_df if play_status_df is not None else team_df
    if not {"atleta_id", points_column}.issubset(score_source.columns):
        return {}

    return (
        score_source[["atleta_id", points_column]]
        .dropna(subset=["atleta_id"])
        .drop_duplicates("atleta_id", keep="last")
        .set_index("atleta_id")[points_column]
        .to_dict()
    )


def _did_not_play(value) -> bool:
    """Return True only when a played flag is explicitly false."""
    if pd.isna(value):
        return False

    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "nao", "n"}

    return value is False or value == 0


def _played_lookup(team_df: pd.DataFrame, play_status_df: pd.DataFrame | None = None) -> dict:
    status_source = play_status_df if play_status_df is not None else team_df
    if not {"atleta_id", "jogou"}.issubset(status_source.columns):
        return {}

    return (
        status_source[["atleta_id", "jogou"]]
        .dropna(subset=["atleta_id"])
        .drop_duplicates("atleta_id", keep="last")
        .set_index("atleta_id")["jogou"]
        .to_dict()
    )


def _player_score(
    player: pd.Series,
    points_by_athlete: dict,
    points_column: str = "pontos",
) -> float:
    value = points_by_athlete.get(player.get("atleta_id"), player.get(points_column, pd.NA))
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return float("-inf")

    return float(value)


def _price_values(players: pd.DataFrame, price_column: str = "preco") -> pd.Series:
    if price_column not in players.columns:
        return pd.Series(float("inf"), index=players.index)

    return pd.to_numeric(players[price_column], errors="coerce").fillna(float("inf"))


def _score_values(
    players: pd.DataFrame,
    score_column: str,
    fallback_column: str = "pontos_previstos",
) -> pd.Series:
    column = score_column if score_column in players.columns else fallback_column
    if column not in players.columns:
        return pd.Series(float("-inf"), index=players.index)

    return pd.to_numeric(players[column], errors="coerce").fillna(float("-inf"))


def _select_indices_by_score(
    players: pd.DataFrame,
    score_column: str,
    count: int,
) -> list:
    if count <= 0 or players.empty:
        return []

    scores = _score_values(players, score_column)
    return scores.sort_values(ascending=False).head(count).index.tolist()


def _reserve_price_threshold(
    position_pool: pd.DataFrame,
    starter_indices: list,
    price_column: str,
) -> float:
    if not starter_indices:
        return float("-inf")

    starters = position_pool.loc[position_pool.index.intersection(starter_indices)]
    if starters.empty:
        return float("-inf")

    return float(_price_values(starters, price_column).min())


def _select_reserve_index(
    position_pool: pd.DataFrame,
    starter_indices: list,
    price_column: str,
    score_column: str,
    candidate_pool: pd.DataFrame | None = None,
) -> int | None:
    threshold = _reserve_price_threshold(position_pool, starter_indices, price_column)
    if threshold in {float("-inf"), float("inf")}:
        return None

    pool = candidate_pool if candidate_pool is not None else position_pool
    candidates = pool.drop(index=starter_indices, errors="ignore").copy()
    if candidates.empty:
        return None

    cheaper_mask = _price_values(candidates, price_column) < threshold
    candidates = candidates[cheaper_mask]
    if candidates.empty:
        return None

    return _select_indices_by_score(candidates, score_column, 1)[0]


def _attack_selection_counts(
    position: int,
    n_players: int,
    policy: dict | None = None,
) -> tuple[int, int, int]:
    position_policy = (policy or DEFAULT_ATTACK_SELECTION_POLICY).get(
        int(position),
        DEFAULT_ATTACK_SELECTION_POLICY.get(int(position), {}),
    )
    risk_count = int(position_policy.get("risk", 1))
    stable_count = int(position_policy.get("stable", 0))
    default_count = int(
        position_policy.get(
            "default",
            max(n_players - risk_count - stable_count, 0),
        )
    )

    risk_count = max(risk_count, 0)
    stable_count = max(stable_count, 0)
    default_count = max(default_count, 0)
    if risk_count + stable_count > n_players:
        stable_count = max(n_players - risk_count, 0)
        default_count = 0
    elif risk_count + stable_count + default_count > n_players:
        default_count = max(n_players - risk_count - stable_count, 0)
    if risk_count > n_players:
        risk_count = n_players
        stable_count = 0
        default_count = 0

    return risk_count, stable_count, default_count


def _select_attack_starters(
    position_pool: pd.DataFrame,
    n_players: int,
    high_risk_score_column: str,
    safe_score_column: str,
    risk_count: int,
    stable_count: int,
    default_count: int = 0,
) -> tuple[list, list, pd.DataFrame]:
    top_candidates = position_pool.head(max(ATTACK_CANDIDATE_LIMIT, n_players)).copy()
    if n_players <= 0 or top_candidates.empty:
        return [], [], top_candidates

    selected_indices = []
    high_risk_indices = _select_indices_by_score(
        top_candidates,
        high_risk_score_column,
        min(risk_count, n_players),
    )
    selected_indices.extend(high_risk_indices)

    safe_candidates = top_candidates.drop(index=selected_indices, errors="ignore")
    selected_indices.extend(
        _select_indices_by_score(
            safe_candidates,
            safe_score_column,
            min(stable_count, n_players - len(selected_indices)),
        )
    )

    default_candidates = top_candidates.drop(index=selected_indices, errors="ignore")
    selected_indices.extend(
        _select_indices_by_score(
            default_candidates,
            "pontos_previstos",
            min(default_count, n_players - len(selected_indices)),
        )
    )

    if len(selected_indices) < n_players:
        fallback_candidates = top_candidates.drop(index=selected_indices, errors="ignore")
        selected_indices.extend(
            _select_indices_by_score(
                fallback_candidates,
                "pontos_previstos",
                n_players - len(selected_indices),
            )
        )

    if len(selected_indices) < n_players:
        fallback_candidates = position_pool.drop(index=selected_indices, errors="ignore")
        selected_indices.extend(
            _select_indices_by_score(
                fallback_candidates,
                "pontos_previstos",
                n_players - len(selected_indices),
            )
        )

    return selected_indices, high_risk_indices, top_candidates


def _select_cheapest_reserve_from_candidates(
    position_pool: pd.DataFrame,
    candidate_indices: list,
    price_column: str,
) -> int | None:
    candidates = position_pool.loc[position_pool.index.intersection(candidate_indices)]
    if len(candidates) <= 1:
        return None

    prices = _price_values(candidates, price_column)
    reserve_index = prices.idxmin()
    starter_prices = prices.drop(index=reserve_index)
    if starter_prices.empty or prices.loc[reserve_index] >= starter_prices.min():
        return None

    return reserve_index


def _best_risk_captain_index(
    players: pd.DataFrame,
    risk_indices: list,
    starter_indices: list,
    captain_score_column: str,
) -> int | None:
    eligible_indices = [
        player_index
        for player_index in risk_indices
        if player_index in starter_indices
    ]
    if not eligible_indices:
        eligible_indices = starter_indices
    if not eligible_indices:
        return None

    candidates = players.loc[players.index.intersection(eligible_indices)]
    if candidates.empty:
        return None

    return _score_values(candidates, captain_score_column).idxmax()


def assign_captain(
    team_df: pd.DataFrame,
    score_column: str = "pontos_previstos",
    captain_positions: list[int] | tuple[int, ...] | None = CAPTAIN_POS,
) -> pd.DataFrame:
    """Mark one non-reserve player as captain using the best available score."""
    team_df = team_df.copy()
    if team_df.empty:
        team_df[CAPTAIN_COL] = pd.Series(dtype=bool)
        return team_df

    team_df[CAPTAIN_COL] = False

    if "reserva" in team_df.columns:
        starter_mask = ~team_df["reserva"].fillna(False).astype(bool)
    else:
        starter_mask = pd.Series(True, index=team_df.index)

    eligible_mask = starter_mask.copy()
    if captain_positions and "posicao_id" in team_df.columns:
        eligible_mask &= team_df["posicao_id"].isin(captain_positions)

    candidates = team_df[eligible_mask]
    if candidates.empty:
        candidates = team_df[starter_mask]
    if candidates.empty:
        candidates = team_df

    if score_column in candidates.columns:
        scores = pd.to_numeric(candidates[score_column], errors="coerce").fillna(float("-inf"))
    else:
        scores = pd.Series(0.0, index=candidates.index)

    captain_index = scores.idxmax()
    team_df.loc[captain_index, CAPTAIN_COL] = True

    return team_df


def score_with_captain_bonus(
    team_df: pd.DataFrame,
    points_column: str = "pontos",
    captain_bonus: float = CAPTAIN_BONUS,
) -> float:
    """Return team points after applying the captain multiplier."""
    if team_df.empty or points_column not in team_df.columns:
        return 0.0

    points = pd.to_numeric(team_df[points_column], errors="coerce").fillna(0.0)
    if CAPTAIN_COL in team_df.columns:
        captain_mask = team_df[CAPTAIN_COL].fillna(False).astype(bool)
    else:
        captain_mask = pd.Series(False, index=team_df.index)

    multipliers = pd.Series(1.0, index=team_df.index)
    multipliers.loc[captain_mask] = captain_bonus

    return float((points * multipliers).sum())


def lineup_output_table(time_df: pd.DataFrame) -> pd.DataFrame:
    """Return the compact lineup table used for CLI display."""
    output_df = time_df.copy()

    if "posicao" not in output_df.columns:
        output_df["posicao"] = output_df.get("posicao_id", pd.Series(dtype=object)).map(POSICAO_NOME)

    if "clube" not in output_df.columns:
        if "clube_nome" in output_df.columns:
            output_df["clube"] = output_df["clube_nome"]
        else:
            output_df["clube"] = ""

    if "clube adversario" not in output_df.columns:
        output_df["clube adversario"] = output_df.get("adversario", "")

    has_points_column = "pontos" in output_df.columns
    if "pontos_real" in output_df.columns:
        if has_points_column:
            output_df["pontos"] = output_df["pontos"].where(
                output_df["pontos"].notna(),
                output_df["pontos_real"],
            )
        else:
            output_df["pontos"] = output_df["pontos_real"]
            has_points_column = True
    elif not has_points_column:
        output_df["pontos"] = ""
    if "pontos_com_bonus" not in output_df.columns:
        if "pontos_real" in output_df.columns:
            points_source = "pontos_real"
        elif has_points_column:
            points_source = "pontos"
        else:
            points_source = None

        if points_source is not None:
            points = pd.to_numeric(output_df[points_source], errors="coerce").fillna(0.0)
            output_df["pontos_com_bonus"] = points
            if CAPTAIN_COL in output_df.columns:
                captain_mask = output_df[CAPTAIN_COL].fillna(False).astype(bool)
                output_df.loc[captain_mask, "pontos_com_bonus"] = (
                    points.loc[captain_mask] * CAPTAIN_BONUS
                )
        else:
            output_df["pontos_com_bonus"] = ""

    for column in LINEUP_OUTPUT_COLUMNS:
        if column not in output_df.columns:
            output_df[column] = ""

    output_df["mando"] = output_df["mando"].map({
        1: "CASA",
        -1: "FORA",
    }).fillna(output_df["mando"])

    return output_df[LINEUP_OUTPUT_COLUMNS]


def apply_reserve_substitutions(
    team_df: pd.DataFrame,
    play_status_df: pd.DataFrame | None = None,
    points_column: str = "pontos",
) -> pd.DataFrame:
    """
    Return the scoring lineup after same-position luxury reserve substitutions.

    The luxury reserve is restricted to MEI and ATA. Each eligible position can
    have at most one reserve, and that reserve replaces the lowest-scoring
    starter in the same position only when the reserve scores more points.
    """
    if team_df.empty:
        return team_df.copy()

    team_df = team_df.copy()
    if "reserva" not in team_df.columns:
        team_df["reserva"] = False
    if "reserva_de_luxo" not in team_df.columns:
        team_df["reserva_de_luxo"] = False
    if CAPTAIN_COL not in team_df.columns:
        team_df[CAPTAIN_COL] = False

    points_by_athlete = _points_lookup(team_df, play_status_df, points_column)
    played_by_athlete = _played_lookup(team_df, play_status_df)
    starters = team_df[~team_df["reserva"].fillna(False).astype(bool)].copy()
    reserves = team_df[team_df["reserva"].fillna(False).astype(bool)].copy()
    final_players = []

    for position, position_starters in starters.groupby("posicao_id", sort=False):
        position_starters = position_starters.copy()
        replacement_row = None
        replaced_index = None

        position_reserves = reserves[reserves["posicao_id"] == position]
        if not position_reserves.empty and int(position) != TEC_POSITION_ID:
            luxury_reserves = position_reserves[
                position_reserves["reserva_de_luxo"].fillna(False).astype(bool)
            ]
            if int(position) in LUXURY_RESERVE_POSITIONS and not luxury_reserves.empty:
                reserve = luxury_reserves.iloc[0].copy()
                reserve_score = _player_score(reserve, points_by_athlete, points_column)
                starter_scores = position_starters.apply(
                    lambda starter: _player_score(starter, points_by_athlete, points_column),
                    axis=1,
                )
                lowest_starter_index = starter_scores.idxmin()
                lowest_starter_score = starter_scores.loc[lowest_starter_index]

                if reserve_score > lowest_starter_score:
                    starter = position_starters.loc[lowest_starter_index]
                    if bool(starter.get(CAPTAIN_COL, False)):
                        reserve[CAPTAIN_COL] = True
                    reserve["substituiu_atleta_id"] = starter["atleta_id"]
                    reserve["substituiu_apelido"] = starter.get("apelido", "")
                    replacement_row = reserve
                    replaced_index = lowest_starter_index

            if replacement_row is None:
                reserve = position_reserves.iloc[0].copy()
                for starter_index, starter in position_starters.iterrows():
                    played_value = played_by_athlete.get(
                        starter["atleta_id"],
                        starter.get("jogou", pd.NA),
                    )
                    if _did_not_play(played_value):
                        if bool(starter.get(CAPTAIN_COL, False)):
                            reserve[CAPTAIN_COL] = True
                        reserve["substituiu_atleta_id"] = starter["atleta_id"]
                        reserve["substituiu_apelido"] = starter.get("apelido", "")
                        replacement_row = reserve
                        replaced_index = starter_index
                        break

        if replacement_row is not None:
            position_starters = position_starters.drop(index=replaced_index)
            final_players.extend(position_starters.to_dict("records"))
            final_players.append(replacement_row.to_dict())
        else:
            final_players.extend(position_starters.to_dict("records"))

    final_df = pd.DataFrame(final_players)
    if final_df.empty:
        return final_df

    if "substituiu_atleta_id" not in final_df.columns:
        final_df["substituiu_atleta_id"] = pd.NA
    if "substituiu_apelido" not in final_df.columns:
        final_df["substituiu_apelido"] = ""

    sort_score = "pontos_previstos" if "pontos_previstos" in final_df.columns else "pontos"
    sort_columns = ["posicao_id"]
    ascending = [True]
    if sort_score in final_df.columns:
        sort_columns.append(sort_score)
        ascending.append(False)

    return final_df.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def build_team(
    market_df: pd.DataFrame,
    models_by_position: dict,
    formation: dict = FORMATION,
    odds_filter: dict | None = ODDS_FILTER,
    include_reserves: bool = False,
    captain_score_column: str = "captain_score",    
    captain_std_column: str = "std_pts_5r",
    captain_std_weight: float = 0.8,
    align_luxury_reserve_with_captain: bool = True,
    reserve_price_column: str = "preco",
    attack_selection_policy: dict | None = None,
) -> pd.DataFrame:
    df = market_df[market_df['status_id'] == STATUS["Provavel"]].copy()
    df["pontos_previstos"] = 0.0

    for posicao_id, model_info in models_by_position.items():
        mask = df["posicao_id"] == posicao_id
        if not mask.any():
            continue

        feat_cols = model_info["feature_cols"]
        model = model_info["model"]

        X = model_feature_matrix(
            df.loc[mask],
            feat_cols,
            model_info.get("feature_fill_values"),
        )
        df.loc[mask, "pontos_previstos"] = model.predict(X)

    predicted_points = pd.to_numeric(df["pontos_previstos"], errors="coerce").fillna(0.0)
    if captain_std_column in df.columns:
        captain_std = pd.to_numeric(df[captain_std_column], errors="coerce").fillna(0.0)
    else:
        captain_std = pd.Series(0.0, index=df.index)
    safe_score_column = "safe_score"
    df[captain_score_column] = predicted_points + captain_std_weight * captain_std
    df[safe_score_column] = predicted_points - captain_std_weight * captain_std

    df = df.sort_values("pontos_previstos", ascending=False)

    position_pools = {}

    for position in formation:
        position_pool = df[df["posicao_id"] == position].copy()
        if position in POS_THRESHOLD and odds_filter:
            position_pool = apply_odds_filter(
                position_pool=position_pool,
                min_prob_win=odds_filter["min_prob_win"],
                max_prob_loss=odds_filter["max_prob_loss"],
            )
        position_pools[position] = position_pool

    position_selections = {}
    attack_candidate_pools = {}
    captain_position_candidates = {}
    captain_position_risk_indices = {}

    for position, n_players in formation.items():
        position_pool = position_pools[position]
        if int(position) in LUXURY_RESERVE_POSITIONS:
            captain_candidate_indices, captain_risk_indices, captain_candidates = _select_attack_starters(
                position_pool,
                (
                    CAPTAIN_POSITION_SELECTION_POLICY.get("risk", 0)
                    + CAPTAIN_POSITION_SELECTION_POLICY.get("stable", 0)
                    + CAPTAIN_POSITION_SELECTION_POLICY.get("default", 0)
                ),
                captain_score_column,
                safe_score_column,
                CAPTAIN_POSITION_SELECTION_POLICY["risk"],
                CAPTAIN_POSITION_SELECTION_POLICY["stable"],
                CAPTAIN_POSITION_SELECTION_POLICY.get("default", 0),
            )
            captain_position_candidates[position] = captain_candidate_indices
            captain_position_risk_indices[position] = captain_risk_indices
            attack_candidate_pools[position] = captain_candidates

            risk_count, stable_count, default_count = _attack_selection_counts(
                int(position),
                n_players,
                attack_selection_policy,
            )
            starter_indices, _, _ = _select_attack_starters(
                position_pool,
                n_players,
                captain_score_column,
                safe_score_column,
                risk_count,
                stable_count,
                default_count,
            )
        else:
            starter_indices = position_pool.head(n_players).index.tolist()

        position_selections[position] = {
            "starter_indices": starter_indices,
            "reserve_index": None,
            "reserve_de_luxo": False,
        }

    captain_index = None
    captain_position = None
    if align_luxury_reserve_with_captain and captain_position_risk_indices:
        high_risk_candidate_indices = [
            player_index
            for position_indices in captain_position_risk_indices.values()
            for player_index in position_indices
        ]
        high_risk_rows = df.loc[high_risk_candidate_indices]
        if not high_risk_rows.empty:
            captain_position_index = _score_values(high_risk_rows, captain_score_column).idxmax()
            captain_position = int(df.loc[captain_position_index, "posicao_id"])

            captain_candidate_indices = captain_position_candidates.get(captain_position, [])
            reserve_index = _select_cheapest_reserve_from_candidates(
                position_pools[captain_position],
                captain_candidate_indices,
                reserve_price_column,
            )
            if reserve_index is not None:
                starter_indices = [
                    player_index
                    for player_index in captain_candidate_indices
                    if player_index != reserve_index
                ][: formation[captain_position]]
                position_selections[captain_position]["starter_indices"] = starter_indices
                position_selections[captain_position]["reserve_index"] = reserve_index
                position_selections[captain_position]["reserve_de_luxo"] = True
            else:
                starter_indices = position_selections[captain_position]["starter_indices"]

            captain_index = _best_risk_captain_index(
                df,
                captain_position_risk_indices.get(captain_position, []),
                starter_indices,
                captain_score_column,
            )

    for position, selection in position_selections.items():
        if int(position) == TEC_POSITION_ID:
            continue
        if selection["reserve_index"] is not None:
            continue
        if captain_position is not None and int(position) == captain_position:
            continue

        reserve_score_column = "pontos_previstos"
        reserve_candidate_pool = None
        reserve_de_luxo = False
        if (
            align_luxury_reserve_with_captain
            and captain_position is not None
            and int(position) == captain_position
        ):
            reserve_score_column = safe_score_column
            reserve_candidate_pool = attack_candidate_pools.get(position)
            reserve_de_luxo = True

        reserve_index = _select_reserve_index(
            position_pools[position],
            selection["starter_indices"],
            reserve_price_column,
            reserve_score_column,
            candidate_pool=reserve_candidate_pool,
        )
        if reserve_index is None:
            reserve_de_luxo = False

        selection["reserve_index"] = reserve_index
        selection["reserve_de_luxo"] = reserve_de_luxo

    selected_players = []
    for position, selection in position_selections.items():
        position_pool = position_pools[position]
        selected_indices = list(selection["starter_indices"])
        reserve_index = selection["reserve_index"]
        if reserve_index is not None:
            selected_indices.append(reserve_index)

        for player_index in selected_indices:
            player = position_pool.loc[player_index].copy()
            player["reserva"] = player_index == reserve_index
            player["reserva_de_luxo"] = (
                selection["reserve_de_luxo"]
                and player_index == reserve_index
            )
            player[CAPTAIN_COL] = player_index == captain_index
            selected_players.append(player)

    team_df = pd.DataFrame(selected_players)
    team_df = team_df.sort_values(["reserva", "posicao_id", "pontos_previstos"], ascending=[True, True, False])
    if team_df.empty or not team_df[CAPTAIN_COL].fillna(False).astype(bool).any():
        team_df = assign_captain(team_df, score_column="pontos_previstos")

    if include_reserves:
        return team_df.reset_index(drop=True)

    starters = team_df[~team_df["reserva"].fillna(False).astype(bool)].copy()
    return starters.reset_index(drop=True)


# ──────────────────────────────────────────────
# 4. DISPLAY
# ──────────────────────────────────────────────

def imprimir_time(time_df: pd.DataFrame):
    if "reserva" in time_df.columns:
        scoring_df = time_df[~time_df["reserva"].fillna(False).astype(bool)].copy()
    else:
        scoring_df = time_df

    total_pred = score_with_captain_bonus(scoring_df, points_column="pontos_previstos")
    total_preco = scoring_df["preco"].sum() if "preco" in scoring_df.columns else 0.0
    output_df = lineup_output_table(time_df)

    print("\n" + "="*140)
    print(f"{'CARTOLA FC - TIME':^140}")
    print("="*140)
    if "reserva" in output_df.columns:
        reserve_mask = output_df["reserva"].fillna(False).astype(bool)
        starters_table = output_df[~reserve_mask]
        reserves_table = output_df[reserve_mask]

        print(starters_table.to_string(index=False))
        if not reserves_table.empty:
            print("-"*140)
            print(reserves_table.to_string(index=False, header=True))
    else:
        print(output_df.to_string(index=False))
    print("-"*140)
    print(f"Budget usado: {total_preco:.1f} | Pts previstos com capitao: {total_pred:.2f}")
    print("="*140)


def prepare_market_data(
    df_players_per_round: pd.DataFrame,
    rodada_alvo: int,
    season: int = CURRENT_SEASON,
    df_odds: pd.DataFrame | None = None,
    df_matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    market_file = DATA_DIR / "mercado_atual.parquet"
    market_df = pd.read_parquet(market_file)

    token = os.environ.get("CARTOLA_TOKEN")
    api = CartolaAPI(token=token)
    log.info("Buscando atletas disponíveis no mercado...")
    
    clubes_map = {int(k): v["nome"] for k, v in api.clubes().items()}
    market_df["clube_nome"] = market_df["clube_id"].map(clubes_map).fillna("")
    market_df["temporada"]  = season
    market_df["rodada"]     = rodada_alvo

    live_market_values = market_df[
        [col for col in ["atleta_id", "preco", "media"] if col in market_df.columns]
    ].drop_duplicates("atleta_id")
    market_df = build_target_round_market_features(
        df_players_per_round=df_players_per_round,
        df_matches=df_matches,
        df_odds=df_odds,
        df_market=market_df,
        season=season,
        rodada_alvo=rodada_alvo,
    )

    market_df = market_df.merge(
        live_market_values,
        on="atleta_id",
        how="left",
        suffixes=("", "_atual"),
    )
    for col in ["preco", "media"]:
        current_col = f"{col}_atual"
        if current_col in market_df.columns:
            market_df[col] = market_df[current_col].combine_first(market_df[col])
            market_df = market_df.drop(columns=current_col)

    market_df = merge_target_round_match_context(
        market_df,
        df_matches,
        season,
        rodada_alvo,
        clubes_map,
    )

    return market_df


def main():
    parser = argparse.ArgumentParser(description="Cartola FC team builder")
    parser.add_argument(
        "--model-strategy",
        choices=available_model_strategies(),
        default=DEFAULT_MODEL_STRATEGY,
        help="Metodo de ML usado para treinar os modelos por posicao",
    )
    args = parser.parse_args()

    api = CartolaAPI()
    rodada_alvo = get_current_round(api)
    df_players_per_round, df_matches, df_odds = read_datasets()

    log.info(f"Histórico carregado: {df_players_per_round['temporada'].nunique()} temporadas, "
             f"{df_players_per_round['atleta_id'].nunique()} atletas únicos")
    
    features = build_features(df_players_per_round, df_matches, df_odds)

    models_by_pos = train_models_by_position(
        features,
        round_limit=rodada_alvo,
        season=CURRENT_SEASON,
        tuning=TUNING,
        strategy=args.model_strategy,
    )
    feature_cols = feature_cols_from_models(models_by_pos)
    mae = mean_mae_from_models(models_by_pos)
    log.info(f"MAE médio por posição: {mae:.3f} pts | Features: {len(feature_cols)}")

    market_data = prepare_market_data(
        df_players_per_round,
        rodada_alvo,
        df_odds=df_odds,
        df_matches=df_matches,
    )

    team = build_team(market_data, models_by_pos, include_reserves=True)

    imprimir_time(team)


if __name__ == "__main__":
    main()
