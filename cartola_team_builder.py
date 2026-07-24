import logging
import os
import argparse
import pandas as pd
from dotenv import load_dotenv
import lightgbm as lgb

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from cartola_data.current import get_current_round
from feature_engineering import FEATURE_COLS, build_features

from cartola_data.config import (
    CAPTAIN_BONUS,
    CAPTAIN_POS,
    CURRENT_SEASON,
    DATA_DIR,
    FORMATION,
    GRADIENT_BOOSTING_TUNING,
    POSICAO_NOME,
    RISK_TUNING,
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

DEFAULT_MODEL_STRATEGY = "random_forest"
GRADIENT_BOOSTING_STRATEGY = "gradient_boosting"


class ModelTrainingStrategy:
    """Builds a regression model for Cartola point prediction."""

    name: str
    default_tuning: dict

    def merged_tuning(self, tuning: dict | None = None) -> dict:
        return {**self.default_tuning, **(tuning or {})}

    def build_model(self, tuning: dict | None = None):
        raise NotImplementedError

    def fit_model(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tuning: dict | None = None,
    ):
        model.fit(X_train, y_train)
        return model


class RandomForestTrainingStrategy(ModelTrainingStrategy):
    name = DEFAULT_MODEL_STRATEGY
    default_tuning = TUNING

    def build_model(self, tuning: dict | None = None) -> RandomForestRegressor:
        params = self.merged_tuning(tuning)
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=params["random_state"],
            min_samples_split=params["min_samples_split"],
            max_features=params["max_features"],
            n_jobs=params["n_jobs"],
        )


class GradientBoostingTrainingStrategy(ModelTrainingStrategy):
    name = GRADIENT_BOOSTING_STRATEGY
    default_tuning = GRADIENT_BOOSTING_TUNING

    def build_model(self, tuning: dict | None = None) -> LGBMRegressor:
        params = self.merged_tuning(tuning)
        return LGBMRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            num_leaves=params["num_leaves"],
            min_child_samples=params["min_child_samples"],
            random_state=params["random_state"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            objective=params["objective"],
            metric=params["metric"],
            n_jobs=params["n_jobs"],
            verbosity=params["verbosity"],
        )

    def fit_model(
        self,
        model: LGBMRegressor,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tuning: dict | None = None,
    ) -> LGBMRegressor:
        params = self.merged_tuning(tuning)
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=params["early_stopping_rounds"],
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=params["metric"],
            callbacks=callbacks,
        )
        return model


MODEL_TRAINING_STRATEGIES = {
    DEFAULT_MODEL_STRATEGY: RandomForestTrainingStrategy(),
    GRADIENT_BOOSTING_STRATEGY: GradientBoostingTrainingStrategy(),
}

MODEL_STRATEGY_ALIASES = {
    "rf": DEFAULT_MODEL_STRATEGY,
    "random-forest": DEFAULT_MODEL_STRATEGY,
    "random_forest": DEFAULT_MODEL_STRATEGY,
    "gb": GRADIENT_BOOSTING_STRATEGY,
    "gradient-boosting": GRADIENT_BOOSTING_STRATEGY,
    "gradient_boosting": GRADIENT_BOOSTING_STRATEGY,
}


def available_model_strategies() -> list[str]:
    return list(MODEL_TRAINING_STRATEGIES)


def resolve_model_strategy(
    strategy: str | ModelTrainingStrategy = DEFAULT_MODEL_STRATEGY,
) -> ModelTrainingStrategy:
    if isinstance(strategy, ModelTrainingStrategy):
        return strategy

    requested = str(strategy).lower()
    strategy_key = MODEL_STRATEGY_ALIASES.get(requested, requested)
    if strategy_key not in MODEL_TRAINING_STRATEGIES:
        available = ", ".join(available_model_strategies())
        raise ValueError(f"Estrategia de modelo desconhecida: {strategy}. Opcoes: {available}")
    return MODEL_TRAINING_STRATEGIES[strategy_key]


def model_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce").fillna(0)


def train_model(
    df: pd.DataFrame,
    round_limit: int,
    season: int | None = None,
    tuning: dict | None = None,
    strategy: str | ModelTrainingStrategy = DEFAULT_MODEL_STRATEGY,
):
    """
    Train a regression model with time validation.
    Returns (model, feature_cols, mae).

    When `season` is provided and the dataset has a `temporada` column, all
    previous seasons are used for training. The validation fold is the 3
    rounds immediately before `round_limit` in the target season.
    """
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    df_model = df[df["pontos"].notna()].copy()

    validation_start = round_limit - 3
    if season is not None and "temporada" in df_model.columns:
        training_mask = (
            (df_model["temporada"] < season)
            | (
                (df_model["temporada"] == season)
                & (df_model["rodada"] < validation_start)
            )
        )
        validation_mask = (
            (df_model["temporada"] == season)
            & (df_model["rodada"] >= validation_start)
            & (df_model["rodada"] < round_limit)
        )
        training_df = df_model[training_mask]
        test_df = df_model[validation_mask]
    else:
        training_df = df_model[df_model["rodada"] < validation_start]
        test_df = df_model[
            (df_model["rodada"] >= validation_start)
            & (df_model["rodada"] < round_limit)
        ]

    if training_df.empty or test_df.empty:
        raise ValueError("Dados insuficientes para treino/validação.")

    X_train, y_train = model_feature_matrix(training_df, feat_cols), training_df["pontos"]
    X_val,   y_val   = model_feature_matrix(test_df, feat_cols),     test_df["pontos"]

    training_strategy = resolve_model_strategy(strategy)
    model = training_strategy.build_model(tuning)
    model = training_strategy.fit_model(model, X_train, y_train, X_val, y_val, tuning)

    mae = mean_absolute_error(y_val, model.predict(X_val))
    log.info(f"Validação MAE: {mae:.3f} pts | Features: {len(feat_cols)}")
    return model, feat_cols, mae


def train_models_by_position(
    df: pd.DataFrame,
    round_limit: int,
    season: int | None = None,
    tuning: dict | None = None,
    strategy: str | ModelTrainingStrategy = DEFAULT_MODEL_STRATEGY,
):
    models = {}
    training_strategy = resolve_model_strategy(strategy)

    for posicao_id, df_pos in df.groupby("posicao_id"):
        model, feat_cols, mae = train_model(
            df_pos,
            round_limit,
            season,
            tuning,
            training_strategy,
        )
        models[int(posicao_id)] = {
            "model": model,
            "feature_cols": feat_cols,
            "mae": mae,
            "strategy": training_strategy.name,
        }

    return models


def feature_cols_from_models(models_by_position: dict) -> list[str]:
    feature_cols = []
    for model_info in models_by_position.values():
        for col in model_info["feature_cols"]:
            if col not in feature_cols:
                feature_cols.append(col)
    return feature_cols


def mean_mae_from_models(models_by_position: dict) -> float:
    maes = [model_info["mae"] for model_info in models_by_position.values()]
    if not maes:
        raise ValueError("Nenhum modelo por posição foi treinado.")
    return float(sum(maes) / len(maes))


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


def _price_values(players: pd.DataFrame) -> pd.Series:
    if "preco" not in players.columns:
        return pd.Series(float("inf"), index=players.index)

    return pd.to_numeric(players["preco"], errors="coerce").fillna(float("inf"))


def _luxury_reserve_candidate(
    position_pool: pd.DataFrame,
    n_players: int,
) -> tuple[float, int] | None:
    candidates = position_pool.head(n_players + 1).copy()
    if len(candidates) <= n_players:
        return None

    prices = _price_values(candidates)
    reserve_index = prices.idxmin()
    starter_prices = prices.drop(index=reserve_index)
    if starter_prices.empty or prices.loc[reserve_index] >= starter_prices.min():
        return None

    reserve_points = pd.to_numeric(
        pd.Series([candidates.loc[reserve_index, "pontos_previstos"]]),
        errors="coerce",
    ).fillna(float("-inf")).iloc[0]

    return float(reserve_points), reserve_index


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
) -> pd.DataFrame:
    df = market_df[market_df['status_id'] == STATUS["Provavel"]].copy()
    df["pontos_previstos"] = 0.0

    for posicao_id, model_info in models_by_position.items():
        mask = df["posicao_id"] == posicao_id
        if not mask.any():
            continue

        feat_cols = model_info["feature_cols"]
        model = model_info["model"]

        X = model_feature_matrix(df.loc[mask], feat_cols)
        df.loc[mask, "pontos_previstos"] = model.predict(X)

    df = df.sort_values("pontos_previstos", ascending=False)

    selected_players = []
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

    reserve_candidates = []
    for position in LUXURY_RESERVE_POSITIONS:
        if position not in formation:
            continue

        candidate = _luxury_reserve_candidate(position_pools[position], formation[position])
        if candidate is None:
            continue

        reserve_points, _ = candidate
        reserve_candidates.append((reserve_points, position))

    luxury_reserve_position = None
    if reserve_candidates:
        _, luxury_reserve_position = max(reserve_candidates)

    for position, n_players in formation.items():
        position_pool = position_pools[position]
        chosen = []
        reserve_index = None
        reserve_slots = 0 if int(position) == TEC_POSITION_ID else 1
        selection_limit = n_players + reserve_slots

        if int(position) == luxury_reserve_position:
            candidate = _luxury_reserve_candidate(position_pool, n_players)
            if candidate is not None:
                _, reserve_index = candidate

        for _, player in position_pool.iterrows():            
            if len(chosen) >= selection_limit:
                break

            player = player.copy()
            player["reserva"] = (
                player.name == reserve_index
                if reserve_index is not None
                else len(chosen) >= n_players
            )
            player["reserva_de_luxo"] = player.name == reserve_index
            chosen.append(player)

        selected_players.extend(chosen)

    team_df = pd.DataFrame(selected_players)
    team_df = team_df.sort_values(["reserva", "posicao_id", "pontos_previstos"], ascending=[True, True, False])
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
    df_feat: pd.DataFrame,
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

    feature_cols = ["atleta_id"] + [c for c in FEATURE_COLS if c not in market_df.columns]
    previous_features = df_feat.loc[
        (df_feat["temporada"] == season)
        & (df_feat["rodada"] < rodada_alvo)
    ].copy()
    sort_cols = ["temporada", "rodada"]

    if previous_features.empty:
        ultima_feat = pd.DataFrame(columns=feature_cols)
    else:
        ultima_feat = (
            previous_features
            .sort_values(sort_cols)
            [feature_cols]
            .drop_duplicates("atleta_id", keep="last")
        )

    market_df = market_df.merge(ultima_feat, on="atleta_id", how="left")
    for col in FEATURE_COLS:
        if col in market_df.columns:
            market_df[col] = market_df[col].fillna(
                market_df.groupby("posicao_id")[col].transform("median")
            ).fillna(0)

    market_df = merge_target_round_match_context(
        market_df,
        df_matches,
        season,
        rodada_alvo,
        clubes_map,
    )

    market_df = merge_target_round_odds(market_df, df_odds, season, rodada_alvo)
    for col in ODDS_COLS:
        if col in market_df.columns:
            market_df[col] = market_df[col].fillna(1 / 3)

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
        features,
        rodada_alvo,
        df_odds=df_odds,
        df_matches=df_matches,
    )

    team = build_team(market_data, models_by_pos, include_reserves=True)

    imprimir_time(team)


if __name__ == "__main__":
    main()
