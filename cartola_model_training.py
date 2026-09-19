import logging

import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from cartola_data.config import GRADIENT_BOOSTING_TUNING, TUNING
from feature_engineering import (
    FEATURE_COLS,
    FORM_FEATURE_DEFAULTS,
    FORM_ZERO_DEFAULT_FEATURE_COLS,
    GOL_FEATURE_COLS,
    TEC_FEATURE_COLS,
)

log = logging.getLogger(__name__)

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


def fit_feature_fill_values(
    training_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float]:
    """Learn feature defaults from training data only."""
    fill_values = {}
    for col in feature_cols:
        if col in FORM_FEATURE_DEFAULTS:
            fill_values[col] = float(FORM_FEATURE_DEFAULTS[col])
        elif col in FORM_ZERO_DEFAULT_FEATURE_COLS:
            fill_values[col] = 0.0
        else:
            values = pd.to_numeric(training_df[col], errors="coerce").astype(float)
            median = values.median()
            fill_values[col] = float(median) if pd.notna(median) else 0.0
    return fill_values


def model_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    fill_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build a numeric matrix using training-fitted or semantic defaults."""
    matrix = df.reindex(columns=feature_cols).apply(
        lambda values: pd.to_numeric(values, errors="coerce").astype(float)
    )
    defaults = {
        col: (
            FORM_FEATURE_DEFAULTS.get(col, 0.0)
            if col in FORM_ZERO_DEFAULT_FEATURE_COLS
            else 0.0
        )
        for col in feature_cols
    }
    if fill_values:
        defaults.update(fill_values)
    return matrix.fillna(defaults).fillna(0.0)


def train_model(
    df: pd.DataFrame,
    round_limit: int,
    feature_cols: list[str],
    season: int | None = None,
    tuning: dict | None = None,
    strategy: str | ModelTrainingStrategy = DEFAULT_MODEL_STRATEGY,
):
    """
    Train a regression model with time validation.
    Returns (model, feature_cols, mae).

    When `season` is provided and the dataset has a `temporada` column, all
    previous seasons are used for training. The validation fold is the 5
    rounds immediately before `round_limit` in the target season.
    """
    feat_cols = [c for c in feature_cols if c in df.columns]
    df_model = df[df["pontos"].notna()].copy()

    validation_start = round_limit - 5
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
        raise ValueError("Dados insuficientes para treino/validacao.")

    fill_values = fit_feature_fill_values(training_df, feat_cols)
    X_train = model_feature_matrix(training_df, feat_cols, fill_values)
    X_val = model_feature_matrix(test_df, feat_cols, fill_values)
    y_train, y_val = training_df["pontos"], test_df["pontos"]

    training_strategy = resolve_model_strategy(strategy)
    model = training_strategy.build_model(tuning)
    model = training_strategy.fit_model(model, X_train, y_train, X_val, y_val, tuning)

    mae = mean_absolute_error(y_val, model.predict(X_val))
    log.info(f"Validacao MAE: {mae:.3f} pts | Features: {len(feat_cols)}")
    return model, feat_cols, mae, fill_values


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
        feature_cols = FEATURE_COLS
        if posicao_id == 1:
            feature_cols = GOL_FEATURE_COLS
        elif posicao_id == 6:
            feature_cols = TEC_FEATURE_COLS

        model, feat_cols, mae, fill_values = train_model(
            df_pos,
            round_limit,
            feature_cols,
            season,
            tuning,
            training_strategy,
        )
        models[int(posicao_id)] = {
            "model": model,
            "feature_cols": feat_cols,
            "mae": mae,
            "feature_fill_values": fill_values,
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
        raise ValueError("Nenhum modelo por posicao foi treinado.")
    return float(sum(maes) / len(maes))
