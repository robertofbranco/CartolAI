import logging
import os
import pandas as pd
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from feature_engineering import FEATURE_COLS, build_features

from cartola_data.config import (
    CURRENT_SEASON,
    DATA_DIR,
    FORMATION,
    POSICAO_NOME,
    STATUS,
    ODDS_FILTER,
)
from cartola_data.api import CartolaAPI
from cartola_data.datasets import read_datasets

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ODDS_COLS = ["prob_win", "prob_draw", "prob_loss"]
POS_THRESHOLD = [1, 3, 6]
TEC_POSITION_ID = 6


def train_model(df: pd.DataFrame, round_limit: int, season: int | None = None):
    """
    Train RandomForest with time validation.
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

    X_train, y_train = training_df[feat_cols].fillna(0), training_df["pontos"]
    X_val,   y_val   = test_df[feat_cols].fillna(0),     test_df["pontos"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_val, model.predict(X_val))
    log.info(f"Validação MAE: {mae:.3f} pts | Features: {len(feat_cols)}")
    return model, feat_cols, mae


def train_models_by_position(df: pd.DataFrame, round_limit: int, season: int | None = None):
    models = {}

    for posicao_id, df_pos in df.groupby("posicao_id"):
        model, feat_cols, mae = train_model(df_pos, round_limit, season)
        models[int(posicao_id)] = {
            "model": model,
            "feature_cols": feat_cols,
            "mae": mae,
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


def apply_reserve_substitutions(
    team_df: pd.DataFrame,
    play_status_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Return the scoring lineup after same-position reserve substitutions.

    Each non-TEC position can have at most one reserve, and that reserve can
    replace at most one starter whose `jogou` flag is explicitly False.
    """
    if team_df.empty:
        return team_df.copy()

    team_df = team_df.copy()
    if "reserva" not in team_df.columns:
        team_df["reserva"] = False

    played_by_athlete = _played_lookup(team_df, play_status_df)
    starters = team_df[~team_df["reserva"].fillna(False).astype(bool)].copy()
    reserves = team_df[team_df["reserva"].fillna(False).astype(bool)].copy()
    final_players = []

    for position, position_starters in starters.groupby("posicao_id", sort=False):
        position_starters = position_starters.copy()
        replacement_row = None
        replaced_index = None

        if int(position) != TEC_POSITION_ID:
            position_reserves = reserves[reserves["posicao_id"] == position]
            if not position_reserves.empty:
                reserve = position_reserves.iloc[0].copy()
                for starter_index, starter in position_starters.iterrows():
                    played_value = played_by_athlete.get(
                        starter["atleta_id"],
                        starter.get("jogou", pd.NA),
                    )
                    if _did_not_play(played_value):
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
        model: RandomForestRegressor = model_info["model"]

        X = df.loc[mask].reindex(columns=feat_cols).fillna(0)
        df.loc[mask, "pontos_previstos"] = model.predict(X)

    df = df.sort_values("pontos_previstos", ascending=False)

    selected_players = []

    for position, n_players in formation.items():        
        position_pool = df[df["posicao_id"] == position].copy()
        if position in POS_THRESHOLD and odds_filter:
            position_pool = apply_odds_filter(
                position_pool=position_pool,
                min_prob_win=odds_filter["min_prob_win"],
                max_prob_loss=odds_filter["max_prob_loss"],
            )
        chosen = []
        selection_limit = n_players + (0 if int(position) == TEC_POSITION_ID else 1)

        for _, player in position_pool.iterrows():            
            if len(chosen) >= selection_limit:
                break

            player = player.copy()
            player["reserva"] = len(chosen) >= n_players
            chosen.append(player)

        selected_players.extend(chosen)

    team_df = pd.DataFrame(selected_players)
    team_df = team_df.sort_values(["posicao_id", "pontos_previstos"], ascending=[True, False])

    if include_reserves:
        return team_df.reset_index(drop=True)

    team_df = apply_reserve_substitutions(team_df)

    return team_df


# ──────────────────────────────────────────────
# 4. DISPLAY
# ──────────────────────────────────────────────

def imprimir_time(time_df: pd.DataFrame):
    total_pred  = time_df["pontos_previstos"].sum()
    total_preco = time_df["preco"].sum()

    print("\n" + "="*60)
    print(f"{'CARTOLA FC - TIME':^60}")
    print("="*60)
    print(f"{'Pos':<6} {'Apelido':<22} {'Clube':<18} {'Preço':>6} {'Pts Prev':>8}")
    print("-"*60)
    for _, row in time_df.iterrows():
        pos      = POSICAO_NOME.get(row["posicao_id"], "?")       
        print(f"{pos:<6} {row['apelido'][:20]:<22} "
              f"{row['clube_nome'][:16]:<18} "
              f"{row['preco']:>6.1f} {row['pontos_previstos']:>8.2f}")
    print("-"*60)
    print(f"{'TOTAL':<50} {total_preco:>6.1f} {total_pred:>8.2f}")
    print("="*60)


def prepare_market_data(
    df_feat: pd.DataFrame,
    rodada_alvo: int,
    season: int = CURRENT_SEASON,
    df_odds: pd.DataFrame | None = None,
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

    market_df = merge_target_round_odds(market_df, df_odds, season, rodada_alvo)
    for col in ODDS_COLS:
        if col in market_df.columns:
            market_df[col] = market_df[col].fillna(1 / 3)

    return market_df


def main():
    rodada_alvo = 16
    df_players_per_round, df_matches, df_odds = read_datasets()

    log.info(f"Histórico carregado: {df_players_per_round['rodada'].nunique()} rodadas, "
             f"{df_players_per_round['atleta_id'].nunique()} atletas únicos")
    
    features = build_features(df_players_per_round, df_matches, df_odds)

    models_by_pos = train_models_by_position(
        features,
        round_limit=rodada_alvo,
        season=CURRENT_SEASON,
    )
    feature_cols = feature_cols_from_models(models_by_pos)
    mae = mean_mae_from_models(models_by_pos)
    log.info(f"MAE médio por posição: {mae:.3f} pts | Features: {len(feature_cols)}")

    market_data = prepare_market_data(features, rodada_alvo, df_odds=df_odds)

    team = build_team(market_data, models_by_pos)

    imprimir_time(team)


if __name__ == "__main__":
    main()
