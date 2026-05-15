import logging
import os
import pandas as pd
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from feature_engineering import FEATURE_COLS, build_features

from cartola_data.config import DATA_DIR, FORMATION, POSICAO_NOME, STATUS
from cartola_data.api import CartolaAPI
from cartola_data.datasets import read_datasets

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


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


def build_team(
    market_df: pd.DataFrame,
    models_by_position: dict,
    formation: dict = FORMATION
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
        chosen = []

        for _, player in position_pool.iterrows():            
            if len(chosen) >= n_players:
                break

            chosen.append(player)

        selected_players.extend(chosen)

    team_df = pd.DataFrame(selected_players)
    team_df = team_df.sort_values(["posicao_id", "pontos_previstos"], ascending=[True, False])

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


def prepare_market_data(df_feat: pd.DataFrame, rodada_alvo: int) -> pd.DataFrame:
    market_file = DATA_DIR / "mercado_atual.parquet"
    market_df = pd.read_parquet(market_file)

    token = os.environ.get("CARTOLA_TOKEN")
    api = CartolaAPI(token=token)
    log.info("Buscando atletas disponíveis no mercado...")
    
    clubes_map = {int(k): v["nome"] for k, v in api.clubes().items()}
    market_df["clube_nome"] = market_df["clube_id"].map(clubes_map).fillna("")
    market_df["rodada"]     = rodada_alvo

    ultima_feat = (
        df_feat[df_feat["rodada"] == df_feat["rodada"].max()]
        [["atleta_id"] + [c for c in FEATURE_COLS if c not in market_df.columns]]
        .drop_duplicates("atleta_id")
    )
    market_df = market_df.merge(ultima_feat, on="atleta_id", how="left")
    for col in FEATURE_COLS:
        if col in market_df.columns:
            market_df[col] = market_df[col].fillna(
                market_df.groupby("posicao_id")[col].transform("median")
            ).fillna(0)

    return market_df


def main():
    df_players_per_round, df_matches, df_odds = read_datasets()

    log.info(f"Histórico carregado: {df_players_per_round['rodada'].nunique()} rodadas, "
             f"{df_players_per_round['atleta_id'].nunique()} atletas únicos")
    
    features = build_features(df_players_per_round, df_matches, df_odds)

    models_by_pos = train_models_by_position(features, 15)
    feature_cols = feature_cols_from_models(models_by_pos)
    mae = mean_mae_from_models(models_by_pos)
    log.info(f"MAE médio por posição: {mae:.3f} pts | Features: {len(feature_cols)}")

    market_data = prepare_market_data(features, 16)

    team = build_team(market_data, models_by_pos)

    imprimir_time(team)


if __name__ == "__main__":
    main()
