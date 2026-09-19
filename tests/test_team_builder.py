import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from cartola_data.config import CAPTAIN_BONUS
from cartola_data.config import STATUS
import cartola_team_builder as team_builder
from cartola_model_training import (
    DEFAULT_MODEL_STRATEGY,
    GRADIENT_BOOSTING_STRATEGY,
    fit_feature_fill_values,
    model_feature_matrix,
    resolve_model_strategy,
    train_models_by_position,
)
from cartola_team_builder import (
    assign_captain,
    apply_reserve_substitutions,
    build_target_round_market_features,
    build_team,
    imprimir_time,
    lineup_output_table,
    merge_target_round_match_context,
    prepare_market_data,
    refresh_opponent_encoding_from_features,
    score_with_captain_bonus,
)


class ScoreModel:
    def predict(self, X):
        return X["score"].to_numpy()


class ModelFeatureImputationTest(unittest.TestCase):
    def test_uses_training_median_and_semantic_form_defaults(self):
        training = pd.DataFrame(
            {
                "media_lag": pd.Series([1, 2, pd.NA], dtype="Int64"),
                "aparicoes_5r": pd.Series([1, 2, pd.NA], dtype="Int64"),
                "rodadas_desde_ultima_aparicao": pd.Series(
                    [1, 2, pd.NA], dtype="Int64"
                ),
            }
        )
        feature_cols = [
            "media_lag",
            "aparicoes_5r",
            "rodadas_desde_ultima_aparicao",
        ]

        fill_values = fit_feature_fill_values(training, feature_cols)
        result = model_feature_matrix(training, feature_cols, fill_values)

        self.assertEqual(result["media_lag"].dtype, "float64")
        self.assertEqual(result.loc[2, "media_lag"], 1.5)
        self.assertEqual(result.loc[2, "aparicoes_5r"], 0.0)
        self.assertEqual(result.loc[2, "rodadas_desde_ultima_aparicao"], 10.0)


def training_frame():
    rows = []
    for posicao_id in [4, 5]:
        for atleta_id in range(1, 4):
            for rodada in range(1, 10):
                base_score = atleta_id + rodada + posicao_id / 10
                rows.append(
                    {
                        "temporada": 2026,
                        "rodada": rodada,
                        "atleta_id": posicao_id * 100 + atleta_id,
                        "posicao_id": posicao_id,
                        "pontos": base_score * 1.2,
                        "media_pts_5r": base_score,
                        "media_pts_3r": base_score - 0.2,
                        "mando": 1 if rodada % 2 else -1,
                    }
                )
    return pd.DataFrame(rows)


class ModelTrainingStrategyTest(unittest.TestCase):
    def test_random_forest_strategy_remains_default(self):
        strategy = resolve_model_strategy(DEFAULT_MODEL_STRATEGY)
        model = strategy.build_model(
            {
                "n_estimators": 5,
                "max_depth": 3,
                "min_samples_leaf": 1,
                "random_state": 42,
                "min_samples_split": 2,
                "max_features": 1.0,
                "n_jobs": 1,
            }
        )

        self.assertIsInstance(model, RandomForestRegressor)

    def test_train_models_by_position_supports_gradient_boosting(self):
        models = train_models_by_position(
            training_frame(),
            round_limit=8,
            season=2026,
            tuning={
                "n_estimators": 5,
                "max_depth": 2,
                "random_state": 42,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "num_leaves": 7,
                "min_child_samples": 1,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "objective": "regression",
                "metric": "mae",
                "early_stopping_rounds": 3,
                "n_jobs": 1,
                "verbosity": -1,
            },
            strategy=GRADIENT_BOOSTING_STRATEGY,
        )

        self.assertEqual(set(models), {4, 5})
        for model_info in models.values():
            self.assertIsInstance(model_info["model"], LGBMRegressor)
            self.assertEqual(model_info["strategy"], GRADIENT_BOOSTING_STRATEGY)
            self.assertIn("media_pts_5r", model_info["feature_cols"])
            self.assertGreaterEqual(model_info["mae"], 0.0)


class TargetRoundFeaturesTest(unittest.TestCase):
    @staticmethod
    def player_history() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "atleta_id": 100,
                    "apelido": "Mei 1",
                    "posicao_id": 4,
                    "clube_id": 1,
                    "status_id": STATUS["Provavel"],
                    "pontos": 5.0,
                    "jogou": True,
                    "preco": 10.0,
                    "media": 5.0,
                    "jogos": 1,
                    "scout_G": 0.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "atleta_id": 100,
                    "apelido": "Mei 1",
                    "posicao_id": 4,
                    "clube_id": 1,
                    "status_id": STATUS["Provavel"],
                    "pontos": 7.0,
                    "jogou": True,
                    "preco": 12.0,
                    "media": 6.0,
                    "jogos": 2,
                    "scout_G": 1.0,
                },
            ]
        )

    @staticmethod
    def matches() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": rodada,
                    "clube_id": 1,
                    "mando": 1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 1,
                    "gols_sofridos_clube": 0,
                }
                for rodada in [1, 2, 3]
            ]
        )

    def test_target_round_includes_previous_round_results_only(self):
        target_market = pd.DataFrame(
            [
                {
                    "atleta_id": 100,
                    "apelido": "Mei 1",
                    "posicao_id": 4,
                    "clube_id": 1,
                    "status_id": STATUS["Provavel"],
                    "preco": 50.0,
                    "media": 40.0,
                    "jogos": 3,
                    "pontos": 100.0,
                    "jogou": True,
                    "scout_G": 100.0,
                }
            ]
        )

        market = build_target_round_market_features(
            df_players_per_round=self.player_history(),
            df_matches=self.matches(),
            df_odds=pd.DataFrame(
                columns=["temporada", "rodada", "clube_id", "prob_win", "prob_draw", "prob_loss"]
            ),
            df_market=target_market,
            season=2026,
            rodada_alvo=3,
        )

        row = market.iloc[0]
        self.assertEqual(row["media_pts_3r"], 6.0)
        self.assertEqual(row["pts_ultima_rodada"], 7.0)
        self.assertEqual(row["scout_G_3r"], 0.5)
        self.assertTrue(pd.isna(row["pontos"]))
        self.assertEqual(row["preco_lag1"], 12.0)
        self.assertEqual(row["preco"], 12.0)

    def test_prepare_market_data_builds_target_features_and_keeps_live_price(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "atleta_id": 100,
                        "apelido": "Mei 1",
                        "posicao_id": 4,
                        "clube_id": 1,
                        "status_id": STATUS["Provavel"],
                        "preco": 13.0,
                        "media": 6.5,
                        "jogos": 2,
                    }
                ]
            ).to_parquet(data_dir / "mercado_atual.parquet", index=False)
            api = MagicMock()
            api.clubes.return_value = {"1": {"nome": "Clube A"}, "2": {"nome": "Clube B"}}

            with (
                patch.object(team_builder, "DATA_DIR", data_dir),
                patch.object(team_builder, "CartolaAPI", return_value=api),
            ):
                market = prepare_market_data(
                    self.player_history(),
                    rodada_alvo=3,
                    season=2026,
                    df_odds=pd.DataFrame(
                        columns=["temporada", "rodada", "clube_id", "prob_win", "prob_draw", "prob_loss"]
                    ),
                    df_matches=self.matches(),
                )

        row = market.iloc[0]
        self.assertEqual(row["media_pts_3r"], 6.0)
        self.assertEqual(row["preco_lag1"], 12.0)
        self.assertEqual(row["preco"], 13.0)
        self.assertEqual(row["media"], 6.5)
        self.assertEqual(row["status_id"], STATUS["Provavel"])
        self.assertEqual(row["adversario"], "Clube B")

    def test_prepare_market_data_keeps_goalkeeper_attack_volume_features(self):
        history = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": rodada,
                    "atleta_id": atleta_id,
                    "apelido": apelido,
                    "posicao_id": posicao_id,
                    "clube_id": clube_id,
                    "status_id": STATUS["Provavel"],
                    "pontos": pontos,
                    "jogou": True,
                    "preco": preco,
                    "media": pontos,
                    "jogos": rodada,
                    "scout_FD": scout_fd,
                }
                for rodada, values in [
                    (1, [(100, "Gol 1", 1, 1, 5.0, 10.0, 0.0), (200, "Ata 1", 5, 2, 3.0, 8.0, 2.0)]),
                    (2, [(100, "Gol 1", 1, 1, 6.0, 11.0, 0.0), (200, "Ata 1", 5, 2, 4.0, 9.0, 5.0)]),
                ]
                for atleta_id, apelido, posicao_id, clube_id, pontos, preco, scout_fd in values
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": rodada,
                    "clube_id": clube_id,
                    "mando": mando,
                    "clube_adversario_id": adversario_id,
                    "gols_feitos_clube": 1,
                    "gols_sofridos_clube": 0,
                }
                for rodada in [1, 2, 3]
                for clube_id, mando, adversario_id in [(1, 1, 2), (2, -1, 1)]
            ]
        )

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "atleta_id": atleta_id,
                        "apelido": apelido,
                        "posicao_id": posicao_id,
                        "clube_id": clube_id,
                        "status_id": STATUS["Provavel"],
                        "preco": preco,
                        "media": media,
                        "jogos": 2,
                    }
                    for atleta_id, apelido, posicao_id, clube_id, preco, media in [
                        (100, "Gol 1", 1, 1, 12.0, 5.5),
                        (200, "Ata 1", 5, 2, 10.0, 3.5),
                    ]
                ]
            ).to_parquet(data_dir / "mercado_atual.parquet", index=False)
            api = MagicMock()
            api.clubes.return_value = {"1": {"nome": "Clube A"}, "2": {"nome": "Clube B"}}

            with (
                patch.object(team_builder, "DATA_DIR", data_dir),
                patch.object(team_builder, "CartolaAPI", return_value=api),
            ):
                market = prepare_market_data(
                    history,
                    rodada_alvo=3,
                    season=2026,
                    df_odds=pd.DataFrame(
                        columns=["temporada", "rodada", "clube_id", "prob_win", "prob_draw", "prob_loss"]
                    ),
                    df_matches=matches,
                )

        goalkeeper = market.loc[market["atleta_id"] == 100].iloc[0]
        self.assertEqual(goalkeeper["FD_avd_3r"], 2.5)
        self.assertEqual(goalkeeper["FD_adv_5r"], 2.5)


class TeamBuilderReservesTest(unittest.TestCase):
    def test_build_team_selects_attack_starters_by_high_risk_and_safe_scores(self):
        market = pd.DataFrame(
            [
                {"atleta_id": 101, "apelido": "Gol 1", "posicao_id": 1, "status_id": STATUS["Provavel"], "score": 9.0, "preco": 12.0},
                {"atleta_id": 102, "apelido": "Gol 2", "posicao_id": 1, "status_id": STATUS["Provavel"], "score": 8.0, "preco": 1.0},
                {"atleta_id": 201, "apelido": "Lat 1", "posicao_id": 2, "status_id": STATUS["Provavel"], "score": 9.0, "preco": 12.0},
                {"atleta_id": 202, "apelido": "Lat 2", "posicao_id": 2, "status_id": STATUS["Provavel"], "score": 8.0, "preco": 11.0},
                {"atleta_id": 203, "apelido": "Lat 3", "posicao_id": 2, "status_id": STATUS["Provavel"], "score": 7.0, "preco": 1.0},
                {"atleta_id": 204, "apelido": "Lat 4", "posicao_id": 2, "status_id": STATUS["Provavel"], "score": 6.0, "preco": 1.0},
                {"atleta_id": 401, "apelido": "Mei 1", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 10.0, "std_pts_5r": 0.0, "preco": 20.0},
                {"atleta_id": 402, "apelido": "Mei 2", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 9.5, "std_pts_5r": 0.0, "preco": 19.0},
                {"atleta_id": 403, "apelido": "Mei 3", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 9.0, "std_pts_5r": 0.0, "preco": 18.0},
                {"atleta_id": 404, "apelido": "Mei 4", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 8.5, "std_pts_5r": 0.0, "preco": 1.0},
                {"atleta_id": 405, "apelido": "Mei 5", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 8.0, "std_pts_5r": 0.0, "preco": 17.0},
                {"atleta_id": 501, "apelido": "Ata 1", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 10.0, "std_pts_5r": 8.0, "preco": 20.0},
                {"atleta_id": 502, "apelido": "Ata 2", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.9, "std_pts_5r": 0.0, "preco": 19.0},
                {"atleta_id": 503, "apelido": "Ata 3", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.8, "std_pts_5r": 0.0, "preco": 18.0},
                {"atleta_id": 504, "apelido": "Ata 4", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.7, "std_pts_5r": 0.0, "preco": 1.0},
                {"atleta_id": 505, "apelido": "Ata 5", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.6, "std_pts_5r": 0.0, "preco": 17.0},
                {"atleta_id": 601, "apelido": "Tec 1", "posicao_id": 6, "status_id": STATUS["Provavel"], "score": 9.0, "preco": 12.0},
                {"atleta_id": 602, "apelido": "Tec 2", "posicao_id": 6, "status_id": STATUS["Provavel"], "score": 8.0, "preco": 1.0},
            ]
        )
        formation = {1: 1, 2: 2, 4: 3, 5: 3, 6: 1}
        models = {
            position: {"model": ScoreModel(), "feature_cols": ["score"], "mae": 0.0}
            for position in formation
        }

        team = build_team(
            market,
            models,
            formation=formation,
            odds_filter=None,
            include_reserves=True,
        )

        reserves_by_position = team[team["reserva"]].groupby("posicao_id").size().to_dict()
        self.assertEqual(reserves_by_position, {1: 1, 2: 1, 4: 1, 5: 1})
        self.assertEqual(team[team["posicao_id"] == 6]["reserva"].sum(), 0)
        self.assertEqual(team["reserva_de_luxo"].sum(), 1)
        self.assertEqual(team.loc[team["reserva_de_luxo"], "atleta_id"].iloc[0], 504)
        self.assertCountEqual(
            team[(team["posicao_id"] == 4) & ~team["reserva"]]["atleta_id"].tolist(),
            [401, 402, 403],
        )
        self.assertCountEqual(
            team[(team["posicao_id"] == 5) & ~team["reserva"]]["atleta_id"].tolist(),
            [501, 502, 503],
        )
        self.assertLess(
            team.loc[team["reserva_de_luxo"], "preco"].iloc[0],
            team[(team["posicao_id"] == 5) & ~team["reserva"]]["preco"].min(),
        )
        self.assertEqual(team["capitao"].sum(), 1)
        self.assertEqual(team.loc[team["capitao"], "atleta_id"].iloc[0], 501)
        self.assertFalse(team.loc[team["capitao"], "reserva"].iloc[0])

    def test_assign_captain_uses_best_predicted_starter(self):
        team = pd.DataFrame(
            [
                {"atleta_id": 401, "posicao_id": 4, "reserva": False, "pontos_previstos": 8.0},
                {"atleta_id": 402, "posicao_id": 4, "reserva": True, "pontos_previstos": 11.0},
                {"atleta_id": 501, "posicao_id": 5, "reserva": False, "pontos_previstos": 9.0},
            ]
        )

        team = assign_captain(team)

        self.assertEqual(team["capitao"].sum(), 1)
        self.assertEqual(team.loc[team["capitao"], "atleta_id"].iloc[0], 501)

    def test_attack_selection_uses_safe_score_after_high_risk_pick(self):
        market = pd.DataFrame(
            [
                {"atleta_id": 501, "apelido": "Ata 1", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 10.0, "std_pts_5r": 8.0, "preco": 20.0},
                {"atleta_id": 502, "apelido": "Ata 2", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.9, "std_pts_5r": 0.0, "preco": 19.0},
                {"atleta_id": 503, "apelido": "Ata 3", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.8, "std_pts_5r": 0.0, "preco": 18.0},
                {"atleta_id": 504, "apelido": "Ata 4", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.7, "std_pts_5r": 0.0, "preco": 1.0},
                {"atleta_id": 505, "apelido": "Ata 5", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.6, "std_pts_5r": 0.0, "preco": 17.0},
            ]
        )
        formation = {5: 3}
        models = {5: {"model": ScoreModel(), "feature_cols": ["score"], "mae": 0.0}}

        team = build_team(
            market,
            models,
            formation=formation,
            odds_filter=None,
            include_reserves=True,
        )

        starters = team[~team["reserva"]]["atleta_id"].tolist()

        self.assertCountEqual(starters, [501, 502, 503])
        self.assertEqual(team.loc[team["capitao"], "atleta_id"].iloc[0], 501)

    def test_attack_selection_policy_can_parameterize_risk_and_stable_counts_by_position(self):
        market = pd.DataFrame(
            [
                {"atleta_id": 401, "apelido": "Mei 1", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 10.0, "std_pts_5r": 8.0, "preco": 20.0},
                {"atleta_id": 402, "apelido": "Mei 2", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 9.9, "std_pts_5r": 7.0, "preco": 19.0},
                {"atleta_id": 403, "apelido": "Mei 3", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 9.8, "std_pts_5r": 0.0, "preco": 18.0},
                {"atleta_id": 404, "apelido": "Mei 4", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 9.7, "std_pts_5r": 0.0, "preco": 1.0},
                {"atleta_id": 405, "apelido": "Mei 5", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 9.6, "std_pts_5r": 0.0, "preco": 17.0},
                {"atleta_id": 501, "apelido": "Ata 1", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 10.0, "std_pts_5r": 8.0, "preco": 20.0},
                {"atleta_id": 502, "apelido": "Ata 2", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.9, "std_pts_5r": 7.0, "preco": 19.0},
                {"atleta_id": 503, "apelido": "Ata 3", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.8, "std_pts_5r": 0.0, "preco": 18.0},
                {"atleta_id": 504, "apelido": "Ata 4", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.7, "std_pts_5r": 0.0, "preco": 1.0},
                {"atleta_id": 505, "apelido": "Ata 5", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.6, "std_pts_5r": 0.0, "preco": 17.0},
            ]
        )
        formation = {4: 3, 5: 3}
        models = {
            position: {"model": ScoreModel(), "feature_cols": ["score"], "mae": 0.0}
            for position in formation
        }

        team = build_team(
            market,
            models,
            formation=formation,
            odds_filter=None,
            include_reserves=True,
            attack_selection_policy={
                4: {"risk": 2, "stable": 1},
                5: {"risk": 0, "stable": 3},
            },
        )

        mei_starters = team[(team["posicao_id"] == 4) & ~team["reserva"]]["atleta_id"].tolist()
        ata_starters = team[(team["posicao_id"] == 5) & ~team["reserva"]]["atleta_id"].tolist()

        self.assertCountEqual(mei_starters, [401, 402, 403])
        self.assertCountEqual(ata_starters, [503, 504, 505])
        self.assertEqual(team.loc[team["capitao"], "atleta_id"].iloc[0], 401)

    def test_luxury_reserve_is_cheapest_of_captain_position_candidates(self):
        market = pd.DataFrame(
            [
                {"atleta_id": 501, "apelido": "Ata 1", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 10.0, "std_pts_5r": 8.0, "preco": 20.0},
                {"atleta_id": 502, "apelido": "Ata 2", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.9, "std_pts_5r": 0.0, "preco": 19.0},
                {"atleta_id": 503, "apelido": "Ata 3", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.8, "std_pts_5r": 0.0, "preco": 18.0},
                {"atleta_id": 504, "apelido": "Ata 4", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.7, "std_pts_5r": 1.0, "preco": 1.0},
                {"atleta_id": 505, "apelido": "Ata 5", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.6, "std_pts_5r": 0.0, "preco": 17.0},
            ]
        )
        formation = {5: 3}
        models = {5: {"model": ScoreModel(), "feature_cols": ["score"], "mae": 0.0}}

        team = build_team(
            market,
            models,
            formation=formation,
            odds_filter=None,
            include_reserves=True,
        )

        luxury_reserve = team.loc[team["reserva_de_luxo"]].iloc[0]

        self.assertEqual(luxury_reserve["atleta_id"], 504)
        self.assertEqual(luxury_reserve["posicao_id"], 5)
        self.assertLess(luxury_reserve["preco"], team[~team["reserva"]]["preco"].min())

    def test_non_captain_mei_uses_one_risk_and_two_default_when_captain_is_ata(self):
        market = pd.DataFrame(
            [
                {"atleta_id": 401, "apelido": "Mei 1", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 10.0, "std_pts_5r": 8.0, "preco": 20.0},
                {"atleta_id": 402, "apelido": "Mei 2", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 9.9, "std_pts_5r": 7.0, "preco": 19.0},
                {"atleta_id": 403, "apelido": "Mei 3", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 9.8, "std_pts_5r": 0.0, "preco": 18.0},
                {"atleta_id": 404, "apelido": "Mei 4", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 9.7, "std_pts_5r": 0.0, "preco": 17.0},
                {"atleta_id": 405, "apelido": "Mei 5", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 9.6, "std_pts_5r": 0.0, "preco": 1.0},
                {"atleta_id": 501, "apelido": "Ata 1", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 12.0, "std_pts_5r": 8.0, "preco": 20.0},
                {"atleta_id": 502, "apelido": "Ata 2", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 11.9, "std_pts_5r": 7.0, "preco": 19.0},
                {"atleta_id": 503, "apelido": "Ata 3", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 11.8, "std_pts_5r": 0.0, "preco": 18.0},
                {"atleta_id": 504, "apelido": "Ata 4", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 11.7, "std_pts_5r": 0.0, "preco": 1.0},
                {"atleta_id": 505, "apelido": "Ata 5", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 11.6, "std_pts_5r": 0.0, "preco": 17.0},
            ]
        )
        formation = {4: 3, 5: 3}
        models = {
            position: {"model": ScoreModel(), "feature_cols": ["score"], "mae": 0.0}
            for position in formation
        }

        team = build_team(
            market,
            models,
            formation=formation,
            odds_filter=None,
            include_reserves=True,
        )

        mei_starters = team[(team["posicao_id"] == 4) & ~team["reserva"]]["atleta_id"].tolist()

        self.assertEqual(team.loc[team["capitao"], "posicao_id"].iloc[0], 5)
        self.assertCountEqual(mei_starters, [401, 402, 403])

    def test_reserves_must_be_cheaper_than_cheapest_selected_starter(self):
        market = pd.DataFrame(
            [
                {"atleta_id": 201, "apelido": "Lat 1", "posicao_id": 2, "status_id": STATUS["Provavel"], "score": 10.0, "preco": 20.0},
                {"atleta_id": 202, "apelido": "Lat 2", "posicao_id": 2, "status_id": STATUS["Provavel"], "score": 9.0, "preco": 19.0},
                {"atleta_id": 203, "apelido": "Lat 3", "posicao_id": 2, "status_id": STATUS["Provavel"], "score": 8.0, "preco": 19.0},
                {"atleta_id": 204, "apelido": "Lat 4", "posicao_id": 2, "status_id": STATUS["Provavel"], "score": 7.0, "preco": 18.0},
            ]
        )
        formation = {2: 2}
        models = {2: {"model": ScoreModel(), "feature_cols": ["score"], "mae": 0.0}}

        team = build_team(
            market,
            models,
            formation=formation,
            odds_filter=None,
            include_reserves=True,
        )

        self.assertEqual(team["reserva"].sum(), 1)
        self.assertEqual(team.loc[team["reserva"], "atleta_id"].iloc[0], 204)

    def test_apply_reserve_substitutions_uses_luxury_reserve_when_he_scores_more(self):
        team = pd.DataFrame(
            [
                {"atleta_id": 401, "apelido": "Mei 1", "posicao_id": 4, "reserva": False, "pontos_previstos": 9.0},
                {"atleta_id": 402, "apelido": "Mei 2", "posicao_id": 4, "reserva": False, "pontos_previstos": 8.0},
                {"atleta_id": 403, "apelido": "Mei 3", "posicao_id": 4, "reserva": True, "reserva_de_luxo": True, "pontos_previstos": 7.0},
            ]
        )
        scores = pd.DataFrame(
            [
                {"atleta_id": 401, "pontos": 10.0},
                {"atleta_id": 402, "pontos": 5.0},
                {"atleta_id": 403, "pontos": 7.0},
            ]
        )

        final_team = apply_reserve_substitutions(team, scores)

        self.assertCountEqual(final_team["atleta_id"].tolist(), [401, 403])
        luxury_reserve = final_team[final_team["atleta_id"] == 403].iloc[0]
        self.assertTrue(luxury_reserve["reserva"])
        self.assertEqual(luxury_reserve["substituiu_atleta_id"], 402)

    def test_apply_reserve_substitutions_moves_captain_to_reserve(self):
        team = pd.DataFrame(
            [
                {"atleta_id": 401, "apelido": "Mei 1", "posicao_id": 4, "reserva": False, "capitao": True, "pontos_previstos": 9.0},
                {"atleta_id": 402, "apelido": "Mei 2", "posicao_id": 4, "reserva": True, "reserva_de_luxo": True, "capitao": False, "pontos_previstos": 7.0},
            ]
        )
        scores = pd.DataFrame(
            [
                {"atleta_id": 401, "pontos": 6.0},
                {"atleta_id": 402, "pontos": 7.0},
            ]
        )

        final_team = apply_reserve_substitutions(team, scores)

        self.assertEqual(final_team["atleta_id"].tolist(), [402])
        self.assertTrue(final_team.iloc[0]["capitao"])

    def test_apply_reserve_substitutions_ignores_non_luxury_positions(self):
        team = pd.DataFrame(
            [
                {"atleta_id": 201, "apelido": "Lat 1", "posicao_id": 2, "reserva": False, "pontos": 3.0},
                {"atleta_id": 202, "apelido": "Lat 2", "posicao_id": 2, "reserva": False, "pontos": 4.0},
                {"atleta_id": 203, "apelido": "Lat 3", "posicao_id": 2, "reserva": True, "pontos": 12.0},
            ]
        )

        final_team = apply_reserve_substitutions(team)

        self.assertCountEqual(final_team["atleta_id"].tolist(), [201, 202])

    def test_apply_reserve_substitutions_keeps_normal_bench_behavior(self):
        team = pd.DataFrame(
            [
                {"atleta_id": 201, "apelido": "Lat 1", "posicao_id": 2, "reserva": False, "pontos": 3.0},
                {"atleta_id": 202, "apelido": "Lat 2", "posicao_id": 2, "reserva": False, "pontos": 4.0},
                {"atleta_id": 203, "apelido": "Lat 3", "posicao_id": 2, "reserva": True, "pontos": 1.0},
            ]
        )
        played = pd.DataFrame(
            [
                {"atleta_id": 201, "jogou": False},
                {"atleta_id": 202, "jogou": True},
                {"atleta_id": 203, "jogou": True},
            ]
        )

        final_team = apply_reserve_substitutions(team, played)

        self.assertCountEqual(final_team["atleta_id"].tolist(), [202, 203])
        reserve = final_team[final_team["atleta_id"] == 203].iloc[0]
        self.assertEqual(reserve["substituiu_atleta_id"], 201)

    def test_build_team_without_reserves_does_not_apply_substitutions(self):
        market = pd.DataFrame(
            [
                {"atleta_id": 401, "apelido": "Mei 1", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 10.0, "preco": 20.0, "jogou": False},
                {"atleta_id": 402, "apelido": "Mei 2", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 9.0, "preco": 19.0, "jogou": True},
                {"atleta_id": 403, "apelido": "Mei 3", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 8.0, "preco": 18.0, "jogou": True},
                {"atleta_id": 404, "apelido": "Mei 4", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 7.0, "preco": 1.0, "jogou": True},
            ]
        )
        models = {4: {"model": ScoreModel(), "feature_cols": ["score"], "mae": 0.0}}

        team = build_team(
            market,
            models,
            formation={4: 3},
            odds_filter=None,
            include_reserves=False,
        )

        self.assertCountEqual(team["atleta_id"].tolist(), [401, 402, 403])
        self.assertFalse(team["reserva"].any())
        self.assertNotIn("substituiu_atleta_id", team.columns)

    def test_score_with_captain_bonus_multiplies_only_captain(self):
        team = pd.DataFrame(
            [
                {"atleta_id": 401, "pontos": 10.0, "capitao": True},
                {"atleta_id": 501, "pontos": 8.0, "capitao": False},
            ]
        )

        self.assertEqual(score_with_captain_bonus(team), 10.0 * CAPTAIN_BONUS + 8.0)

    def test_lineup_output_table_matches_backtest_csv_columns(self):
        team = pd.DataFrame(
            [
                {
                    "rodada": 7,
                    "atleta_id": 401,
                    "apelido": "Mei 1",
                    "posicao_id": 4,
                    "clube_nome": "Clube A",
                    "adversario": "Clube B",
                    "mando": 1,
                    "pontos": pd.NA,
                    "pontos_real": 10.0,
                    "pontos_previstos": 8.0,
                    "reserva": False,
                    "capitao": True,
                }
            ]
        )

        output = lineup_output_table(team)

        self.assertEqual(
            output.columns.tolist(),
            [
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
            ],
        )
        self.assertEqual(output.loc[0, "mando"], "CASA")

    def test_merge_target_round_match_context_fills_opponent_name(self):
        market = pd.DataFrame(
            [
                {
                    "atleta_id": 401,
                    "temporada": 2026,
                    "rodada": 7,
                    "clube_id": 10,
                    "mando": -1,
                }
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": 7,
                    "clube_id": 10,
                    "mando": 1,
                    "clube_adversario_id": 20,
                }
            ]
        )

        output = merge_target_round_match_context(
            market,
            matches,
            season=2026,
            rodada_alvo=7,
            clubes_lookup={20: "Clube B"},
        )

        self.assertEqual(output.loc[0, "mando"], 1)
        self.assertEqual(output.loc[0, "clube_adversario_id"], 20)
        self.assertEqual(output.loc[0, "adversario"], "Clube B")

    def test_refresh_opponent_encoding_uses_target_round_opponent(self):
        market = pd.DataFrame(
            [
                {
                    "atleta_id": 401,
                    "clube_adversario_id": 20,
                    "clube_adv_enc": 99,
                }
            ]
        )
        features = pd.DataFrame(
            [
                {"clube_adversario_id": 10.0, "clube_adv_enc": 1},
                {"clube_adversario_id": 20.0, "clube_adv_enc": 2},
            ]
        )

        output = refresh_opponent_encoding_from_features(market, features)

        self.assertEqual(output.loc[0, "clube_adv_enc"], 2)

    def test_imprimir_time_prints_backtest_csv_columns(self):
        team = pd.DataFrame(
            [
                {
                    "atleta_id": 401,
                    "apelido": "Mei 1",
                    "posicao_id": 4,
                    "clube_nome": "Clube A",
                    "mando": -1,
                    "preco": 12.0,
                    "pontos_previstos": 8.0,
                    "reserva": False,
                    "capitao": True,
                },
                {
                    "atleta_id": 402,
                    "apelido": "Mei Banco",
                    "posicao_id": 4,
                    "clube_nome": "Clube A",
                    "mando": -1,
                    "preco": 4.0,
                    "pontos_previstos": 20.0,
                    "reserva": True,
                    "reserva_de_luxo": True,
                    "capitao": False,
                },
            ]
        )

        buffer = StringIO()
        with redirect_stdout(buffer):
            imprimir_time(team)
        printed = buffer.getvalue()

        self.assertIn("clube adversario", printed)
        self.assertIn("reserva_de_luxo", printed)
        self.assertIn("Mei Banco", printed)
        self.assertLess(printed.index("Mei 1"), printed.index("Mei Banco"))
        self.assertIn("-"*140, printed[printed.index("Mei 1"):printed.index("Mei Banco")])
        self.assertIn("Budget usado: 12.0 | Pts previstos com capitao: 12.00", printed)


if __name__ == "__main__":
    unittest.main()
