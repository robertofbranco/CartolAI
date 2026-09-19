import unittest

import pandas as pd

from feature_engineering import build_features


class FeatureEngineeringTest(unittest.TestCase):
    def test_adds_opponent_club_encoding_from_matches(self):
        players = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 5.0,
                    "jogou": True,
                    "preco": 10.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 6.0,
                    "jogou": True,
                    "preco": 10.0,
                },
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "clube_id": 1,
                    "mando": 1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 1,
                    "gols_sofridos_clube": 0,
                },
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "clube_id": 1,
                    "mando": -1,
                    "clube_adversario_id": 3,
                    "gols_feitos_clube": 0,
                    "gols_sofridos_clube": 1,
                },
            ]
        )

        features = build_features(players, matches, pd.DataFrame())

        round_1 = features.loc[features["rodada"] == 1].iloc[0]
        round_2 = features.loc[features["rodada"] == 2].iloc[0]

        self.assertIn("clube_adv_enc", features.columns)
        self.assertEqual(round_1["clube_adversario_id"], 2)
        self.assertEqual(round_2["clube_adversario_id"], 3)
        self.assertEqual(round_1["clube_enc"], round_2["clube_enc"])
        self.assertNotEqual(round_1["clube_adv_enc"], round_2["clube_adv_enc"])

    def test_shifts_post_round_market_values_to_previous_round(self):
        players = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 5.0,
                    "jogou": True,
                    "preco": 10.0,
                    "media": 5.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 7.0,
                    "jogou": True,
                    "preco": 12.0,
                    "media": 6.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 3,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 3.0,
                    "jogou": True,
                    "preco": 8.0,
                    "media": 5.0,
                },
            ]
        )
        matches = pd.DataFrame(
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

        features = build_features(players, matches, pd.DataFrame())

        round_1 = features.loc[features["rodada"] == 1].iloc[0]
        round_2 = features.loc[features["rodada"] == 2].iloc[0]
        round_3 = features.loc[features["rodada"] == 3].iloc[0]

        self.assertTrue(pd.isna(round_1["preco"]))
        self.assertTrue(pd.isna(round_1["media"]))
        self.assertTrue(pd.isna(round_1["preco_lag1"]))
        self.assertTrue(pd.isna(round_1["media_lag"]))
        self.assertEqual(round_2["preco"], 10.0)
        self.assertEqual(round_2["media"], 5.0)
        self.assertEqual(round_2["preco_lag1"], 10.0)
        self.assertEqual(round_2["media_lag"], 5.0)
        self.assertEqual(round_3["preco"], 12.0)
        self.assertEqual(round_3["media"], 6.0)
        self.assertEqual(round_3["preco_lag1"], 12.0)
        self.assertEqual(round_3["media_lag"], 6.0)

    def test_separates_appearance_form_from_round_availability(self):
        played_by_round = [False, True, False, True, True, pd.NA]
        points_by_round = [0.0, 4.0, 0.0, 8.0, 6.0, pd.NA]
        players = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": rodada,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": points_by_round[rodada - 1],
                    "jogou": played_by_round[rodada - 1],
                    "preco": 10.0,
                    "media": 5.0,
                }
                for rodada in range(1, 7)
            ]
        )
        matches = pd.DataFrame(
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
                for rodada in range(1, 7)
            ]
        )

        features = build_features(players, matches, pd.DataFrame())

        round_1 = features.loc[features["rodada"] == 1].iloc[0]
        self.assertEqual(round_1["ewma_pts_aparicoes"], 0.0)
        self.assertEqual(round_1["aparicoes_5r"], 0.0)
        self.assertEqual(round_1["rodadas_desde_ultima_aparicao"], 10.0)
        self.assertEqual(round_1["sem_historico"], 1.0)

        round_4 = features.loc[features["rodada"] == 4].iloc[0]
        self.assertEqual(round_4["media_pts_ultimas_3_aparicoes"], 4.0)
        self.assertEqual(round_4["pts_ultima_aparicao"], 4.0)
        self.assertEqual(round_4["mediana_pts_5_aparicoes"], 4.0)
        self.assertEqual(round_4["aparicoes_5r"], 1.0)
        self.assertEqual(round_4["rodadas_desde_ultima_aparicao"], 2.0)
        self.assertEqual(round_4["sequencia_aparicoes"], 0.0)

        target_round = features.loc[features["rodada"] == 6].iloc[0]
        self.assertEqual(target_round["media_pts_ultimas_3_aparicoes"], 6.0)
        self.assertEqual(target_round["media_pts_ultimas_5_aparicoes"], 6.0)
        self.assertEqual(target_round["media_pts_ultimas_10_aparicoes"], 6.0)
        self.assertEqual(target_round["pts_ultima_aparicao"], 6.0)
        self.assertEqual(target_round["mediana_pts_5_aparicoes"], 6.0)
        self.assertEqual(target_round["std_pts_ultimas_5_aparicoes"], 2.0)
        self.assertEqual(target_round["aparicoes_5r"], 3.0)
        self.assertEqual(target_round["aparicoes_10r"], 3.0)
        self.assertEqual(target_round["regularidade_5r"], 0.6)
        self.assertEqual(target_round["regularidade_10r"], 0.6)
        self.assertEqual(target_round["rodadas_desde_ultima_aparicao"], 1.0)
        self.assertEqual(target_round["sequencia_aparicoes"], 2.0)
        self.assertEqual(target_round["aparicoes_anteriores"], 3.0)
        self.assertEqual(target_round["sem_historico"], 0.0)

    def test_recent_form_is_responsive_robust_and_excludes_target_outcome(self):
        points = [5.0, 6.0, 5.0, 25.0, 0.0, 100.0]
        played = [True, True, True, True, False, True]
        players = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": rodada,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": points[rodada - 1],
                    "jogou": played[rodada - 1],
                    "preco": 10.0,
                    "media": 5.0,
                }
                for rodada in range(1, 7)
            ]
        )
        matches = pd.DataFrame(
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
                for rodada in range(1, 7)
            ]
        )

        features = build_features(players, matches, pd.DataFrame())
        target_round = features.loc[features["rodada"] == 6].iloc[0]
        expected_ewma = pd.Series(points[:4]).ewm(halflife=3, adjust=True).mean().iloc[-1]

        self.assertEqual(target_round["pts_ultima_aparicao"], 25.0)
        self.assertEqual(target_round["mediana_pts_5_aparicoes"], 5.5)
        self.assertEqual(target_round["media_pts_ultimas_10_aparicoes"], 10.25)
        self.assertAlmostEqual(target_round["ewma_pts_aparicoes"], expected_ewma)
        self.assertAlmostEqual(
            target_round["tendencia_pts_aparicoes"],
            expected_ewma - 10.25,
        )
        self.assertLess(target_round["ewma_pts_aparicoes"], 25.0)
        self.assertEqual(target_round["aparicoes_anteriores"], 4.0)

    def test_adds_same_mando_point_averages_from_prior_rounds(self):
        players = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 10.0,
                    "jogou": True,
                    "preco": 10.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 2.0,
                    "jogou": True,
                    "preco": 10.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 3,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 6.0,
                    "jogou": True,
                    "preco": 10.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 4,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 4.0,
                    "jogou": True,
                    "preco": 10.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 5,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": pd.NA,
                    "jogou": pd.NA,
                    "preco": 10.0,
                },
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "clube_id": 1,
                    "mando": 1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 1,
                    "gols_sofridos_clube": 0,
                },
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "clube_id": 1,
                    "mando": -1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 0,
                    "gols_sofridos_clube": 1,
                },
                {
                    "temporada": 2026,
                    "rodada": 3,
                    "clube_id": 1,
                    "mando": 1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 2,
                    "gols_sofridos_clube": 1,
                },
                {
                    "temporada": 2026,
                    "rodada": 4,
                    "clube_id": 1,
                    "mando": -1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 1,
                    "gols_sofridos_clube": 2,
                },
                {
                    "temporada": 2026,
                    "rodada": 5,
                    "clube_id": 1,
                    "mando": 1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": pd.NA,
                    "gols_sofridos_clube": pd.NA,
                },
            ]
        )

        features = build_features(players, matches, pd.DataFrame())

        round_1 = features.loc[features["rodada"] == 1].iloc[0]
        self.assertTrue(pd.isna(round_1["avg_pts_mando_5r"]))

        round_3 = features.loc[features["rodada"] == 3].iloc[0]
        self.assertEqual(round_3["avg_pts_mando_5r"], 10.0)

        round_4 = features.loc[features["rodada"] == 4].iloc[0]
        self.assertEqual(round_4["avg_pts_mando_5r"], 2.0)

        round_5 = features.loc[features["rodada"] == 5].iloc[0]
        self.assertEqual(round_5["avg_pts_mando_5r"], 8.0)

    def test_adds_player_points_trend_features(self):
        players = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": rodada,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": pontos,
                    "jogou": True,
                    "preco": 10.0,
                }
                for rodada, pontos in [
                    (1, 1.0),
                    (2, 1.0),
                    (3, 1.0),
                    (4, 1.0),
                    (5, 1.0),
                    (6, 1.0),
                    (7, 1.0),
                    (8, 10.0),
                    (9, 10.0),
                    (10, 10.0),
                    (11, 0.0),
                ]
            ]
        )
        matches = pd.DataFrame(
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
                for rodada in range(1, 12)
            ]
        )

        features = build_features(players, matches, pd.DataFrame())

        round_11 = features.loc[features["rodada"] == 11].iloc[0]
        self.assertEqual(round_11["media_pts_3r"], 10.0)
        self.assertAlmostEqual(round_11["media_pts_5r"], 6.4)
        self.assertAlmostEqual(round_11["media_pts_10r"], 3.7)
        self.assertAlmostEqual(round_11["media_pts_delta_3_5r"], 3.6)
        self.assertAlmostEqual(round_11["media_pts_delta_3_10r"], 6.3)
        self.assertAlmostEqual(round_11["media_pts_delta_5_10r"], 2.7)

    def test_adds_opponent_rolling_goal_features_from_previous_rounds(self):
        players = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": rodada,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 5.0,
                    "jogou": True,
                    "preco": 10.0,
                }
                for rodada in [1, 2, 3]
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "clube_id": 1,
                    "mando": 1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 1,
                    "gols_sofridos_clube": 0,
                },
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "clube_id": 2,
                    "mando": -1,
                    "clube_adversario_id": 1,
                    "gols_feitos_clube": 0,
                    "gols_sofridos_clube": 1,
                },
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "clube_id": 1,
                    "mando": -1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 2,
                    "gols_sofridos_clube": 1,
                },
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "clube_id": 2,
                    "mando": 1,
                    "clube_adversario_id": 1,
                    "gols_feitos_clube": 1,
                    "gols_sofridos_clube": 2,
                },
                {
                    "temporada": 2026,
                    "rodada": 3,
                    "clube_id": 1,
                    "mando": 1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 4,
                    "gols_sofridos_clube": 3,
                },
                {
                    "temporada": 2026,
                    "rodada": 3,
                    "clube_id": 2,
                    "mando": -1,
                    "clube_adversario_id": 1,
                    "gols_feitos_clube": 3,
                    "gols_sofridos_clube": 4,
                },
            ]
        )
        odds = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": rodada,
                    "clube_id": 1,
                    "prob_win": 1 / 3,
                    "prob_draw": 1 / 3,
                    "prob_loss": 1 / 3,
                }
                for rodada in [1, 2, 3]
            ]
        )

        features = build_features(players, matches, odds)

        round_3 = features.loc[features["rodada"] == 3].iloc[0]
        self.assertEqual(round_3["gols_feitos_clube_3r"], 1.5)
        self.assertEqual(round_3["gols_sofridos_clube_3r"], 0.5)
        self.assertEqual(round_3["gols_feitos_adv_3r"], 0.5)
        self.assertEqual(round_3["gols_sofridos_adv_3r"], 1.5)
        self.assertEqual(round_3["gols_feitos_clube_5r"], 1.5)
        self.assertEqual(round_3["gols_sofridos_clube_5r"], 0.5)
        self.assertEqual(round_3["gols_feitos_adv_5r"], 0.5)
        self.assertEqual(round_3["gols_sofridos_adv_5r"], 1.5)

    def test_adds_mando_aware_rolling_goal_features(self):
        players = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": rodada,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 5.0,
                    "jogou": True,
                    "preco": 10.0,
                }
                for rodada in [1, 2, 3, 4, 5]
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "clube_id": 1,
                    "mando": 1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 1,
                    "gols_sofridos_clube": 0,
                },
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "clube_id": 2,
                    "mando": -1,
                    "clube_adversario_id": 1,
                    "gols_feitos_clube": 0,
                    "gols_sofridos_clube": 1,
                },
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "clube_id": 1,
                    "mando": -1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 2,
                    "gols_sofridos_clube": 1,
                },
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "clube_id": 2,
                    "mando": 1,
                    "clube_adversario_id": 1,
                    "gols_feitos_clube": 1,
                    "gols_sofridos_clube": 2,
                },
                {
                    "temporada": 2026,
                    "rodada": 3,
                    "clube_id": 1,
                    "mando": 1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 5,
                    "gols_sofridos_clube": 2,
                },
                {
                    "temporada": 2026,
                    "rodada": 3,
                    "clube_id": 2,
                    "mando": -1,
                    "clube_adversario_id": 1,
                    "gols_feitos_clube": 2,
                    "gols_sofridos_clube": 5,
                },
                {
                    "temporada": 2026,
                    "rodada": 4,
                    "clube_id": 1,
                    "mando": -1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 4,
                    "gols_sofridos_clube": 3,
                },
                {
                    "temporada": 2026,
                    "rodada": 4,
                    "clube_id": 2,
                    "mando": 1,
                    "clube_adversario_id": 1,
                    "gols_feitos_clube": 3,
                    "gols_sofridos_clube": 4,
                },
                {
                    "temporada": 2026,
                    "rodada": 5,
                    "clube_id": 1,
                    "mando": 1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 9,
                    "gols_sofridos_clube": 8,
                },
                {
                    "temporada": 2026,
                    "rodada": 5,
                    "clube_id": 2,
                    "mando": -1,
                    "clube_adversario_id": 1,
                    "gols_feitos_clube": 8,
                    "gols_sofridos_clube": 9,
                },
            ]
        )

        features = build_features(players, matches, pd.DataFrame())

        round_5 = features.loc[features["rodada"] == 5].iloc[0]
        self.assertAlmostEqual(round_5["gols_feitos_clube_3r"], 11 / 3)
        self.assertEqual(round_5["gols_feitos_clube_mando_3r"], 3.0)
        self.assertEqual(round_5["gols_sofridos_clube_mando_3r"], 1.0)
        self.assertEqual(round_5["gols_feitos_adv_mando_3r"], 1.0)
        self.assertEqual(round_5["gols_sofridos_adv_mando_3r"], 3.0)

    def test_adds_rolling_club_points_features_from_previous_rounds(self):
        players = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 5.0,
                    "jogou": True,
                    "preco": 10.0,
                    "media": 5.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "atleta_id": 101,
                    "clube_id": 1,
                    "pontos": 7.0,
                    "jogou": True,
                    "preco": 10.0,
                    "media": 7.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "atleta_id": 200,
                    "clube_id": 2,
                    "pontos": 4.0,
                    "jogou": True,
                    "preco": 10.0,
                    "media": 4.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 5.0,
                    "jogou": True,
                    "preco": 10.0,
                    "media": 5.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "atleta_id": 101,
                    "clube_id": 1,
                    "pontos": 3.0,
                    "jogou": True,
                    "preco": 10.0,
                    "media": 5.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "atleta_id": 200,
                    "clube_id": 2,
                    "pontos": 10.0,
                    "jogou": True,
                    "preco": 10.0,
                    "media": 7.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 3,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 5.0,
                    "jogou": True,
                    "preco": 10.0,
                    "media": 5.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 3,
                    "atleta_id": 101,
                    "clube_id": 1,
                    "pontos": 1.0,
                    "jogou": True,
                    "preco": 10.0,
                    "media": 4.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 3,
                    "atleta_id": 200,
                    "clube_id": 2,
                    "pontos": 14.0,
                    "jogou": True,
                    "preco": 10.0,
                    "media": 8.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 4,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 5.0,
                    "jogou": True,
                    "preco": 10.0,
                    "media": 5.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 4,
                    "atleta_id": 101,
                    "clube_id": 1,
                    "pontos": 2.0,
                    "jogou": True,
                    "preco": 10.0,
                    "media": 4.0,
                },
                {
                    "temporada": 2026,
                    "rodada": 4,
                    "atleta_id": 200,
                    "clube_id": 2,
                    "pontos": 9.0,
                    "jogou": True,
                    "preco": 10.0,
                    "media": 8.0,
                },
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "clube_id": 1,
                    "mando": 1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 2,
                    "gols_sofridos_clube": 0,
                },
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "clube_id": 2,
                    "mando": -1,
                    "clube_adversario_id": 1,
                    "gols_feitos_clube": 0,
                    "gols_sofridos_clube": 2,
                },
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "clube_id": 1,
                    "mando": -1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 1,
                    "gols_sofridos_clube": 1,
                },
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "clube_id": 2,
                    "mando": 1,
                    "clube_adversario_id": 1,
                    "gols_feitos_clube": 1,
                    "gols_sofridos_clube": 1,
                },
                {
                    "temporada": 2026,
                    "rodada": 3,
                    "clube_id": 1,
                    "mando": 1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 0,
                    "gols_sofridos_clube": 2,
                },
                {
                    "temporada": 2026,
                    "rodada": 3,
                    "clube_id": 2,
                    "mando": -1,
                    "clube_adversario_id": 1,
                    "gols_feitos_clube": 2,
                    "gols_sofridos_clube": 0,
                },
                {
                    "temporada": 2026,
                    "rodada": 4,
                    "clube_id": 1,
                    "mando": -1,
                    "clube_adversario_id": 2,
                    "gols_feitos_clube": 1,
                    "gols_sofridos_clube": 0,
                },
                {
                    "temporada": 2026,
                    "rodada": 4,
                    "clube_id": 2,
                    "mando": 1,
                    "clube_adversario_id": 1,
                    "gols_feitos_clube": 0,
                    "gols_sofridos_clube": 1,
                },
            ]
        )

        features = build_features(players, matches, pd.DataFrame())

        round_1 = features.loc[features["rodada"] == 1].iloc[0]
        self.assertTrue(pd.isna(round_1["pontos_conquistados_clube_3r"]))
        self.assertTrue(pd.isna(round_1["pontos_cedidos_adv_3r"]))

        round_4 = features.loc[
            (features["rodada"] == 4) & (features["atleta_id"] == 100)
        ].iloc[0]
        self.assertAlmostEqual(round_4["pontos_conquistados_clube_3r"], 26 / 3)
        self.assertEqual(round_4["pontos_conquistados_clube_mando_3r"], 8.0)
        self.assertAlmostEqual(round_4["pontos_cedidos_adv_3r"], 26 / 3)
        self.assertEqual(round_4["pontos_cedidos_adv_mando_3r"], 8.0)

    def test_adds_foul_and_shot_scout_rolling_features_from_previous_rounds(self):
        players = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 1.0,
                    "jogou": True,
                    "preco": 10.0,
                    "scout_FD": 2,
                    "scout_FF": 1,
                    "scout_FT": 0,
                },
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 2.0,
                    "jogou": True,
                    "preco": 10.0,
                    "scout_FD": 5,
                    "scout_FF": 1,
                    "scout_FT": 1,
                },
                {
                    "temporada": 2026,
                    "rodada": 3,
                    "atleta_id": 100,
                    "clube_id": 1,
                    "pontos": 3.0,
                    "jogou": True,
                    "preco": 10.0,
                    "scout_FD": 9,
                    "scout_FF": 3,
                    "scout_FT": 3,
                },
            ]
        )
        matches = pd.DataFrame(
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

        features = build_features(players, matches, pd.DataFrame())

        round_1 = features.loc[features["rodada"] == 1].iloc[0]
        self.assertTrue(pd.isna(round_1["scout_FD_5r"]))
        self.assertTrue(pd.isna(round_1["scout_FF_5r"]))
        self.assertTrue(pd.isna(round_1["scout_FT_5r"]))

        round_3 = features.loc[features["rodada"] == 3].iloc[0]
        self.assertEqual(round_3["scout_FD_5r"], 2.5)
        self.assertEqual(round_3["scout_FF_5r"], 0.5)
        self.assertEqual(round_3["scout_FT_5r"], 0.5)

    def test_adds_opponent_fd_volume_from_raw_scout_counts(self):
        players = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": rodada,
                    "atleta_id": atleta_id,
                    "clube_id": clube_id,
                    "pontos": 0.0,
                    "jogou": True,
                    "preco": 10.0,
                    "scout_FD": fd,
                }
                for rodada, fd in enumerate([2, 6, 12, 20, 30], start=1)
                for atleta_id, clube_id, fd in [(100, 1, 0), (200, 2, fd)]
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": rodada,
                    "clube_id": clube_id,
                    "mando": 1 if clube_id == 1 else 0,
                    "clube_adversario_id": 2 if clube_id == 1 else 1,
                    "gols_feitos_clube": 0,
                    "gols_sofridos_clube": 0,
                }
                for rodada in range(1, 6)
                for clube_id in [1, 2]
            ]
        )

        features = build_features(players, matches, pd.DataFrame())
        round_1 = features.loc[
            (features["rodada"] == 1) & (features["clube_id"] == 1)
        ].iloc[0]
        round_5 = features.loc[
            (features["rodada"] == 5) & (features["clube_id"] == 1)
        ].iloc[0]

        self.assertTrue(pd.isna(round_1["FD_avd_3r"]))
        # Opponent FD per round is 2, 4, 6, 8, 10.  Round 5 uses only 2, 4, 6, 8.
        self.assertEqual(round_5["FD_avd_3r"], 6.0)
        self.assertEqual(round_5["FD_adv_5r"], 5.0)

if __name__ == "__main__":
    unittest.main()
