import unittest

import pandas as pd

from feature_engineering import build_features


class FeatureEngineeringTest(unittest.TestCase):
    def test_adds_home_away_point_averages_from_prior_rounds(self):
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
        self.assertTrue(pd.isna(round_1["avg_pts_casa_5r"]))
        self.assertTrue(pd.isna(round_1["avg_pts_fora_5r"]))

        round_4 = features.loc[features["rodada"] == 4].iloc[0]
        self.assertEqual(round_4["avg_pts_casa_5r"], 8.0)
        self.assertEqual(round_4["avg_pts_fora_5r"], 2.0)

        round_5 = features.loc[features["rodada"] == 5].iloc[0]
        self.assertEqual(round_5["avg_pts_casa_5r"], 8.0)
        self.assertEqual(round_5["avg_pts_fora_5r"], 3.0)

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
        self.assertEqual(round_3["gols_feitos_clube_5r"], 1.5)
        self.assertEqual(round_3["gols_sofridos_clube_5r"], 0.5)
        self.assertEqual(round_3["gols_feitos_adv_5r"], 0.5)
        self.assertEqual(round_3["gols_sofridos_adv_5r"], 1.5)


if __name__ == "__main__":
    unittest.main()
