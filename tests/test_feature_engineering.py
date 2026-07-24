import unittest

import pandas as pd

from feature_engineering import build_features


class FeatureEngineeringTest(unittest.TestCase):
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
