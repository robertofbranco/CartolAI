import unittest

import pandas as pd

from cartola_data.config import STATUS
from cartola_team_builder import apply_reserve_substitutions, build_team


class ScoreModel:
    def predict(self, X):
        return X["score"].to_numpy()


class TeamBuilderReservesTest(unittest.TestCase):
    def test_build_team_selects_one_reserve_per_non_tec_position(self):
        market = pd.DataFrame(
            [
                {"atleta_id": 101, "apelido": "Gol 1", "posicao_id": 1, "status_id": STATUS["Provavel"], "score": 9.0},
                {"atleta_id": 102, "apelido": "Gol 2", "posicao_id": 1, "status_id": STATUS["Provavel"], "score": 8.0},
                {"atleta_id": 201, "apelido": "Lat 1", "posicao_id": 2, "status_id": STATUS["Provavel"], "score": 9.0},
                {"atleta_id": 202, "apelido": "Lat 2", "posicao_id": 2, "status_id": STATUS["Provavel"], "score": 8.0},
                {"atleta_id": 203, "apelido": "Lat 3", "posicao_id": 2, "status_id": STATUS["Provavel"], "score": 7.0},
                {"atleta_id": 204, "apelido": "Lat 4", "posicao_id": 2, "status_id": STATUS["Provavel"], "score": 6.0},
                {"atleta_id": 601, "apelido": "Tec 1", "posicao_id": 6, "status_id": STATUS["Provavel"], "score": 9.0},
                {"atleta_id": 602, "apelido": "Tec 2", "posicao_id": 6, "status_id": STATUS["Provavel"], "score": 8.0},
            ]
        )
        formation = {1: 1, 2: 2, 6: 1}
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
        self.assertEqual(reserves_by_position, {1: 1, 2: 1})
        self.assertEqual(team[team["posicao_id"] == 6]["reserva"].sum(), 0)

    def test_apply_reserve_substitutions_uses_only_one_reserve_per_position(self):
        team = pd.DataFrame(
            [
                {"atleta_id": 201, "apelido": "Lat 1", "posicao_id": 2, "reserva": False, "pontos_previstos": 9.0},
                {"atleta_id": 202, "apelido": "Lat 2", "posicao_id": 2, "reserva": False, "pontos_previstos": 8.0},
                {"atleta_id": 203, "apelido": "Lat 3", "posicao_id": 2, "reserva": True, "pontos_previstos": 7.0},
            ]
        )
        played = pd.DataFrame(
            [
                {"atleta_id": 201, "jogou": False},
                {"atleta_id": 202, "jogou": False},
                {"atleta_id": 203, "jogou": True},
            ]
        )

        final_team = apply_reserve_substitutions(team, played)

        self.assertCountEqual(final_team["atleta_id"].tolist(), [202, 203])
        substitute = final_team[final_team["atleta_id"] == 203].iloc[0]
        self.assertTrue(substitute["reserva"])
        self.assertEqual(substitute["substituiu_atleta_id"], 201)


if __name__ == "__main__":
    unittest.main()
