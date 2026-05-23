import unittest
from contextlib import redirect_stdout
from io import StringIO

import pandas as pd

from cartola_data.config import CAPTAIN_BONUS
from cartola_data.config import STATUS
from cartola_team_builder import (
    assign_captain,
    apply_reserve_substitutions,
    build_team,
    imprimir_time,
    lineup_output_table,
    merge_target_round_match_context,
    score_with_captain_bonus,
)


class ScoreModel:
    def predict(self, X):
        return X["score"].to_numpy()


class TeamBuilderReservesTest(unittest.TestCase):
    def test_build_team_selects_one_luxury_reserve_from_mei_or_ata_by_expected_points(self):
        market = pd.DataFrame(
            [
                {"atleta_id": 101, "apelido": "Gol 1", "posicao_id": 1, "status_id": STATUS["Provavel"], "score": 9.0, "preco": 12.0},
                {"atleta_id": 102, "apelido": "Gol 2", "posicao_id": 1, "status_id": STATUS["Provavel"], "score": 8.0, "preco": 1.0},
                {"atleta_id": 201, "apelido": "Lat 1", "posicao_id": 2, "status_id": STATUS["Provavel"], "score": 9.0, "preco": 12.0},
                {"atleta_id": 202, "apelido": "Lat 2", "posicao_id": 2, "status_id": STATUS["Provavel"], "score": 8.0, "preco": 11.0},
                {"atleta_id": 203, "apelido": "Lat 3", "posicao_id": 2, "status_id": STATUS["Provavel"], "score": 7.0, "preco": 1.0},
                {"atleta_id": 204, "apelido": "Lat 4", "posicao_id": 2, "status_id": STATUS["Provavel"], "score": 6.0, "preco": 1.0},
                {"atleta_id": 401, "apelido": "Mei 1", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 10.0, "preco": 20.0},
                {"atleta_id": 402, "apelido": "Mei 2", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 9.0, "preco": 19.0},
                {"atleta_id": 403, "apelido": "Mei 3", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 8.0, "preco": 18.0},
                {"atleta_id": 404, "apelido": "Mei 4", "posicao_id": 4, "status_id": STATUS["Provavel"], "score": 7.0, "preco": 1.0},
                {"atleta_id": 501, "apelido": "Ata 1", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 10.0, "preco": 20.0},
                {"atleta_id": 502, "apelido": "Ata 2", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 9.0, "preco": 19.0},
                {"atleta_id": 503, "apelido": "Ata 3", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 8.0, "preco": 1.0},
                {"atleta_id": 504, "apelido": "Ata 4", "posicao_id": 5, "status_id": STATUS["Provavel"], "score": 7.0, "preco": 18.0},
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
        self.assertEqual(reserves_by_position, {5: 1})
        self.assertEqual(team[team["posicao_id"] == 1]["reserva"].sum(), 0)
        self.assertEqual(team[team["posicao_id"] == 2]["reserva"].sum(), 0)
        self.assertEqual(team[team["posicao_id"] == 4]["reserva"].sum(), 0)
        self.assertEqual(team[team["posicao_id"] == 6]["reserva"].sum(), 0)
        self.assertEqual(team.loc[team["reserva"], "atleta_id"].iloc[0], 503)
        self.assertCountEqual(
            team[(team["posicao_id"] == 5) & ~team["reserva"]]["atleta_id"].tolist(),
            [501, 502, 504],
        )
        self.assertLess(
            team.loc[team["reserva"], "preco"].iloc[0],
            team[(team["posicao_id"] == 5) & ~team["reserva"]]["preco"].min(),
        )
        self.assertEqual(team["capitao"].sum(), 1)
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

    def test_apply_reserve_substitutions_uses_luxury_reserve_when_he_scores_more(self):
        team = pd.DataFrame(
            [
                {"atleta_id": 401, "apelido": "Mei 1", "posicao_id": 4, "reserva": False, "pontos_previstos": 9.0},
                {"atleta_id": 402, "apelido": "Mei 2", "posicao_id": 4, "reserva": False, "pontos_previstos": 8.0},
                {"atleta_id": 403, "apelido": "Mei 3", "posicao_id": 4, "reserva": True, "pontos_previstos": 7.0},
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
                {"atleta_id": 402, "apelido": "Mei 2", "posicao_id": 4, "reserva": True, "capitao": False, "pontos_previstos": 7.0},
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
                "pontos",
                "pontos_com_bonus",
                "pontos_previstos",
                "reserva",
                "capitao",
                "substituiu_atleta_id",
                "substituiu_apelido",
            ],
        )
        self.assertEqual(output.loc[0, "mando"], "CASA")
        self.assertEqual(output.loc[0, "pontos"], 10.0)
        self.assertEqual(output.loc[0, "pontos_com_bonus"], 10.0 * CAPTAIN_BONUS)

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
                }
            ]
        )

        buffer = StringIO()
        with redirect_stdout(buffer):
            imprimir_time(team)
        printed = buffer.getvalue()

        self.assertIn("clube adversario", printed)
        self.assertIn("pontos_com_bonus", printed)
        self.assertIn("substituiu_apelido", printed)


if __name__ == "__main__":
    unittest.main()
