import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from cartola_data.config import CAPTAIN_BONUS
import cartola_backtest as backtest
from cartola_team_builder import build_target_round_market_features


class BacktestCaptainTest(unittest.TestCase):
    def test_calcular_teto_includes_captain_bonus(self):
        rodada = pd.DataFrame(
            [
                {"atleta_id": 401, "posicao_id": 4, "pontos": 10.0},
                {"atleta_id": 402, "posicao_id": 4, "pontos": 8.0},
                {"atleta_id": 403, "posicao_id": 4, "pontos": 4.0},
                {"atleta_id": 501, "posicao_id": 5, "pontos": 6.0},
            ]
        )

        teto = backtest.calcular_teto(rodada, {4: 2, 5: 1})

        self.assertEqual(teto, 10.0 * CAPTAIN_BONUS + 8.0 + 6.0)

    def test_load_chaves_ligas_aggregates_max_average_and_participant_percentiles(self):
        original_data_dir = backtest.DATA_DIR
        try:
            with TemporaryDirectory() as tmpdir:
                data_dir = Path(tmpdir)
                pd.DataFrame(
                    [
                        {
                            "liga": "liga-a",
                            "rodada": 6,
                            "chave_index": 0,
                            "pontos": 80.0,
                            "time_mandante_pontuacao": 80.0,
                            "time_visitante_pontuacao": 50.0,
                        },
                        {
                            "liga": "liga-a",
                            "rodada": 6,
                            "chave_index": 1,
                            "pontos": 60.0,
                            "time_mandante_pontuacao": 60.0,
                            "time_visitante_pontuacao": 40.0,
                        },
                        {
                            "liga": "liga-b",
                            "rodada": 6,
                            "chave_index": 0,
                            "pontos": 70.0,
                            "time_mandante_pontuacao": 30.0,
                            "time_visitante_pontuacao": 70.0,
                        },
                        {
                            "liga": "liga-a",
                            "rodada": 7,
                            "chave_index": 0,
                            "pontos": 90.0,
                            "time_mandante_pontuacao": 90.0,
                            "time_visitante_pontuacao": 20.0,
                        },
                    ]
                ).to_parquet(data_dir / "chaves_ligas.parquet", index=False)
                backtest.DATA_DIR = data_dir

                result = backtest.load_chaves_ligas()

            round_6 = result[result["rodada"] == 6].iloc[0]
            round_7 = result[result["rodada"] == 7].iloc[0]
            self.assertEqual(round_6["chaves_ligas_max_pontos"], 80.0)
            self.assertEqual(round_6["chaves_ligas_media_pontos"], 70.0)
            self.assertEqual(round_6["chaves_ligas_mediana_participantes"], 55.0)
            self.assertEqual(round_6["chaves_ligas_qtd_participantes"], 6)
            self.assertEqual(round_7["chaves_ligas_max_pontos"], 90.0)
            self.assertEqual(round_7["chaves_ligas_media_pontos"], 90.0)
        finally:
            backtest.DATA_DIR = original_data_dir

    def test_simulated_market_uses_previous_round_price_for_backtest_lock_time(self):
        players = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": 1,
                    "atleta_id": 100,
                    "apelido": "Mei 1",
                    "posicao_id": 4,
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
                    "apelido": "Mei 1",
                    "posicao_id": 4,
                    "clube_id": 1,
                    "pontos": 7.0,
                    "jogou": True,
                    "preco": 20.0,
                    "media": 6.0,
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
                for rodada in [1, 2]
            ]
        )
        target_market = players[players["rodada"] == 2].copy()

        market = build_target_round_market_features(
            df_players_per_round=players,
            df_matches=matches,
            df_odds=pd.DataFrame(columns=["temporada", "rodada", "clube_id"]),
            df_market=target_market,
            season=2026,
            rodada_alvo=2,
        )

        self.assertEqual(market.iloc[0]["preco"], 10.0)


if __name__ == "__main__":
    unittest.main()
