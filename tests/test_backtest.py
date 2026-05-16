import unittest

import pandas as pd

from cartola_data.config import CAPTAIN_BONUS
from cartola_backtest import calcular_teto


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

        teto = calcular_teto(rodada, {4: 2, 5: 1})

        self.assertEqual(teto, 10.0 * CAPTAIN_BONUS + 8.0 + 6.0)


if __name__ == "__main__":
    unittest.main()
