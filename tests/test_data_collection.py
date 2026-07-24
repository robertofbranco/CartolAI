import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd

import collect_all_data
from cartola_data import current


class MergePartitionedParquetsTest(unittest.TestCase):
    def test_merge_partitioned_parquets_adds_year_sorts_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pd.DataFrame(
                [
                    {"rodada": 2, "clube_id": 10, "prob_win": 0.4},
                    {"rodada": 2, "clube_id": 10, "prob_win": 0.6},
                ]
            ).to_parquet(data_dir / "odds_2025.parquet", index=False)
            pd.DataFrame(
                [
                    {"temporada": 2026, "rodada": 1, "clube_id": 20, "prob_win": 0.7},
                ]
            ).to_parquet(data_dir / "odds_2026.parquet", index=False)
            pd.DataFrame([{"temporada": 2026, "rodada": 1}]).to_parquet(
                data_dir / "partidas_2026.parquet",
                index=False,
            )

            merged = collect_all_data.merge_partitioned_parquets(
                data_dir=data_dir,
                dataset_names=["odds"],
            )

            self.assertEqual(merged, {"odds": data_dir / "odds.parquet"})
            self.assertFalse((data_dir / "partidas.parquet").exists())

            odds = pd.read_parquet(data_dir / "odds.parquet")
            self.assertEqual(odds["temporada"].tolist(), [2025, 2026])
            self.assertEqual(odds["rodada"].tolist(), [2, 1])
            self.assertEqual(odds["clube_id"].tolist(), [10, 20])
            self.assertEqual(odds["prob_win"].tolist(), [0.6, 0.7])


class CollectAllDataFlowTest(unittest.TestCase):
    def test_get_players_data_fetches_only_archive_missing_rounds_from_api(self):
        archive_players = pd.DataFrame(
            [
                {"temporada": 2026, "rodada": 1, "atleta_id": 101, "clube_id": 263},
                {"temporada": 2026, "rodada": 2, "atleta_id": 102, "clube_id": 263},
            ]
        )
        api_players = pd.DataFrame(
            [
                {"temporada": 2026, "rodada": 3, "atleta_id": 103, "clube_id": 263},
                {"temporada": 2026, "rodada": 4, "atleta_id": 104, "clube_id": 263},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            api = MagicMock()
            with (
                patch.object(collect_all_data, "DATA_DIR", data_dir),
                patch.object(
                    collect_all_data,
                    "get_players_data_from_caRtola",
                    return_value=archive_players,
                ) as get_archive_players,
                patch.object(
                    collect_all_data,
                    "get_players_data_from_cartola_api",
                    return_value=api_players,
                ) as get_api_players,
            ):
                collect_all_data.get_players_data(api, current_round=5, season=2026)

            get_archive_players.assert_called_once_with(2026, [1, 2, 3, 4])
            get_api_players.assert_called_once_with(api, [3, 4], temporada=2026)

            saved_players = pd.read_parquet(data_dir / "jogadores_por_rodada_2026.parquet")
            self.assertEqual(saved_players["rodada"].tolist(), [1, 2, 3, 4])

    def test_get_players_data_raises_when_no_player_data_is_collected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            api = MagicMock()
            with (
                patch.object(collect_all_data, "DATA_DIR", data_dir),
                patch.object(
                    collect_all_data,
                    "get_players_data_from_caRtola",
                    return_value=pd.DataFrame(),
                ),
                patch.object(
                    collect_all_data,
                    "get_players_data_from_cartola_api",
                    return_value=pd.DataFrame(),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Nenhum dado de jogadores coletado"):
                    collect_all_data.get_players_data(api, current_round=5, season=2026)

            self.assertFalse((data_dir / "jogadores_por_rodada_2026.parquet").exists())

    def test_run_collectors_wires_historic_current_and_current_players(self):
        args = Namespace(
            current_season=2026,
            historic_years=[2024, 2025],
            skip_current=False,
            skip_gato=True,
            skip_historic=False,
            skip_players=False,
        )
        players = pd.DataFrame(
            [
                {"temporada": 2026, "rodada": 1, "atleta_id": 100, "clube_id": 263},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            with (
                patch.object(collect_all_data, "DATA_DIR", data_dir),
                patch.object(collect_all_data.os.environ, "get", return_value="token-123"),
                patch.object(collect_all_data, "import_historic_season") as import_historic,
                patch.object(
                    collect_all_data,
                    "get_players_data_from_caRtola",
                    return_value=players,
                ) as get_players,
                patch.object(collect_all_data, "collect_current_season") as collect_current,
            ):
                collect_all_data.run_collectors(args)

            self.assertEqual(
                import_historic.call_args_list,
                [
                    call(
                        year=2024,
                        token="token-123",
                        collect_gato_data=False,
                        collect_players_data=True,
                    ),
                    call(
                        year=2025,
                        token="token-123",
                        collect_gato_data=False,
                        collect_players_data=True,
                    ),
                ],
            )
            get_players.assert_called_once_with(2026)
            collect_current.assert_called_once_with(temporada=2026, token="token-123")

            saved_players = pd.read_parquet(data_dir / "jogadores_por_rodada_2026.parquet")
            pd.testing.assert_frame_equal(saved_players, players)

    def test_main_merge_only_skips_collectors(self):
        args = Namespace(
            merge_only=True,
            skip_merge=False,
            merge_datasets=["odds"],
        )

        with (
            patch.object(collect_all_data, "parse_args", return_value=args),
            patch.object(collect_all_data, "run_collectors") as run_collectors,
            patch.object(
                collect_all_data,
                "merge_partitioned_parquets",
                return_value={"odds": Path("data/odds.parquet")},
            ) as merge_partitioned,
        ):
            collect_all_data.main()

        run_collectors.assert_not_called()
        merge_partitioned.assert_called_once_with(
            data_dir=collect_all_data.DATA_DIR,
            dataset_names=["odds"],
        )


class CurrentSeasonCollectionTest(unittest.TestCase):
    def test_collect_current_season_saves_odds_when_token_is_available(self):
        odds = pd.DataFrame(
            [
                {
                    "temporada": 2026,
                    "rodada": 2,
                    "clube_id": 263,
                    "prob_win": 0.5,
                    "prob_draw": 0.3,
                    "prob_loss": 0.2,
                }
            ]
        )
        api = MagicMock()
        api.market_status.return_value = {"rodada_atual": 3}
        api.clubes.return_value = {"263": {"abreviacao": "BOT"}}

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            with (
                patch.object(current, "DATA_DIR", data_dir),
                patch.object(current, "CartolaAPI", return_value=api) as cartola_api,
                patch.object(current, "GatoMestreAPI") as gato_api,
                patch.object(current, "get_odds", return_value=odds) as get_odds,
            ):
                result = current.collect_current_season(
                    temporada=2026,
                    rodadas=[1, 2],
                    token="token-123",
                )

            cartola_api.assert_called_once_with(token="token-123")
            gato_api.assert_called_once_with(token="token-123", temporada=2026)
            get_odds.assert_called_once_with(
                gato_api.return_value,
                {"263": {"abreviacao": "BOT"}},
                [1, 2, 3],
                temporada=2026,
            )
            self.assertEqual(result.files["odds"], data_dir / "odds_2026.parquet")
            saved_odds = pd.read_parquet(data_dir / "odds_2026.parquet")
            pd.testing.assert_frame_equal(saved_odds, odds)

    def test_build_abbr_to_clube_id_prefers_current_season_ids(self):
        clubes = {
            "263": {"abreviacao": "BOT"},
            "276": {"abreviacao": "SAO"},
            "277": {"abreviacao": "SAN"},
            "282": {"abreviacao": "CAM"},
            "306": {"abreviacao": "SAN"},
            "347": {"abreviacao": "BOT"},
            "349": {"abreviacao": "CAM"},
            "386": {"abreviacao": "SAO"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pd.DataFrame(
                [
                    {"clube_id": 263},
                    {"clube_id": 276},
                    {"clube_id": 277},
                    {"clube_id": 282},
                ]
            ).to_parquet(data_dir / "jogadores_por_rodada_2026.parquet", index=False)

            abbr_to_id = current._build_abbr_to_clube_id(
                clubes,
                temporada=2026,
                data_dir=data_dir,
            )

        self.assertEqual(abbr_to_id["BOT"], 263)
        self.assertEqual(abbr_to_id["SAO"], 276)
        self.assertEqual(abbr_to_id["SAN"], 277)
        self.assertEqual(abbr_to_id["CAM"], 282)


if __name__ == "__main__":
    unittest.main()
