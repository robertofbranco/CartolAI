import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd

import collect_latest_data
from cartola_data import current
from cartola_data.file_manager import (
    CartolaUsersMeanDataset,
    CurrentMarketDataset,
    FileManager,
    LeagueBracketsDataset,
    MatchesDataset,
    OddsDataset,
    PlayersDataset,
)


class FileManagerDatasetFilesTest(unittest.TestCase):
    def test_dataset_classes_expose_paths_and_read_write_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            df = pd.DataFrame([{"id": 1, "value": 2.5}])

            cases = [
                (PlayersDataset(season=2026, data_dir=data_dir), "jogadores_por_rodada_2026.parquet"),
                (MatchesDataset(season=2026, data_dir=data_dir), "partidas_2026.parquet"),
                (OddsDataset(season=2026, data_dir=data_dir), "odds_2026.parquet"),
                (CurrentMarketDataset(data_dir=data_dir), "mercado_atual.parquet"),
                (CartolaUsersMeanDataset(data_dir=data_dir), "medias_cartoleiros.parquet"),
                (LeagueBracketsDataset(data_dir=data_dir), "chaves_ligas.parquet"),
            ]

            for dataset, filename in cases:
                self.assertEqual(dataset.path, data_dir / filename)
                self.assertFalse(dataset.exists())
                saved_path = dataset.write(df)
                self.assertEqual(saved_path, data_dir / filename)
                pd.testing.assert_frame_equal(dataset.read(), df)

            self.assertIsInstance(FileManager(data_dir).dataset("odds", 2026), OddsDataset)


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

            merged = collect_latest_data.merge_partitioned_parquets(
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
    def test_get_players_data_fetches_only_latest_archive_missing_round_from_api(self):
        archive_players = pd.DataFrame(
            [
                {"temporada": 2026, "rodada": 1, "atleta_id": 101, "clube_id": 263},
                {"temporada": 2026, "rodada": 2, "atleta_id": 102, "clube_id": 263},
            ]
        )
        api_players = pd.DataFrame(
            [
                {"temporada": 2026, "rodada": 4, "atleta_id": 104, "clube_id": 263},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            api = MagicMock()
            with (
                patch.object(collect_latest_data, "DATA_DIR", data_dir),
                patch.object(
                    collect_latest_data,
                    "get_players_data_from_caRtola",
                    return_value=archive_players,
                ) as get_archive_players,
                patch.object(
                    collect_latest_data,
                    "get_round_players_data_from_cartola",
                    return_value=api_players,
                ) as get_api_players,
            ):
                collect_latest_data.get_previous_round_players_data(api, current_round=5, season=2026)

            get_archive_players.assert_called_once_with(2026, [1, 2, 3, 4])
            get_api_players.assert_called_once_with(api, 4, temporada=2026)

            saved_players = pd.read_parquet(data_dir / "jogadores_por_rodada_2026.parquet")
            self.assertEqual(saved_players["rodada"].tolist(), [1, 2, 4])

    def test_get_players_data_raises_when_no_player_data_is_collected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            api = MagicMock()
            with (
                patch.object(collect_latest_data, "DATA_DIR", data_dir),
                patch.object(
                    collect_latest_data,
                    "get_players_data_from_caRtola",
                    return_value=pd.DataFrame(),
                ),
                patch.object(
                    collect_latest_data,
                    "get_round_players_data_from_cartola",
                    return_value=pd.DataFrame(),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Nenhum dado de jogadores coletado"):
                    collect_latest_data.get_previous_round_players_data(api, current_round=5, season=2026)

            self.assertFalse((data_dir / "jogadores_por_rodada_2026.parquet").exists())

    def test_latest_round_players_data_uses_current_market_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "atleta_id": 101,
                        "preco": 12.5,
                        "media": 7.25,
                        "status_id": 7,
                        "jogos": 3,
                    },
                ]
            ).to_parquet(data_dir / "mercado_atual.parquet", index=False)

            api = MagicMock()
            api.clubes.return_value = {"263": {"nome": "Botafogo"}}
            api.atletas_pontuados.return_value = {
                "atletas": {
                    "101": {
                        "apelido": "Player 101",
                        "pontuacao": 8.4,
                        "posicao_id": 4,
                        "clube_id": 263,
                        "entrou_em_campo": True,
                        "scout": {"G": 1},
                    }
                }
            }

            with patch.object(current, "DATA_DIR", data_dir):
                players = current.get_round_players_data_from_cartola(
                    api,
                    4,
                    temporada=2026,
                )

            api.atletas_pontuados.assert_called_once_with(4)
            self.assertEqual(len(players), 1)
            row = players.iloc[0]
            self.assertEqual(row["status_id"], 7)
            self.assertEqual(row["preco"], 12.5)
            self.assertEqual(row["media"], 7.25)
            self.assertEqual(row["jogos"], 3)

    def test_run_collectors_wires_historic_current_and_current_players(self):
        args = Namespace(
            current_season=2026,
            historic_years=[2024, 2025],
            skip_current=False,
            skip_gato=True,
            skip_historic=False,
            skip_players=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            api = MagicMock()
            with (
                patch.object(collect_latest_data, "DATA_DIR", data_dir),
                patch.object(collect_latest_data.os.environ, "get", return_value="token-123"),
                patch.object(collect_latest_data, "import_historic_season") as import_historic,
                patch.object(collect_latest_data, "CartolaAPI", return_value=api) as cartola_api,
                patch.object(collect_latest_data, "get_current_round", return_value=5) as get_round,
                patch.object(collect_latest_data, "collect_current_season") as collect_current,
                patch.object(collect_latest_data, "get_players_data") as get_players,
            ):
                collect_latest_data.run_collectors(args)

            import_historic.assert_not_called()
            cartola_api.assert_called_once_with(token="token-123")
            get_round.assert_called_once_with(api)
            collect_current.assert_called_once_with(
                api=api,
                current_round=5,
                temporada=2026,
                token="token-123",
            )
            get_players.assert_called_once_with(api, 5, 2026)

    def test_main_merge_only_skips_collectors(self):
        args = Namespace(
            merge_only=True,
            skip_merge=False,
            merge_datasets=["odds"],
        )

        with (
            patch.object(collect_latest_data, "parse_args", return_value=args),
            patch.object(collect_latest_data, "run_collectors") as run_collectors,
            patch.object(
                collect_latest_data,
                "merge_partitioned_parquets",
                return_value={"odds": Path("data/odds.parquet")},
            ) as merge_partitioned,
        ):
            collect_latest_data.main()

        run_collectors.assert_not_called()
        merge_partitioned.assert_called_once_with(
            data_dir=collect_latest_data.DATA_DIR,
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
        api.clubes.return_value = {"263": {"abreviacao": "BOT"}}

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            with (
                patch.object(current, "DATA_DIR", data_dir),
                patch.object(current, "GatoMestreAPI") as gato_api,
                patch.object(current, "preparar_partidas", return_value=pd.DataFrame()),
                patch.object(current, "get_current_market", return_value=pd.DataFrame()),
                patch.object(current, "get_odds", return_value=odds) as get_odds,
                patch.object(current, "get_cartola_users_mean", return_value=pd.DataFrame()),
                patch.object(current, "get_league_brackets", return_value=pd.DataFrame()),
            ):
                result = current.collect_latest_api_data(
                    api=api,
                    current_round=3,
                    season=2026,
                    token="token-123",
                )

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
