"""Compatibility CLI for current-season Cartola data collection."""

import argparse
import logging
import os

from dotenv import load_dotenv

from cartola_data import *  # noqa: F403
from cartola_data import CURRENT_SEASON, collect_current_season

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cartola FC data collector")
    parser.add_argument(
        "--rodadas",
        type=int,
        nargs="+",
        default=None,
        help="Rounds to collect. Defaults to every closed round in the current season.",
    )
    parser.add_argument(
        "--temporada",
        type=int,
        default=CURRENT_SEASON,
        help="Season to collect. Defaults to the configured current season.",
    )
    args = parser.parse_args()

    collect_current_season(
        temporada=args.temporada,
        rodadas=args.rodadas,
        token=os.environ.get("CARTOLA_TOKEN"),
    )


if __name__ == "__main__":
    main()
