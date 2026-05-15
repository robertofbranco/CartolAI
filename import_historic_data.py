"""Compatibility CLI for importing finished Cartola seasons."""

import argparse
import logging
import os

from dotenv import load_dotenv

from cartola_data.historic import *  # noqa: F403
from cartola_data.historic import import_historic_season

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import previous Cartola seasons")
    parser.add_argument("--year", type=int, nargs="+", default=[2025])
    parser.add_argument("--rodadas", type=int, nargs="+", default=None)
    parser.add_argument("--skip-gato", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("CARTOLA_TOKEN")
    for year in args.year:
        import_historic_season(
            year=year,
            rounds=args.rodadas,
            token=token,
            collect_gato_data=not args.skip_gato,
        )


if __name__ == "__main__":
    main()
