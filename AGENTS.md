# CartolAI agent guide

## Architecture and commands

- `cartola_data/` owns API clients, collection, normalization, parquet paths,
  and `read_datasets()`.
- `feature_engineering.py` turns player, match, and odds data into features.
- `cartola_model_training.py` trains one regressor per `posicao_id`.
- `cartola_team_builder.py` prepares the current market and selects a lineup.
- `cartola_backtest.py` retrains and simulates that flow for every target round.

Use `python import_historic_data.py --year ...` for completed seasons and
`python collect_latest_data.py` for the configured current season. Use
`python -m pytest` for the test suite. Do not change generated parquets,
notebooks, or `results/` unless the task explicitly requires regenerated data.

## Data contract and time safety

Read `DATA.md` before changing datasets or features. The key invariant is that
each `jogadores_por_rodada` row is post-round state keyed by
`(temporada, rodada, atleta_id)`.

- `pontos`, played status, `scout_*`, `preco`, `media`, and `jogos` from row R
  must not become inputs for predicting R.
- Preserve the shifts in `build_features()`: market values are shifted by one
  player-round and rolling player/club/scout features use prior rounds only.
- Preserve the backtest boundary: target-round roster/context may be used, but
  target result columns and match scores must be nulled before feature building;
  real points are used only after lineup selection.
- When adding a feature, document its availability time and add a focused test
  that would fail if target-round information leaked.

## Project conventions

- Keep `data/<dataset>_<year>.parquet` as collection partitions. The collector
  merges them into `data/<dataset>.parquet`; merge keys are season, round, and
  athlete or club as appropriate. Deduplication prefers a played player row.
- Keep collection sources and fallback order explicit. Historical players come
  from caRtola; current missing rounds use the Cartola API; CBF supplies
  supported historical scores; Gato Mestre supplies odds with `CARTOLA_TOKEN`.
- Runtime settings belong in `cartola_data/config.py`: season, formation,
  odds filter, captain settings, and model tuning. Do not invent CLI options
  that the parsers do not expose.
- The builder currently has no budget or per-club constraint. Preserve that
  behavior unless the requested change explicitly adds it.

## Selection and validation

- Train/validate by time: previous seasons plus early target-season rows train;
  the five rounds before `round_limit` validate.
- Selection filters to `STATUS["Provavel"]`, applies odds only when eligible
  candidates exist, uses the configured formation, and gives the captain the
  configured multiplier. Luxury reserves are limited to MEI/ATA and are applied
  during the backtest using real results.
- Extend the nearest test module in `tests/` for behavior changes:
  collection/data in `test_data_collection.py`, features in
  `test_feature_engineering.py`, selection in `test_team_builder.py`, and
  simulation/reporting in `test_backtest.py` or `test_plot_backtest_report.py`.
