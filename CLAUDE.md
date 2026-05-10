# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the three stages of the pipeline (each must run in order — later stages read parquet files written by earlier ones):
```bash
# 1. Collect historical rounds into data/historico.parquet + data/partidas.parquet
python cartola_collector.py                    # all rounds up to current
python cartola_collector.py --rodadas 1 5 10   # specific rounds only

# 2. Train model + optimize lineup for the target round → data/time_rodada_N.csv
python cartola_team_builder.py
python cartola_team_builder.py --rodada 15 --budget 130

# 3. Backtest the model across a range of past rounds → data/backtest_*.csv/.png
python cartola_backtest.py --inicio 6 --fim 13 --budget 200
```

There are no tests, linters, or build steps configured.

## Architecture

The project is a three-stage Cartola FC ML pipeline. Each stage is a standalone script that imports symbols from the previous stage — there is no package layout, just three top-level Python files with a strict import chain:

```
cartola_backtest.py  →  cartola_team_builder.py  →  cartola_collector.py
```

When changing shared behavior, edit the lowest stage that owns it and let the upper stages inherit:

- **`cartola_collector.py`** owns `CartolaAPI`, `BASE_URL`, `DATA_DIR`, `BUDGET`, `FORMATION`, `POSICAO_NOME`, `SCOUT_POINTS`. It writes `data/historico.parquet` (one row per atleta×rodada) and `data/partidas.parquet` (mando de campo: 1=home, -1=away). The API wrapper caches non-mutable GETs to `data/cache/` as JSON; `mercado_status` and `atletas_mercado` deliberately bypass cache.
- **`cartola_team_builder.py`** owns `construir_features`, `treinar_modelo`, `otimizar_escalacao`, `FEATURE_COLS`, and `run_pipeline`. Reads the parquet files, builds features, trains a LightGBM L1 regressor, then runs an ILP via PuLP to pick a 12-player lineup.
- **`cartola_backtest.py`** reuses `construir_features`/`treinar_modelo`/`otimizar_escalacao` and adds two more PuLP problems (`calcular_teto` for the oracle ceiling, `calcular_baseline_media` for the `media_num` baseline). All three ILPs share the same constraint shape — formation counts, budget, max 5 per club, exactly one captain — so changes to constraints must be mirrored across all three.

### Data leakage rules (important)

`construir_features` uses `groupby("atleta_id")[col].shift(1).rolling(...)` for every rolling stat. The `shift(1)` is the only thing keeping training-time features from peeking at the target round's points — do not remove it. Similarly, `treinar_modelo` partitions strictly by round (`rodada < rodada_corte - 3` for train, `[rodada_corte - 3, rodada_corte)` for validation), and the backtest re-trains per round using only `df_hist[df_hist["rodada"] < rodada_alvo]`. Any new feature must follow the same shift-then-aggregate pattern.

### Lineup ILP shape

The objective is `Σ pred_i · x_i + (capitao_bonus - 1) · pred_i · cap_i`, with `cap_i ≤ x_i` and exactly one captain. Only players with `status_id == 7` (Provável — escalado para a próxima rodada) are kept before optimization; `2`=Dúvida, `3`=Suspenso, `5`=Contundido, `6`=Nulo are filtered out. The canonical mapping comes from `https://api.cartola.globo.com/atletas/status`. Note that `status_id` in `historico.parquet` is the *post-game* state (e.g., every TEC ends a round as 7, injured players become 5), so it can't be used to reconstruct pre-round availability — `cartola_backtest.py` forces `status_id = 7` on every row of `df_rodada_real` since each row came from `/atletas/pontuados/{rodada}` (the player was on the field, therefore Provável at decision time). Position counts come from `FORMATION` keyed by `posicao_id` (1=GOL, 2=LAT, 3=ZAG, 4=MEI, 5=ATA, 6=TEC); changing the formation dict is the supported way to change the lineup shape.

### Live-prediction enrichment pattern

When predicting for a future round (in both `run_pipeline` and `rodar_backtest`), only the most recent round's features are joined onto the market snapshot, and missing values are filled with the per-position median. If you add a feature, make sure it is included in this merge — otherwise it will be silently zero at inference.

## Auth

Both `cartola_collector.py` and `cartola_team_builder.py` read the Cartola `X-GLB-Token` from `$CARTOLA_TOKEN` (or `--token`); the value must include the `Bearer ` prefix because it goes straight into the `Authorization` header. The token is optional — unauthenticated reads of `/atletas/pontuados/{rodada}`, `/clubes`, `/partidas/{rodada}` work for backtest data collection. Earlier commits contained a hardcoded JWT in `main()` of both scripts; rotate that token at globo.com — it is still in git history.
