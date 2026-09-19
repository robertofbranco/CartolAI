# CartolAI

Ferramentas para coletar dados do Cartola FC, criar features temporais,
prever pontuação por posição, montar uma escalação e avaliá-la em backtests.
Consulte [DATA.md](DATA.md) para o contrato dos datasets e a semântica temporal
das linhas de jogadores.

## Instalação

Requer Python 3.10+. Em um ambiente virtual:

```bash
python -m pip install -e .
```

Odds do Gato Mestre exigem `CARTOLA_TOKEN`; os fluxos públicos em geral não.
Crie um `.env` na raiz quando precisar delas:

```dotenv
CARTOLA_TOKEN=seu_token_aqui
```

Não versione nem publique o token.

## Fluxo usual

```bash
# Base de treino: temporadas concluídas
python import_historic_data.py --year 2023 2024 2025

# Temporada em CURRENT_SEASON: mercado, rodadas concluídas e consolidação
python collect_latest_data.py

# Treino por posição e escalação da rodada atual
python cartola_team_builder.py
```

`CURRENT_SEASON`, `FORMATION`, o bônus de capitão e o filtro de odds ficam em
`cartola_data/config.py`.

## Coleta e dados

`collect_latest_data.py` atualiza o snapshot de mercado, odds (quando há token),
partidas, médias dos cartoleiros, chaves de liga e jogadores de rodadas
encerradas. Os dados são gravados primeiro por temporada e depois consolidados
em `data/*.parquet`, que são os arquivos consumidos pelo modelo.

```bash
# Apenas consolida arquivos anuais existentes
python collect_latest_data.py --merge-only

# Consolida somente os datasets indicados
python collect_latest_data.py --merge-only --merge-datasets jogadores_por_rodada partidas odds

# Atualiza somente o mercado (e odds, se houver token); não consolida
python collect_latest_data.py --mercado-only
```

O importador histórico usa o repositório caRtola para jogadores, a CBF para
placares quando a temporada é suportada e o Gato Mestre para odds quando houver
token:

```bash
python import_historic_data.py --year 2025 --rodadas 1 2 3
python import_historic_data.py --year 2025 --skip-gato --skip-cbf
```

## Modelo, escalação e backtest

As features usam somente informações anteriores à rodada prevista. Há modelos
por posição: `random_forest` (padrão) ou `gradient_boosting` (LightGBM).

```bash
python cartola_team_builder.py --model-strategy gradient_boosting
python cartola_backtest.py --temporada 2026 --inicio 10 --fim 17
python plot_backtest_report.py --output-folder results
```

O construtor considera apenas atletas `Provável`, aplica o filtro de odds e a
formação configurada e inclui o bônus de capitão. Ele não expõe `--token`,
`--rodada` ou `--budget`, nem impõe orçamento máximo ou limite por clube.

O backtest retreina rodada a rodada, simula o mercado no fechamento, compara o
time aos pontos reais e grava CSVs e gráficos na pasta de saída. O `teto` é um
oracle calculado após a rodada, usado apenas como referência de eficiência.

## Estrutura

```text
cartola_data/              APIs, coleta, normalização e acesso a parquets
feature_engineering.py     features temporais e contextuais
cartola_model_training.py  treino e validação por posição
cartola_team_builder.py    montagem da escalação atual
cartola_backtest.py        simulação e relatórios por rodada
tests/                     testes automatizados
```

Para instruções de alteração assistida por IA, veja [AGENTS.md](AGENTS.md).
