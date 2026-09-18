# Dados

Os parquets vivem em `data/`. O coletor escreve arquivos anuais primeiro e
`collect_latest_data.py` gera as versões sem sufixo; `read_datasets()` lê estas
versões consolidadas.

| Dataset | Grão / chave | Origem | Uso principal |
|---|---|---|---|
| `jogadores_por_rodada[_AAAA]` | atleta por temporada e rodada; `(temporada, rodada, atleta_id)` | caRtola histórico; API Cartola como fallback corrente | alvo, histórico e proxy de mercado no backtest |
| `partidas[_AAAA]` | clube por temporada e rodada; `(temporada, rodada, clube_id)` | Cartola corrente; CBF histórica; Gato Mestre como fallback | mando, adversário e placares históricos |
| `odds[_AAAA]` | clube por temporada e rodada; `(temporada, rodada, clube_id)` | Gato Mestre | `prob_win`, `prob_draw`, `prob_loss` |
| `mercado_atual` | atleta; `atleta_id` | `/atletas/mercado` | escalação da rodada aberta |
| `medias_cartoleiros[_AAAA]` | temporada e rodada | API Cartola | comparação nos relatórios |
| `chaves_ligas` | liga, rodada e chave; `(liga, rodada, chave_index)` | API autenticada de liga | comparação de playoffs nos relatórios |

`_AAAA` separa a coleta por temporada. As versões consolidadas relevantes para
modelagem são `jogadores_por_rodada.parquet`, `partidas.parquet` e
`odds.parquet`. Respostas HTTP podem ficar em `data/cache/`.

## `jogadores_por_rodada`: estado após a rodada

Cada linha representa o registro do atleta **após o resultado da rodada**. As
colunas de identidade e elenco (`atleta_id`, `apelido`, `posicao_id`,
`clube_id`, `status_id`) descrevem o atleta na linha. `pontos`, `jogou`/
`entrou_em_campo` e `scout_*` são resultados da rodada; os scouts são valores
acumulados na fonte e o projeto calcula a diferença por rodada para as
features. `preco`, `media` e `jogos` também são tratados como estado
pós-rodada.

Na coleta corrente, quando o histórico caRtola não contém uma rodada, a API de
pontos é enriquecida com o `mercado_atual` salvo no momento da coleta. Portanto,
esses campos de mercado podem ser um snapshot posterior, não uma reprodução
perfeita do fechamento histórico. O deslocamento temporal continua obrigatório.

| Momento | Pode ser usado para prever a rodada R? | Exemplos |
|---|---|---|
| Antes do fechamento de R | Sim | elenco/posição disponíveis, partida (mando e adversário), odds de R, histórico até R-1 |
| Depois de R | Não como input de R | `pontos`, `jogou`, scouts, `preco`, `media`, `jogos` registrados na linha R, placar de R |

Consequentemente, não una uma linha R diretamente ao treino ou mercado da
própria R sem aplicar a transformação temporal abaixo.

## Como as features evitam vazamento

`build_features(players, partidas, odds)` ordena por `(temporada, atleta_id,
rodada)` e conserva `pontos` como alvo da linha. Para prever R, ele:

1. desloca `preco` e `media` uma rodada por atleta (`preco_lag1` e `media_lag`);
2. calcula médias, desvios, regularidade, último ponto e scouts usando
   `shift(1)` antes das janelas móveis;
3. calcula métricas de clube/adversário e gols com partidas anteriores;
4. junta mando, adversário e odds da rodada alvo. Sem odds, usa probabilidades
   neutras de `1/3`.

Na escalação real, `mercado_atual` fornece o elenco aberto e os valores atuais;
o contexto da partida e as odds da rodada alvo são juntados depois. No
backtest, a linha histórica da rodada alvo é apenas uma **proxy do elenco de
mercado**: o pipeline mantém seus atributos de elenco, anula `pontos`,
`jogou`, `entrou_em_campo` e `scout_*`, anula os placares da partida alvo e
reconstrói as features com o histórico até R-1. Depois da seleção, somente os
resultados reais de R são usados para pontuar o time.

## Consumidores

`cartola_team_builder.py` lê os três datasets consolidados, treina por posição
e acrescenta o snapshot atual. `cartola_backtest.py` reexecuta esse fluxo para
cada rodada, produzindo `backtest_resultados.csv`,
`backtest_mae_por_posicao.csv`, `backtest_time_escalado.csv` e gráficos na
pasta de saída. Os datasets de médias e chaves alimentam apenas sobreposições e
benchmarks dos relatórios.
