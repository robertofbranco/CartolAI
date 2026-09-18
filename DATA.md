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
pós-rodada: na linha R, eles descrevem o mercado aberto depois de R e podem ser
usados para prever R+1.

Na coleta corrente, a API de pontos só pode complementar a rodada imediatamente
anterior, cujo estado pós-rodada corresponde ao `mercado_atual`. Rodadas mais
antigas exigem o snapshot histórico do caRtola; um mercado posterior não pode
ser associado a elas.

| Momento | Pode ser usado para prever a rodada R? | Exemplos |
|---|---|---|
| Antes do fechamento de R | Sim | elenco/posição disponíveis, partida (mando e adversário), odds de R, histórico até R-1 |
| Depois de R | Não como input de R | `pontos`, `jogou`, scouts e placar de R; `preco`, `media` e `jogos` da linha R pertencem ao mercado de R+1 |

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

Forma e disponibilidade são sinais distintos. `media_pts_ultimas_3_aparicoes`,
`media_pts_ultimas_5_aparicoes` e `std_pts_ultimas_5_aparicoes` consideram
somente jogos em que o atleta entrou em campo. `aparicoes_5r`, `aparicoes_10r`,
`regularidade_*`, `rodadas_desde_ultima_aparicao`, `sequencia_aparicoes` e
`aparicoes_anteriores` medem frequência e confiabilidade no calendário. Todas
essas features usam apenas rodadas anteriores à linha alvo.

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
