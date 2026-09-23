# Dados

Os parquets vivem em `data/`. O coletor escreve arquivos anuais primeiro e
`collect_latest_data.py` gera as versões sem sufixo; `read_datasets()` lê estas
versões consolidadas.

| Dataset | Grão / chave | Origem | Uso principal |
|---|---|---|---|
| `jogadores_por_rodada[_AAAA]` | atleta por temporada e rodada; `(temporada, rodada, atleta_id)` | caRtola histórico; API Cartola como fallback corrente | alvo, histórico e proxy de mercado no backtest |
| `partidas[_AAAA]` | clube por temporada e rodada; `(temporada, rodada, clube_id)` | Cartola corrente; CBF histórica; Gato Mestre como fallback | mando, adversário e placares históricos |
| `odds[_AAAA]` | clube por temporada e rodada; `(temporada, rodada, clube_id)` | Gato Mestre | `prob_win`, `prob_draw`, `prob_loss` |
| `mercado_atual` | atleta da rodada aberta; `(temporada, rodada, atleta_id)` | `/atletas/mercado` | escalação da rodada aberta e finalização do status da rodada anterior |
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

O coletor identifica cada snapshot de `mercado_atual` com temporada e rodada.
Quando a API avança para a rodada seguinte, antes de substituir esse arquivo,
o último `status_id` observado no mercado da rodada encerrada é copiado para
as linhas correspondentes de `jogadores_por_rodada`. Os demais campos não são
alterados. Assim, o backtest usa a disponibilidade mais próxima do fechamento
que foi efetivamente coletada.

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

Forma e disponibilidade são sinais distintos. Médias, mediana, desvio, último
ponto e EWMA de aparições consideram somente jogos em que o atleta entrou em
campo; `tendencia_pts_aparicoes` compara a EWMA com a média das últimas dez
aparições. `aparicoes_*`, `regularidade_*`, distância desde a última aparição,
sequência, histórico e `sem_historico` medem frequência e confiabilidade no
calendário. Todas usam apenas rodadas anteriores à linha alvo. Sem histórico,
forma/contagens recebem zero, `sem_historico=1` e a distância recebe 10.

Na escalação real, `mercado_atual` fornece o elenco aberto e os valores atuais;
o contexto da partida e as odds da rodada alvo são juntados depois. No
backtest, a linha histórica da rodada alvo é apenas uma **proxy do elenco de
mercado**: o pipeline mantém seus atributos de elenco, anula `pontos`,
`jogou`, `entrou_em_campo` e `scout_*`, anula os placares da partida alvo e
reconstrói as features com o histórico até R-1. Depois da seleção, somente os
resultados reais de R são usados para pontuar o time.

## Perfis posicionais de clube e adversário

Os perfis posicionais usam no máximo os cinco jogos concluídos anteriores da
mesma temporada. Cada taxa é regularizada para a média expansiva da liga,
calculada somente com rodadas anteriores, usando cinco pseudo-observações:
`(soma_observada + 5 * media_liga) / (contagem_observada + 5)`. Sem qualquer
histórico anterior da liga, a taxa permanece ausente para ser imputada apenas
com dados de treino.

- `pontos_cedidos_adv_pos_total_5r` mede o total esperado por jogo que o
  adversário cede à posição da própria linha; sua contagem é de partidas.
- `pontos_cedidos_adv_pos_por_jogador_5r` divide o mesmo numerador pelos atletas
  da posição que efetivamente participaram; sua contagem é de participantes.
- Finalizações usam as diferenças por rodada de `FD + FF + FT`. Para jogadores
  de linha, as features representam finalizações cedidas pelo adversário; para
  GOL, representam o volume produzido pelo adversário.
- GOL combina esse volume com `DE` por aparição e gols sofridos pelo próprio
  clube. LAT e ZAG combinam frequência de saldo de gols do clube, frequência de
  gol do adversário, `DS` e riscos disciplinares (`FC`, `CA`, `CV`). Ambos
  preservam histórico ofensivo (`G`, `A`, `FD`, `FF`, `FT`); LAT recebe também
  os componentes detalhados das finalizações cedidas.

Colunas terminadas em `_obs_5r` ou `_obs_5a` expõem, respectivamente, o número
de partidas/participantes ou aparições reais usado antes da regularização. Uma
participação é uma linha com `jogou` ou `entrou_em_campo` verdadeiro e resultado
concluído. Resultados, scouts e placares da rodada alvo nunca entram nesses
perfis.

## Consumidores

`cartola_team_builder.py` lê os três datasets consolidados, treina por posição
e acrescenta o snapshot atual. `cartola_backtest.py` reexecuta esse fluxo para
cada rodada, produzindo `backtest_resultados.csv`,
`backtest_mae_por_posicao.csv`, `backtest_time_escalado.csv` e gráficos na
pasta de saída. Os datasets de médias e chaves alimentam apenas sobreposições e
benchmarks dos relatórios.
