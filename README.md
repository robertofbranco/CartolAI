# Cartola FC — ML Team Builder

Pipeline de Machine Learning para montar o melhor time no Cartola FC a cada rodada do Brasileirão.

---

## Como funciona

```
API Cartola  →  Coleta histórica  →  Engenharia de Features  →  LightGBM  →  ILP Optimizer  →  Time escalado
```

### 4 etapas principais

| Etapa | O que faz |
|-------|----------|
| **Coleta** | Busca pontuações históricas de todas as rodadas via `/atletas/pontuados/{rodada}` |
| **Features** | Cria indicadores temporais: média móvel 3/5/10 rodadas, desvio padrão, tendência, mando de campo, regularidade |
| **Modelo** | Treina LightGBM (regressão L1/MAE) para prever pontos esperados de cada jogador |
| **Otimização** | Programação linear inteira (ILP via PuLP) escolhe os 12 melhores jogadores respeitando formação, budget e máx. por clube |

---

## Instalação

```bash
pip install -r requirements.txt
```

---

## Uso

### Básico (rodada atual, sem autenticação)
```bash
python cartola_team_builder.py
```

### Com token (recomendado para salvar escalação)
```bash
python cartola_team_builder.py --token SEU_TOKEN_AQUI
```

### Especificar rodada e budget
```bash
python cartola_team_builder.py --rodada 15 --budget 130.0
```

### Usando no código Python
```python
from cartola_team_builder import run_pipeline

time_df, model = run_pipeline(
    token="seu_token",   # opcional
    rodada_alvo=15,
    budget=140.0,
)
```

---

## Features do modelo

| Feature | Descrição |
|---------|----------|
| `media_pts_3r` | Média de pontos nas 3 rodadas anteriores |
| `media_pts_5r` | Média de pontos nas 5 rodadas anteriores |
| `media_pts_10r` | Média de pontos nas 10 rodadas anteriores |
| `std_pts_3r/5r` | Desvio padrão — mede inconsistência |
| `pts_ultima_rodada` | Pontuação na rodada imediatamente anterior |
| `tendencia` | `media_3r - media_10r` — está melhorando ou piorando? |
| `regularidade_5r` | % das últimas 5 rodadas em que o jogador pontuou |
| `preco_lag1` | Preço na rodada anterior (proxy de qualidade) |
| `mando` | 1=casa, -1=fora, 0=desconhecido |
| `acc_scout_G/A/SG/DD/GS` | Scouts acumulados nas últimas 5 rodadas |
| `posicao_enc` | ID de posição (1=GOL, 2=LAT, 3=ZAG, 4=MEI, 5=ATA, 6=TEC) |
| `clube_enc` | Clube codificado |

---

## Restrições da otimização

- Formação padrão **4-3-3** (ajustável via `FORMATION`)  
  → 1 GOL + 2 LAT + 2 ZAG + 3 MEI + 3 ATA + 1 TEC = 12 jogadores
- **Budget**: 140 cartoletas (padrão)
- **Máximo 5 jogadores do mesmo clube**
- **1 capitão** (pontuação dobrada)
- Jogadores suspensos/machucados excluídos automaticamente

---

## Estrutura de arquivos

```
cartola_ml/
├── cartola_team_builder.py   # Pipeline principal
├── requirements.txt
├── README.md
└── data/
    ├── cache/                # Cache de respostas da API
    ├── historico.parquet     # Histórico de pontuações coletado
    └── time_rodada_N.csv     # Time gerado para cada rodada
```

---

## Obtendo o token (X-GLB-Token)

1. Acesse [cartola.globo.com](https://cartola.globo.com) e faça login
2. Abra o DevTools do navegador (F12) → aba **Network**
3. Filtre por `api.cartola.globo.com`
4. Copie o valor do header `X-GLB-Token` de qualquer request

---

## Próximos passos sugeridos

- [ ] Adicionar feature de **adversário** (força do time contra)
- [ ] Incluir **odds de mercado** para prever probabilidade de SG/gols
- [ ] Treinar modelo separado por posição
- [ ] Otimizar banco de reservas (substituições)
- [ ] Backtesting rodada a rodada para medir ROI do modelo
