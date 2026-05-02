"""
Generates realistic synthetic historical data for backtesting
when the Cartola API is unavailable (off-season / no token).
"""

import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

N_ROUNDS   = 25
N_CLUBS    = 20
PLAYERS_PER_CLUB = {1: 2, 2: 4, 3: 5, 4: 7, 5: 6, 6: 1}  # pos_id -> count per club

CLUBS = [
    (10, "Flamengo"), (11, "Palmeiras"), (12, "São Paulo"), (13, "Corinthians"),
    (14, "Santos"),   (15, "Grêmio"),    (16, "Internacional"), (17, "Atlético-MG"),
    (18, "Cruzeiro"), (19, "Bahia"),     (20, "Fluminense"),    (21, "Vasco"),
    (22, "Botafogo"), (23, "Athletico-PR"), (24, "Fortaleza"),  (25, "Ceará"),
    (26, "Bragantino"),  (27, "Cuiabá"), (28, "Goiás"),         (29, "Coritiba"),
]

# Mean points and std per position (realistic Cartola distributions)
POS_STATS = {
    1: (3.5, 5.0),   # GOL: high variance, SG bonus
    2: (3.0, 3.5),   # LAT
    3: (2.8, 3.0),   # ZAG
    4: (3.5, 4.0),   # MEI
    5: (3.8, 5.5),   # ATA: high variance, G bonus
    6: (2.5, 2.5),   # TEC
}

# Price ranges per position (cartoletas)
POS_PRICE = {
    1: (4, 14), 2: (4, 16), 3: (4, 16),
    4: (4, 22), 5: (4, 22), 6: (3, 10),
}

records = []
atleta_id = 1000

for club_id, club_nome in CLUBS:
    for pos_id, n_players in PLAYERS_PER_CLUB.items():
        mu, sigma = POS_STATS[pos_id]
        p_low, p_high = POS_PRICE[pos_id]

        for p in range(n_players):
            # Give each player a fixed quality multiplier (star players score more)
            quality = rng.uniform(0.5, 1.8)
            base_price = rng.uniform(p_low, p_high)
            apelido = f"P{atleta_id}"

            prev_pts = []
            for rodada in range(1, N_ROUNDS + 1):
                # ~15% chance of not playing
                played = rng.random() > 0.15
                if played:
                    raw = rng.normal(mu * quality, sigma)
                    pts = max(-5.0, raw)
                else:
                    pts = 0.0

                # Price drifts with recent performance
                if prev_pts:
                    recent_avg = np.mean(prev_pts[-3:])
                    price_delta = (recent_avg - mu) * 0.05
                    base_price = float(np.clip(base_price + price_delta, p_low, p_high))

                # Scout columns (simplified)
                scout_G  = int(pts > 7 and pos_id in (4, 5) and rng.random() > 0.6)
                scout_A  = int(pts > 5 and pos_id in (4, 5) and rng.random() > 0.5)
                scout_SG = int(pts > 4 and pos_id in (1, 3) and rng.random() > 0.4)
                scout_GS = int(pts < -0.5 and pos_id == 1 and rng.random() > 0.5)
                scout_DD = int(pos_id == 1 and rng.random() > 0.7)

                records.append({
                    "rodada":     rodada,
                    "atleta_id":  atleta_id,
                    "apelido":    apelido,
                    "posicao_id": pos_id,
                    "clube_id":   club_id,
                    "clube_nome": club_nome,
                    "status_id":  2,          # available
                    "pontos":     round(pts, 2),
                    "preco":      round(base_price, 2),
                    "media":      round(float(np.mean(prev_pts[-5:])) if prev_pts else mu * quality, 2),
                    "jogos":      rodada,
                    "scout_G":    scout_G,
                    "scout_A":    scout_A,
                    "scout_SG":   scout_SG,
                    "scout_GS":   scout_GS,
                    "scout_DD":   scout_DD,
                })
                prev_pts.append(pts)

            atleta_id += 1

df = pd.DataFrame(records)
df = df.sort_values(["atleta_id", "rodada"]).reset_index(drop=True)
df.to_parquet(DATA_DIR / "historico.parquet", index=False)
print(f"Saved {len(df)} rows | {df['atleta_id'].nunique()} players | {df['rodada'].nunique()} rounds")
print(df.head())
