"""Creates a synthetic e-commerce dataset (data/cart_data.csv).

Replace this file's output with a real dataset later (see README) --
just keep the same column names.
"""
import numpy as np
import pandas as pd
from pathlib import Path

def generate(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    browsing = np.clip(rng.gamma(3, 5, n), 1, 60)            # minutes on site
    viewed = np.clip(rng.poisson(8, n) + 1, 1, 40)           # products viewed
    cart = np.clip(rng.lognormal(4.3, 0.7, n), 10, 1000)     # cart value
    prev = rng.poisson(2.5, n)                               # previous purchases
    disc = rng.binomial(1, 0.4, n)                           # discount used?

    zc = (np.log(cart) - 4.3) / 0.7
    zb = (browsing - 15) / 8
    zv = (viewed - 9) / 3
    zp = (prev - 2.5) / 1.6

    # Cart value matters most, then browsing time, products viewed,
    # previous purchases, and discount usage the least (as in the project slides).
    z = (-0.3 + 1.5 * zc + 0.9 * zb + 0.5 * zv - 0.35 * zp - 0.2 * disc
         + 0.6 * ((zc > 1) & (disc == 0))          # expensive carts w/o discount: non-linear effect
         + rng.normal(0, 0.8, n))
    p = 1 / (1 + np.exp(-z))
    y = rng.binomial(1, p)

    return pd.DataFrame({
        "browsing_time": browsing.round(1),
        "products_viewed": viewed,
        "cart_value": cart.round(2),
        "previous_purchases": prev,
        "discount_usage": disc,
        "abandoned": y,
    })

if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "cart_data.csv"
    df = generate()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}  (abandon rate: {df.abandoned.mean():.1%})")
