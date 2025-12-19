from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

DEFAULT_RANDOM_SEED = 42
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
SKUS = ["SKU_001", "SKU_002", "SKU_003"]

BASE_DEMAND = {"SKU_001": 80.0, "SKU_002": 45.0, "SKU_003": 18.0}
BASE_PRICE = {"SKU_001": 19.99, "SKU_002": 9.99, "SKU_003": 49.99}
WEEKLY_STRENGTH = {"SKU_001": 0.20, "SKU_002": 0.28, "SKU_003": 0.12}
YEARLY_STRENGTH = {"SKU_001": 0.30, "SKU_002": 0.22, "SKU_003": 0.08}
TREND_FRACTION = {"SKU_001": 0.12, "SKU_002": 0.10, "SKU_003": 0.18}

DAILY_PRICE_NOISE_STD = 0.007
DAILY_PRICE_NOISE_STD_SKU = {"SKU_001": DAILY_PRICE_NOISE_STD, "SKU_002": DAILY_PRICE_NOISE_STD, "SKU_003": DAILY_PRICE_NOISE_STD * 3.0}
PRICE_ELASTICITY_SKU = {"SKU_001": -1.5, "SKU_002": -1.5, "SKU_003": -0.8}

PROMO_PROB_RANGE = (0.15, 0.25)
PROMO_DISCOUNT_RANGE = (0.05, 0.20)
PROMO_DISCOUNT_PROB = 0.6
PROMO_LIFT_RANGE = (0.25, 0.60)
PROMO_WINDOW_MEAN = 4
PROMO_WINDOW_MIN = 2
PROMO_WINDOW_MAX = 8

SHOCK_PROB = 0.01
SHOCK_MULT_RANGE = (1.5, 3.0)
DISPERSION_SHAPE_SKU3 = 0.9


def generate_dataset(
    out_file: str = "ecommerce_demand.csv",
    seed: int = DEFAULT_RANDOM_SEED,
    variant_params: Optional[Dict[str, Any]] = None,
    currency: str = "USD",
    variant_name: str = "base",
    n_locations: int = 1,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    n_days = len(dates)
    day_idx = np.arange(n_days)
    day_of_year = dates.dayofyear.values

    vp = variant_params or {}
    base_price_scale = vp.get("base_price_scale", 1.0)
    price_min_factor = vp.get("price_min_factor", {s: (0.8 if s != "SKU_003" else 0.7) for s in SKUS})
    price_max_factor = vp.get("price_max_factor", {s: (1.2 if s != "SKU_003" else 1.3) for s in SKUS})
    reversion_rate = vp.get("reversion_rate", {s: (0.04 if s != "SKU_003" else 0.02) for s in SKUS})
    price_noise_multiplier = vp.get("price_noise_multiplier", {s: 1.0 for s in SKUS})

    rows = []
    for loc in range(n_locations):
        loc_demand_mult = rng.normal(1.0, 0.05)
        loc_price_shift = rng.normal(0.0, 0.02)
        for sku in SKUS:
            base = BASE_DEMAND[sku] * loc_demand_mult
            base_price = BASE_PRICE[sku] * base_price_scale * (1.0 + loc_price_shift)

            weekly_component = WEEKLY_STRENGTH[sku] * base * np.sin(2 * np.pi * day_idx / 7)
            yearly_component = YEARLY_STRENGTH[sku] * base * np.sin(2 * np.pi * day_of_year / 365.25)
            trend_total = base * TREND_FRACTION[sku]
            trend = trend_total * (day_idx / float(n_days - 1))

            noise_std = max(1.0, base * 0.12)
            noise = rng.normal(loc=0.0, scale=noise_std, size=n_days)

            promo_p = rng.uniform(PROMO_PROB_RANGE[0], PROMO_PROB_RANGE[1])
            if sku == "SKU_003":
                expected_promo_days = int(np.round(promo_p * n_days))
                avg_len = vp.get("promo_window_mean", PROMO_WINDOW_MEAN)
                num_windows = max(1, expected_promo_days // max(1, avg_len))
                promotion_flag = np.zeros(n_days, dtype=int)
                starts = rng.choice(np.arange(n_days), size=num_windows, replace=False)
                for s in starts:
                    length = rng.integers(vp.get("promo_window_min", PROMO_WINDOW_MIN), vp.get("promo_window_max", PROMO_WINDOW_MAX)+1)
                    end = min(n_days, s + int(length))
                    promotion_flag[s:end] = 1
                discount_pct = np.zeros(n_days)
                window_days = promotion_flag.astype(bool)
                if window_days.sum() > 0:
                    discount_pct[window_days] = rng.uniform(0.08, 0.25, size=window_days.sum())
                promo_lift = rng.uniform(PROMO_LIFT_RANGE[0] * 1.2, PROMO_LIFT_RANGE[1] * 1.2)
            else:
                promotion_flag = rng.binomial(1, promo_p, size=n_days)
                promo_lift = rng.uniform(PROMO_LIFT_RANGE[0], PROMO_LIFT_RANGE[1])
                will_discount = (rng.random(n_days) < PROMO_DISCOUNT_PROB) & (promotion_flag == 1)
                discount_pct = np.zeros(n_days)
                if will_discount.sum() > 0:
                    discount_pct[will_discount] = rng.uniform(PROMO_DISCOUNT_RANGE[0], PROMO_DISCOUNT_RANGE[1], size=will_discount.sum())

            price = np.empty(n_days, dtype=float)
            price[0] = base_price * (1.0 + rng.normal(0.0, 0.003))

            min_f = price_min_factor[sku] if isinstance(price_min_factor, dict) else price_min_factor
            max_f = price_max_factor[sku] if isinstance(price_max_factor, dict) else price_max_factor
            rev = reversion_rate[sku] if isinstance(reversion_rate, dict) else reversion_rate
            local_price_noise = DAILY_PRICE_NOISE_STD_SKU.get(sku, DAILY_PRICE_NOISE_STD) * (price_noise_multiplier.get(sku, 1.0) if isinstance(price_noise_multiplier, dict) else price_noise_multiplier)

            price_season_amp = 0.012 * base_price
            season_phase = rng.uniform(0, 2 * np.pi)
            for t in range(1, n_days):
                drift = rev * (base_price - price[t - 1])
                p_noise = price[t - 1] * rng.normal(0.0, local_price_noise)
                price_candidate = price[t - 1] + drift + p_noise
                if discount_pct[t] > 0:
                    price_candidate = price_candidate * (1.0 - discount_pct[t])
                season = price_season_amp * np.sin(2 * np.pi * t / 7 + season_phase)
                jitter = rng.normal(0.0, 0.03 * base_price)
                price_candidate += season + jitter
                lower_bound = base_price * min_f
                upper_bound = base_price * max_f
                if price_candidate <= lower_bound:
                    price_candidate = lower_bound + rng.uniform(0.005 * base_price, 0.03 * base_price)
                if price_candidate >= upper_bound:
                    price_candidate = upper_bound - rng.uniform(0.005 * base_price, 0.03 * base_price)
                price[t] = float(np.clip(price_candidate, lower_bound, upper_bound))
            price = np.round(price.clip(min=0.01), 2)

            price_mean = price.mean()
            price_effect = PRICE_ELASTICITY_SKU[sku] * (price - price_mean) / price_mean * base
            units_raw = base + trend + weekly_component + yearly_component + noise + price_effect

            shocks = (rng.random(n_days) < SHOCK_PROB)
            if shocks.any():
                shock_mults = rng.uniform(SHOCK_MULT_RANGE[0], SHOCK_MULT_RANGE[1], size=shocks.sum())
                units_raw[shocks] = units_raw[shocks] * shock_mults

            units_promoted = units_raw * (1.0 + (promotion_flag * promo_lift))

            lam = np.clip(units_promoted, a_min=0.1, a_max=None)
            if sku == "SKU_003":
                shape = vp.get("dispersion_shape_sku3", DISPERSION_SHAPE_SKU3)
                lam_gamma = rng.gamma(shape, (lam / shape).clip(min=1e-6))
                units_final = rng.poisson(lam_gamma).astype(int)
            else:
                units_final = rng.poisson(lam).astype(int)
            units_final = np.maximum(0, units_final)

            sku_col = np.array([sku] * n_days)
            rows.append(pd.DataFrame({
                "date": dates.strftime("%Y-%m-%d"),
                "sku_id": sku_col,
                "units_sold": units_final,
                "price": price,
                "promotion_flag": promotion_flag,
            }))

    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(["date", "sku_id"]).reset_index(drop=True)

    expected_rows = n_days * len(SKUS) * n_locations
    if df.shape[0] != expected_rows:
        raise RuntimeError(f"Unexpected number of rows: {df.shape[0]} != {expected_rows}")

    if currency.upper() == "INR":
        df["price"] = (df["price"] * 100).round(2)

    df.to_csv(out_file, index=False)
    print(f"Saved {out_file} (variant={variant_name}, currency={currency}, locations={n_locations})")
    print(f"Dataset shape: {df.shape} (rows, cols)")
    return df
