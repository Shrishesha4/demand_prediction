"""Run A/B variants of the synthetic dataset generation and produce EDA outputs for each.

Generates two variants:
 - Variant A (tight prices, stronger mean reversion)
 - Variant B (looser prices, higher noise, longer promos)

Outputs are saved as:
 - ecommerce_demand_variant_{A,B}_INR.csv
 - EDA outputs under eda_outputs/variant_{A,B}/
"""
import subprocess
import sys
from pathlib import Path
from be.scripts.generate_lib import generate_dataset

ROOT = Path(__file__).resolve().parent
EDA_SCRIPT = ROOT / "eda_ecommerce.py"

variants = {
    "A": {
        "desc": "Tight prices, strong reversion",
        "params": {
            "base_price_scale": 100.0,
            "price_min_factor": {"SKU_001": 0.9, "SKU_002": 0.9, "SKU_003": 0.88},
            "price_max_factor": {"SKU_001": 1.1, "SKU_002": 1.1, "SKU_003": 1.12},
            "reversion_rate": {"SKU_001": 0.06, "SKU_002": 0.06, "SKU_003": 0.05},
            "price_noise_multiplier": {"SKU_001": 0.9, "SKU_002": 0.9, "SKU_003": 1.2},
            "promo_window_mean": 3,
            "promo_window_min": 2,
            "promo_window_max": 5,
            "dispersion_shape_sku3": 1.2,
        },
    },
    "B": {
        "desc": "Looser prices, higher noise, longer promos",
        "params": {
            "base_price_scale": 100.0,
            "price_min_factor": {"SKU_001": 0.8, "SKU_002": 0.78, "SKU_003": 0.72},
            "price_max_factor": {"SKU_001": 1.2, "SKU_002": 1.25, "SKU_003": 1.28},
            "reversion_rate": {"SKU_001": 0.03, "SKU_002": 0.03, "SKU_003": 0.02},
            "price_noise_multiplier": {"SKU_001": 1.2, "SKU_002": 1.2, "SKU_003": 2.0},
            "promo_window_mean": 6,
            "promo_window_min": 3,
            "promo_window_max": 9,
            "dispersion_shape_sku3": 0.7,
        },
    },
}

artifacts = []

for name, v in variants.items():
    out_csv = ROOT / f"ecommerce_demand_variant_{name}_INR.csv"
    print(f"Generating variant {name}: {v['desc']}")
    generate_dataset(out_file=str(out_csv), seed=42, variant_params=v["params"], currency="INR", variant_name=name, n_locations=9)
    artifacts.append(str(out_csv))

    out_dir = ROOT / "eda_outputs" / f"variant_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running EDA for variant {name} -> {out_dir}")
    cmd = [sys.executable, str(EDA_SCRIPT), "--input", str(out_csv), "--out-dir", str(out_dir)]
    subprocess.run(cmd, check=True)
    artifacts.extend([str(p) for p in sorted(out_dir.iterdir())])

print("\nA/B run completed. Artifacts:")
for a in artifacts:
    print(" -", a)
