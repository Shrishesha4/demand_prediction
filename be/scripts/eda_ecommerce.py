from pathlib import Path
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
    print("matplotlib not installed; skipping plots. Install with `pip install matplotlib` to enable plots.")

import argparse

p = argparse.ArgumentParser()
p.add_argument("--input", required=False, default="ecommerce_demand.csv", help="Input CSV file path")
p.add_argument("--out-dir", required=False, default="eda_outputs", help="Directory to save plotting outputs")
args = p.parse_args()

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(exist_ok=True)

csv_file = Path(args.input)
if not csv_file.exists():
    raise FileNotFoundError(f"Missing expected file: {csv_file}")

df = pd.read_csv(csv_file, parse_dates=["date"]) 

print("\n--- Basic info ---")
print("Shape:", df.shape)
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
print("Columns:", list(df.columns))
print("\nMissing values per column:\n", df.isnull().sum())

neg_sales = (df["units_sold"] < 0).sum()
print(f"Negative units_sold count: {neg_sales}")

print("\n--- Per-SKU summary ---")
sku_groups = df.groupby("sku_id")
summary = sku_groups.agg({
    "units_sold": ["mean", "std"],
    "price": ["mean"],
    "promotion_flag": ["mean"],
})
summary.columns = ["mean_units","std_units","mean_price","promo_frac"]
summary = summary[["mean_units","std_units","mean_price","promo_frac"]]
print(summary.round(3).to_string())

promo_stats = []
for sku, g in sku_groups:
    mean_promo = g.loc[g["promotion_flag"]==1, "units_sold"].mean()
    mean_no = g.loc[g["promotion_flag"]==0, "units_sold"].mean()
    lift = (mean_promo - mean_no) / mean_no if mean_no and not np.isnan(mean_no) else np.nan
    promo_stats.append((sku, mean_no, mean_promo, lift))

promo_df = pd.DataFrame(promo_stats, columns=["sku_id","mean_no_promo","mean_promo","relative_lift"])
print("\nPromotion effect (avg units, no-promo vs promo):\n", promo_df.round(3).to_string(index=False))

daily_total = df.groupby("date")["units_sold"].sum()

if HAVE_MPL:
    plt.figure(figsize=(12,4))
    plt.plot(daily_total.index, daily_total.values, lw=0.8)
    plt.title("Total units sold per day (all SKUs)")
    plt.xlabel("Date")
    plt.ylabel("Units sold")
    plt.tight_layout()
    fn_total = OUT_DIR / "total_units_timeseries.png"
    plt.savefig(fn_total)
    plt.close()
else:
    print("Skipping total units timeseries plot (matplotlib not available)")

if HAVE_MPL:
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12,9), sharex=True)
    for ax, sku in zip(axes, sorted(df["sku_id"].unique())):
        s = df.loc[df["sku_id"]==sku].set_index("date")["units_sold"].sort_index()
        ax.plot(s.index, s.rolling(14, min_periods=1).mean(), label=f"{sku} (14d MA)")
        ax.set_ylabel("Units")
        ax.legend()
    fig.suptitle("Per-SKU 14-day moving average of units sold")
    plt.tight_layout(rect=[0,0,1,0.97])
    fn_per_sku = OUT_DIR / "per_sku_timeseries.png"
    plt.savefig(fn_per_sku)
    plt.close()
else:
    print("Skipping per-SKU timeseries plots (matplotlib not available)")

df["weekday"] = df["date"].dt.weekday
wk = df.groupby(["sku_id","weekday"])["units_sold"].mean().reset_index()
pivot = wk.pivot(index="weekday", columns="sku_id", values="units_sold")
pivot.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
if HAVE_MPL:
    ax = pivot.plot(kind="bar", figsize=(10,5))
    ax.set_ylabel("Avg units sold")
    ax.set_title("Avg units sold by weekday and SKU")
    plt.tight_layout()
    fn_weekday = OUT_DIR / "weekday_seasonality.png"
    plt.savefig(fn_weekday)
    plt.close()
else:
    print("Skipping weekday seasonality plot (matplotlib not available)")

promo_compare = df.groupby(["sku_id","promotion_flag"])["units_sold"].mean().unstack()
promo_compare.columns = ["no_promo","promo"]
if HAVE_MPL:
    promo_compare.plot(kind="bar", figsize=(8,4))
    plt.title("Average units sold: promo vs no-promo")
    plt.ylabel("Avg units")
    plt.tight_layout()
    fn_promo = OUT_DIR / "promotion_impact.png"
    plt.savefig(fn_promo)
    plt.close()
else:
    print("Skipping promotion impact plot (matplotlib not available)")

for sku in sorted(df["sku_id"].unique()):
    g = df[df["sku_id"]==sku].sample(n=min(1000, len(df)), random_state=0)
    if HAVE_MPL:
        plt.figure(figsize=(6,4))
        plt.scatter(g["price"], g["units_sold"], alpha=0.5, s=10)
        plt.xlabel("Price")
        plt.ylabel("Units sold")
        plt.title(f"Price vs units sold ({sku})")
        plt.tight_layout()
        fn = OUT_DIR / f"price_vs_units_{sku}.png"
        plt.savefig(fn)
        plt.close()
    else:
        print(f"Skipping price vs units scatter for {sku} (matplotlib not available)")

promo_df.to_csv(OUT_DIR / "promo_summary.csv", index=False)
summary.round(3).to_csv(OUT_DIR / "per_sku_summary.csv")

artifacts = [str(p) for p in sorted(OUT_DIR.iterdir())]
print("\nSaved artifacts:")
for a in artifacts:
    print(" -", a)

print("\nHighlights:")
for _, row in promo_df.iterrows():
    print(f"{row['sku_id']}: mean (no promo)={row['mean_no_promo']:.2f}, mean(promo)={row['mean_promo']:.2f}, lift={(row['relative_lift']*100):.1f}%")

print("\nDone.")
