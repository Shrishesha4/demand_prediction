import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf as sm_plot_acf, plot_pacf as sm_plot_pacf
except Exception as e:
    raise RuntimeError("statsmodels is required for this script. Install with: pip install statsmodels")

warnings.filterwarnings("ignore")


def run_adf_test(series, title="Series"):
    
    print(f"\n--- ADF test for: {title} ---")
    res = adfuller(series, autolag="AIC")
    stat = res[0]
    pvalue = res[1]
    usedlag = res[2]
    nobs = res[3]
    crit = res[4]

    print(f"ADF Statistic: {stat:.5f}")
    print(f"p-value: {pvalue:.5f}")
    print(f"Used lag: {usedlag}, Observations: {nobs}")
    print("Critical values:")
    for k, v in crit.items():
        print(f"  {k}: {v:.5f}")

    stationary = (pvalue < 0.05) and (stat < crit.get("5%", 0))
    if stationary:
        print("Interpretation: The series appears to be STATIONARY (rejects unit root null at 5% level).")
    else:
        print("Interpretation: The series appears to be NON-STATIONARY (fail to reject unit root null at 5% level).")

    return {"stat": stat, "pvalue": pvalue, "crit": crit, "stationary": stationary}


def plot_acf(series, lags=40, out_path: Path = None, title="ACF"):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    sm_plot_acf(series, lags=lags, ax=ax, alpha=0.05)
    ax.set_title(f"Autocorrelation Function (ACF) - {title}")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path)
        print(f"Saved ACF plot to {out_path}")
    plt.close(fig)


def plot_pacf(series, lags=40, out_path: Path = None, title="PACF"):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    sm_plot_pacf(series, lags=lags, ax=ax, alpha=0.05, method="ywm")
    ax.set_title(f"Partial Autocorrelation Function (PACF) - {title}")
    ax.set_xlabel("Lag")
    ax.set_ylabel("PACF")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path)
        print(f"Saved PACF plot to {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=False, default="ecommerce_demand.csv", help="Input CSV file path")
    p.add_argument("--out-dir", required=False, default="stationarity_outputs", help="Output directory for plots")
    p.add_argument("--max-lag", type=int, default=40, help="Max lag for ACF/PACF plots")
    args = p.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path, parse_dates=["date"]) 
    print(f"Loaded {input_path} with shape {df.shape}")

    agg = df.groupby("date")["units_sold"].sum().sort_index()

    full_idx = pd.date_range(start=agg.index.min(), end=agg.index.max(), freq="D")
    agg = agg.reindex(full_idx, fill_value=0)
    agg.index.name = "date"

    print("\nFirst 5 aggregated daily totals:")
    print(agg.head().to_string())

    res_orig = run_adf_test(agg, title="Aggregated total daily demand (original)")

    diff1 = agg.diff().dropna()
    res_diff = run_adf_test(diff1, title="First-order differenced series")

    if res_orig["stationary"]:
        series_for_ac = agg
        series_label = "Original"
    elif res_diff["stationary"]:
        series_for_ac = diff1
        series_label = "First-differenced"
    else:

        series_for_ac = diff1
        series_label = "First-differenced (not stationary)"
        print("Warning: Neither original nor first-differenced series passed stationarity test at 5% level. Using differenced series for ACF/PACF diagnostics.")

    fig_ts = plt.figure(figsize=(12, 4))
    ax_ts = fig_ts.add_subplot(111)
    ax_ts.plot(agg.index, agg.values, label="Total units sold")
    ax_ts.set_title("Aggregated daily units sold (all SKUs)")
    ax_ts.set_xlabel("Date")
    ax_ts.set_ylabel("Units sold")
    ax_ts.legend()
    plt.tight_layout()
    ts_path = out_dir / "aggregated_timeseries.png"
    fig_ts.savefig(ts_path)
    plt.close(fig_ts)
    print(f"Saved time-series plot to {ts_path}")

    acf_path = out_dir / f"acf_{series_label.replace(' ', '_')}.png"
    pacf_path = out_dir / f"pacf_{series_label.replace(' ', '_')}.png"
    plot_acf(series_for_ac, lags=args.max_lag, out_path=acf_path, title=series_label)
    plot_pacf(series_for_ac, lags=args.max_lag, out_path=pacf_path, title=series_label)

    print("\n--- Recommendation for ARIMA ---")
    if res_orig["stationary"]:
        print("Use the original series (d=0) for ARIMA as it is stationary.")
    elif res_diff["stationary"]:
        print("Use first differencing (d=1) for ARIMA as differenced series is stationary.")
    else:
        print("Consider higher order differencing or seasonal differencing; current tests indicate non-stationarity.")


if __name__ == "__main__":
    main()
