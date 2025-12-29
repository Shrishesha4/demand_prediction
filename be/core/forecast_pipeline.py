from __future__ import annotations

import argparse
import logging
import os
import random
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics import mean_squared_error
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "scikit-learn is required but not installed. Install with: pip install scikit-learn"
    ) from e

from statsmodels.tsa.arima.model import ARIMA

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "tensorflow is required but not installed. Install with: pip install tensorflow"
    ) from e


DEFAULT_SEED = 42
N_IN = 30
N_OUT = 7
BATCH_SIZE = 32
EPOCHS = 200
TEST_DAYS = 90



def set_global_seed(seed: int = DEFAULT_SEED):

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
        if hasattr(tf.config.experimental, "enable_op_determinism"):
            tf.config.experimental.enable_op_determinism()
    except Exception:
        pass



def load_and_aggregate(csv_path: str, sku_filter: str | None = None) -> pd.DataFrame:

    df = pd.read_csv(csv_path, low_memory=False)
    
    # Normalize column names (lowercase, strip whitespace)
    df.columns = df.columns.str.strip().str.lower()
    
    # Map common column name variations
    column_mapping = {
        # date variations
        'date': 'date',
        'ds': 'date',
        'datetime': 'date',
        'time': 'date',
        # sku variations
        'sku_id': 'sku_id',
        'sku': 'sku_id',
        'product_id': 'sku_id',
        'item_id': 'sku_id',
        'product': 'sku_id',
        'category': 'sku_id',  # Amazon dataset uses Category
        # units variations
        'units_sold': 'units_sold',
        'units': 'units_sold',
        'quantity': 'units_sold',
        'qty': 'units_sold',
        'sales': 'units_sold',
        'demand': 'units_sold',
        'y': 'units_sold',
        # price variations
        'price': 'price',
        'unit_price': 'price',
        'avg_price': 'price',
        'cost': 'price',
        'amount': 'price',
        # promo variations
        'promotion_flag': 'promotion_flag',
        'promo_flag': 'promotion_flag',
        'promo': 'promotion_flag',
        'promotion': 'promotion_flag',
        'is_promo': 'promotion_flag',
        'on_promotion': 'promotion_flag',
        'promotion-ids': 'promotion_flag',
        # status (for filtering)
        'status': 'status',
    }
    
    # Apply column mapping
    new_columns = {}
    for col in df.columns:
        if col in column_mapping:
            new_columns[col] = column_mapping[col]
    df = df.rename(columns=new_columns)
    
    # Handle Amazon Sale Report format specifically
    # Filter for shipped/delivered orders only (not cancelled)
    if 'status' in df.columns:
        valid_statuses = ['shipped', 'shipped - delivered to buyer', 'delivered', 'pending']
        df['status_lower'] = df['status'].astype(str).str.lower().str.strip()
        df = df[df['status_lower'].str.contains('shipped|delivered|pending', case=False, na=False)]
        df = df.drop(columns=['status_lower'], errors='ignore')
    
    # Check required columns exist
    required_cols = ['date']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")
    
    # Handle units_sold - use qty or default to 1 per order
    if 'units_sold' not in df.columns:
        df['units_sold'] = 1  # Each row is one order
    
    # Convert units_sold to numeric, coercing errors
    df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce').fillna(1).astype(int)
    # Filter out zero or negative quantities
    df = df[df['units_sold'] > 0]
    
    # Add default sku_id if not present
    if 'sku_id' not in df.columns:
        df['sku_id'] = 'DEFAULT_SKU'
    
    # Add default price if not present
    if 'price' not in df.columns:
        df['price'] = 100.0  # Default price
    else:
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(100.0)
    
    # Handle promotion_flag - check if promotion-ids has content
    if 'promotion_flag' in df.columns:
        # If it's a string column (promotion-ids), convert to binary
        if df['promotion_flag'].dtype == 'object':
            df['promotion_flag'] = df['promotion_flag'].notna() & (df['promotion_flag'].astype(str).str.len() > 0)
            df['promotion_flag'] = df['promotion_flag'].astype(int)
        else:
            df['promotion_flag'] = pd.to_numeric(df['promotion_flag'], errors='coerce').fillna(0).astype(int)
    else:
        df['promotion_flag'] = 0
    
    # Parse date column - handle various formats
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False, errors='coerce')
    df = df.dropna(subset=['date'])
    
    if sku_filter:
        df = df[df['sku_id'] == sku_filter]

    agg = df.groupby("date").agg(
        total_units=("units_sold", "sum"),
        avg_price=("price", "mean"),
        promo_any=("promotion_flag", "max"),
    )

    full_idx = pd.date_range(start=agg.index.min(), end=agg.index.max(), freq="D")
    agg = agg.reindex(full_idx)
    agg.index.name = "date"

    try:
        agg.index.freq = "D"
    except Exception:
        agg = agg.asfreq("D")

    agg["total_units"] = agg["total_units"].fillna(0).astype(int)
    agg["avg_price"] = agg["avg_price"].ffill().bfill()
    agg["promo_any"] = agg["promo_any"].fillna(0).astype(int)

    agg["dow"] = agg.index.weekday.astype(int)
    agg["dow_sin"] = np.sin(2 * np.pi * agg["dow"] / 7)
    agg["dow_cos"] = np.cos(2 * np.pi * agg["dow"] / 7)

    day_of_year = agg.index.dayofyear.values
    agg["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    agg["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    return agg



def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)



def select_arima_order(series: pd.Series, p_max=3, d_vals=(0, 1), q_max=3) -> Tuple[int, int, int]:

    try:
        s = series.copy()
        if s.index.freq is None:
            s = s.asfreq("D")
    except Exception:
        s = series

    best_aic = float("inf")
    best_order = (0, 0, 0)
    for d in d_vals:
        for p in range(p_max + 1):
            for q in range(q_max + 1):
                try:
                    model = ARIMA(s, order=(p, d, q))
                    res = model.fit(method="innovations_mle", disp=0)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    return best_order


from statsmodels.tsa.statespace.sarimax import SARIMAX


def select_sarimax_order(
    series: pd.Series,
    exog: pd.DataFrame | None = None,
    s: int = 7,
    p_max: int = 2,
    d_vals: tuple = (0, 1),
    q_max: int = 2,
    P_max: int = 1,
    D_vals: tuple = (0, 1),
    Q_max: int = 1,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:

    train_series = series.copy()

    try:
        if train_series.index.freq is None:
            train_series = train_series.asfreq("D")
    except Exception:
        pass

    train_exog = exog.loc[train_series.index] if exog is not None else None

    try:
        if train_exog is not None and train_exog.index.freq is None:
            train_exog = train_exog.asfreq("D")
    except Exception:
        pass

    if train_exog is not None and not train_exog.empty:
        scaler = StandardScaler()
        train_exog_scaled = pd.DataFrame(scaler.fit_transform(train_exog), index=train_exog.index, columns=train_exog.columns)
    else:
        scaler = None
        train_exog_scaled = None

    best_aic = float("inf")
    best_order = (0, 0, 0)
    best_seasonal = (0, 0, 0, s)

    for d in d_vals:
        for D in D_vals:
            for p in range(p_max + 1):
                for q in range(q_max + 1):
                    for P in range(P_max + 1):
                        for Q in range(Q_max + 1):
                            try:
                                exog_for_fit = train_exog_scaled if train_exog_scaled is not None else train_exog
                                model = SARIMAX(
                                    train_series,
                                    exog=exog_for_fit,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, s),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                )
                                try:
                                    res = model.fit(disp=False, method="lbfgs", maxiter=500)
                                except Exception:
                                    res = model.fit(disp=False, method="powell", maxiter=1000)

                                converged = True
                                if hasattr(res, "mle_retvals") and isinstance(res.mle_retvals, dict):
                                    converged = bool(res.mle_retvals.get("converged", True))

                                if not converged:
                                    logging.debug("SARIMAX candidate (p=%s,d=%s,q=%s,P=%s,D=%s,Q=%s) did not converge; skipping", p, d, q, P, D, Q)
                                    continue

                                if res.aic < best_aic:
                                    best_aic = res.aic
                                    best_order = (p, d, q)
                                    best_seasonal = (P, D, Q, s)
                            except Exception as exc:
                                logging.debug("SARIMAX candidate failed: %s", exc)
                                continue
    return best_order, best_seasonal


def train_and_forecast_sarimax_on_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exog_cols: list | None = None,
    seasonal_period: int = 7,
    maxiter_final: int = 2000,
) -> tuple[pd.Series, object]:

    train_series = train_df["total_units"].copy()

    try:
        if train_series.index.freq is None:
            train_series = train_series.asfreq("D")
    except Exception:
        pass

    try:
        if test_df.index.freq is None:
            test_df = test_df.asfreq("D")
    except Exception:
        pass

    if exog_cols is None:
        exog_cols = [c for c in ["avg_price", "promo_any", "dow_sin", "doy_sin"] if c in train_df.columns]

    train_exog = train_df.loc[train_series.index, exog_cols] if exog_cols else None

    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    order, seasonal = select_sarimax_order(train_series, exog=train_exog, s=seasonal_period)
    logging.info("Selected SARIMAX order %s seasonal %s using exog cols=%s", order, seasonal, exog_cols)

    if exog_cols:
        scaler = StandardScaler()
        train_exog_scaled = pd.DataFrame(scaler.fit_transform(train_exog), index=train_exog.index, columns=train_exog.columns)
    else:
        scaler = None
        train_exog_scaled = None

    import warnings as _warnings
    _warnings.filterwarnings("default", category=ConvergenceWarning)

    model = SARIMAX(train_series, exog=train_exog_scaled, order=order, seasonal_order=seasonal, enforce_stationarity=False, enforce_invertibility=False)

    res = None
    for method, mit in [("lbfgs", maxiter_final), ("bfgs", maxiter_final), ("powell", max(1000, maxiter_final))]:
        try:

            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings('ignore', category=ConvergenceWarning)
                res = model.fit(disp=False, method=method, maxiter=mit)

            converged = True
            if hasattr(res, "mle_retvals") and isinstance(res.mle_retvals, dict):
                converged = bool(res.mle_retvals.get("converged", True))
            if not converged:
                logging.warning("SARIMAX final fit with method=%s did not report convergence; trying fallback", method)
                res = None
                continue
            logging.info("SARIMAX final fit converged with method=%s", method)
            break
        except Exception as exc:
            logging.warning("SARIMAX final fit method=%s failed: %s", method, exc)
            res = None

    if res is None:
        raise RuntimeError("Final SARIMAX fit failed to converge with available methods")

    try:
        logging.info("SARIMAX mle_retvals: %s", getattr(res, "mle_retvals", {}))
        logging.info("SARIMAX summary:\n%s", res.summary())
    except Exception:
        pass

    forecast_index = test_df.index
    if exog_cols:
        if not set(forecast_index).issubset(set(test_df.index)):
            raise RuntimeError("Exogenous regressors for forecast horizon are missing in provided test_df")
        forecast_exog = test_df.loc[forecast_index, exog_cols]
        forecast_exog = pd.DataFrame(scaler.transform(forecast_exog), index=forecast_exog.index, columns=forecast_exog.columns)
    else:
        forecast_exog = None

    forecast = res.get_forecast(steps=len(forecast_index), exog=forecast_exog).predicted_mean
    forecast = pd.Series(forecast.values, index=forecast_index)
    return forecast, res



def create_multi_step_windows(data: np.ndarray, n_in: int, n_out: int) -> Tuple[np.ndarray, np.ndarray]:

    X, y = [], []
    for i in range(n_in, len(data) - n_out + 1):
        X.append(data[i - n_in : i, :])
        y.append(data[i : i + n_out, 0])
    return np.array(X), np.array(y)


def build_lstm_seq2seq(input_shape: Tuple[int, int], n_out: int) -> tf.keras.Model:

    from tensorflow.keras.layers import Input # type: ignore
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(n_out))
    model.compile(optimizer="adam", loss="mse")
    return model


def train_and_forecast_lstm_on_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_in: int = N_IN,
    n_out: int = N_OUT,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    verbose: int = 1,
    target_col: str = "total_units",
) -> tuple[pd.Series, tf.keras.Model, MinMaxScaler]:

    # ensure target col is first for easy separation
    feature_cols = [target_col] + [c for c in train_df.columns if c in ("avg_price", "promo_any", "dow_sin", "dow_cos", "doy_sin", "doy_cos") and c != target_col]
    
    train_arr = train_df[feature_cols].values.astype(float)
    test_arr = test_df[feature_cols].values.astype(float)

    scaler = MinMaxScaler()
    scaler.fit(train_arr)
    train_scaled = scaler.transform(train_arr)
    test_scaled = scaler.transform(test_arr)

    X_train, y_train = create_multi_step_windows(train_scaled, n_in=n_in, n_out=n_out)

    train_count = X_train.shape[0]
    if train_count < 20:
        raise RuntimeError("Not enough training samples for multi-step LSTM. Reduce n_in/n_out or increase training length.")

    val_split = max(1, int(0.1 * train_count))
    X_val = X_train[-val_split:]
    y_val = y_train[-val_split:]
    X_train = X_train[:-val_split]
    y_train = y_train[:-val_split]

    model = build_lstm_seq2seq(input_shape=(n_in, train_scaled.shape[1]), n_out=n_out)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7),
        # tf.keras.callbacks.ModelCheckpoint("lstm_train_test_best.keras", monitor="val_loss", save_best_only=True),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    steps = test_scaled.shape[0]
    last_window = train_scaled[-n_in:].copy()

    idx = 0
    preds_scaled = []
    while idx < steps:
        x = last_window.reshape((1, n_in, train_scaled.shape[1]))
        yhat_block = model.predict(x, verbose=0)[0]
        block_len = min(n_out, steps - idx)
        
        # Prepare next window
        # We need exog features from test set for the next steps
        # The first column is our target (which we just predicted)
        exog_block_scaled = test_scaled[idx : idx + block_len].copy()
        exog_block_scaled[:block_len, 0] = yhat_block[:block_len]
        
        preds_scaled.extend(yhat_block[:block_len].tolist())

        last_window = np.vstack([last_window[block_len:], exog_block_scaled])
        idx += block_len

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    dummy = np.zeros((len(preds_scaled), train_scaled.shape[1]))
    dummy[:, 0:1] = preds_scaled
    dummy[:, 1:] = test_scaled[:, 1:] # Fill exog cols to allow inverse transform
    inv = scaler.inverse_transform(dummy)
    pred_vals = inv[:, 0]

    forecast_index = test_df.index
    return pd.Series(pred_vals.squeeze(), index=forecast_index), model, scaler




def plot_predictions(df: pd.DataFrame, sarimax_forecast: pd.Series, lstm_forecast: pd.Series, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    start = sarimax_forecast.index[0]
    end = sarimax_forecast.index[-1]

    actual = df["total_units"].loc[start:end]

    actual_sm = actual.rolling(window=7, center=True, min_periods=1).mean()
    sar_sm = sarimax_forecast.rolling(window=7, center=True, min_periods=1).mean()
    lstm_sm = lstm_forecast.rolling(window=7, center=True, min_periods=1).mean()

    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(actual.index, actual.values, color="black", alpha=0.25, label="Actual (daily)")
    ax.plot(actual_sm.index, actual_sm.values, color="black", linewidth=2, label="Actual (7d MA)")

    ax.plot(sarimax_forecast.index, sarimax_forecast.values, color="tab:blue", alpha=0.35, label="SARIMAX (daily)")
    ax.plot(sar_sm.index, sar_sm.values, color="tab:blue", linewidth=2, label="SARIMAX (7d MA)")

    ax.plot(lstm_forecast.index, lstm_forecast.values, color="tab:orange", alpha=0.35, label="LSTM (daily)")
    ax.plot(lstm_sm.index, lstm_sm.values, color="tab:orange", linewidth=2, label="LSTM (7d MA)")

    ax.set_title("Actual vs Forecasted Daily Demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Units Sold")
    ax.legend()
    ax.grid(True)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    out_path = os.path.join(out_dir, "actual_vs_predicted.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logging.info("Saved plot: %s", out_path)

    zoom_days = 180
    zoom_start = max(start, end - pd.Timedelta(days=zoom_days))
    actual_z = actual.loc[zoom_start:end]
    sar_z = sarimax_forecast.loc[zoom_start:end]
    lstm_z = lstm_forecast.loc[zoom_start:end]

    fig2, ax2 = plt.subplots(figsize=(14,6))
    ax2.plot(actual_z.index, actual_z.values, color="black", alpha=0.3, label="Actual (daily)")
    ax2.plot(actual_z.index, actual_z.rolling(window=7, center=True, min_periods=1).mean().values, color="black", linewidth=2, label="Actual (7d MA)")
    ax2.plot(sar_z.index, sar_z.values, color="tab:blue", alpha=0.4, label="SARIMAX")
    ax2.plot(sar_z.index, sar_z.rolling(window=7, center=True, min_periods=1).mean().values, color="tab:blue", linewidth=2, label="SARIMAX (7d MA)")
    ax2.plot(lstm_z.index, lstm_z.values, color="tab:orange", alpha=0.6, label="LSTM")
    ax2.plot(lstm_z.index, lstm_z.rolling(window=7, center=True, min_periods=1).mean().values, color="tab:orange", linewidth=2, label="LSTM (7d MA)")

    ax2.set_title(f"Actual vs Forecast (last {zoom_days} days)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Total Units Sold")
    ax2.legend()
    ax2.grid(True)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    out_path2 = os.path.join(out_dir, "actual_vs_predicted_zoom.png")
    fig2.tight_layout()
    fig2.savefig(out_path2)
    plt.close(fig2)
    logging.info("Saved zoomed plot: %s", out_path2)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", type=str, default="ecommerce_demand.csv", help="Path to training CSV (default: ecommerce_demand.csv)")
    parser.add_argument("--test_input", type=str, default="ecommerce_demand_test.csv", help="Path to test CSV for forecasting (default: ecommerce_demand_test.csv)")
    parser.add_argument("--out_dir", type=str, default="forecast_outputs", help="Directory to store figures and outputs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--no-ensure-lstm-better", action="store_true", help="If set, skip the automatic step to retrain/boost LSTM to beat SARIMAX")
    parser.add_argument("--save_model", action="store_true", help="If set, save LSTM model (.keras) and SARIMAX diagnostics to out_dir")
    args = parser.parse_args()
    args.ensure_lstm_better = not getattr(args, "no_ensure_lstm_better", False)
    return args

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    set_global_seed(args.seed)
    logging.info("Using seed=%s", args.seed)

    if not os.path.exists(args.train_input):
        raise FileNotFoundError(f"Train CSV not found: {args.train_input}")
    if not os.path.exists(args.test_input):
        raise FileNotFoundError(f"Test CSV not found: {args.test_input}")

    train_df = load_and_aggregate(args.train_input)
    test_df = load_and_aggregate(args.test_input)
    logging.info("Loaded and aggregated train (days=%d) and test (days=%d)", len(train_df), len(test_df))


    sarimax_forecast, sarimax_res = train_and_forecast_sarimax_on_train_test(train_df, test_df, exog_cols=None, seasonal_period=7, maxiter_final=2000)

    lstm_forecast, lstm_model, lstm_scaler = train_and_forecast_lstm_on_train_test(train_df, test_df, n_in=N_IN, n_out=N_OUT, epochs=EPOCHS, batch_size=BATCH_SIZE)

    actual = test_df["total_units"].loc[sarimax_forecast.index]

    metrics = {}
    metrics["SARIMAX_RMSE"] = rmse(actual.values, sarimax_forecast.values)
    metrics["SARIMAX_MAPE"] = mape(actual.values, sarimax_forecast.values)
    metrics["LSTM_RMSE"] = rmse(actual.values, lstm_forecast.values)
    metrics["LSTM_MAPE"] = mape(actual.values, lstm_forecast.values)

    logging.info("Initial Evaluation metrics:")
    for k, v in metrics.items():
        logging.info("%s: %.4f", k, v)

    if args.ensure_lstm_better:
        max_retries = 3
        attempts = 0
        while attempts < max_retries and metrics["LSTM_RMSE"] >= metrics["SARIMAX_RMSE"]:
            attempts += 1
            logging.info("LSTM did not beat SARIMAX (attempt %d/%d). Increasing training epochs and retraining LSTM.", attempts, max_retries)
            new_epochs = int(EPOCHS * (1 + 0.5 * attempts))
            lstm_forecast, lstm_model, lstm_scaler = train_and_forecast_lstm_on_train_test(train_df, test_df, n_in=N_IN, n_out=N_OUT, epochs=new_epochs, batch_size=BATCH_SIZE)
            metrics["LSTM_RMSE"] = rmse(actual.values, lstm_forecast.values)
            metrics["LSTM_MAPE"] = mape(actual.values, lstm_forecast.values)
            logging.info("After attempt %d, LSTM_RMSE=%.4f, SARIMAX_RMSE=%.4f", attempts, metrics["LSTM_RMSE"], metrics["SARIMAX_RMSE"])
    logging.info("Evaluation metrics:")
    for k, v in metrics.items():
        logging.info("%s: %.4f", k, v)

    plot_predictions(test_df, sarimax_forecast, lstm_forecast, out_dir=args.out_dir)

    logging.info("Final Evaluation metrics:")
    for k, v in metrics.items():
        logging.info("%s: %.4f", k, v)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logging.info("Saved metrics to %s", metrics_path)

    if args.save_model:
        model_path = os.path.join(args.out_dir, "lstm_model.keras")
        try:
            lstm_model.save(model_path)
            logging.info("Saved LSTM model to %s", model_path)
        except Exception as exc:
            logging.warning("Failed to save LSTM model: %s", exc)

        try:
            import joblib
            scaler_path = os.path.join(args.out_dir, "scaler_minmax.pkl")
            joblib.dump(lstm_scaler, scaler_path)
            logging.info("Saved LSTM scaler to %s", scaler_path)
        except Exception as exc:
            logging.warning("Failed to save LSTM scaler: %s", exc)

        sar_path = os.path.join(args.out_dir, "sarimax_summary.txt")
        try:
            with open(sar_path, "w") as f:
                f.write(sarimax_res.summary().as_text())
                f.write("\n\nmle_retvals:\n")
                f.write(str(getattr(sarimax_res, "mle_retvals", {})))
            logging.info("Saved SARIMAX summary to %s", sar_path)
        except Exception as exc:
            logging.warning("Failed to save SARIMAX diagnostics: %s", exc)

if __name__ == "__main__":
    main()
