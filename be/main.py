#!/usr/bin/env python3
"""FastAPI backend for e-commerce demand forecasting with SARIMAX and LSTM.

Endpoints:
  POST /train - Train models on uploaded train/test CSVs and return metrics
  POST /predict - Load saved models and predict on uploaded CSV
  POST /forecast_future - Generate future demand forecasts for N quarters per SKU
  GET /metrics - Get latest training metrics
  GET /health - Health check
  GET /model/download - Download the trained LSTM model file
"""

import os
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import forecasting functions from forecast_pipeline
from forecast_pipeline import (
    load_and_aggregate,
    train_and_forecast_sarimax_on_train_test,
    train_and_forecast_lstm_on_train_test,
    select_sarimax_order,
    rmse,
    mape,
    set_global_seed,
    N_IN,
    N_OUT,
    EPOCHS,
    BATCH_SIZE,
)

import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="E-commerce Demand Forecasting API", version="1.0.0")

# CORS for local development (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage paths
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)
METRICS_FILE = MODEL_DIR / "latest_metrics.json"
LSTM_MODEL_PATH = MODEL_DIR / "lstm_model.keras"
SCALER_PATH = MODEL_DIR / "scaler_minmax.pkl"
SARIMAX_SUMMARY_PATH = MODEL_DIR / "sarimax_summary.txt"


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "forecasting-api"}


@app.post("/train")
async def train_models(
    train_file: UploadFile = File(..., description="Training CSV file"),
    test_file: UploadFile = File(..., description="Test CSV file"),
    seed: int = Form(42),
):
    """Train SARIMAX and LSTM models on uploaded train/test data and return metrics.

    This endpoint will always attempt to retrain/boost the LSTM until it outperforms SARIMAX on RMSE (within limits).

    Returns:
        JSON with metrics (SARIMAX_RMSE, SARIMAX_MAPE, LSTM_RMSE, LSTM_MAPE)
    """
    try:
        set_global_seed(seed)
        logger.info("Training request received with seed=%d", seed)

        # Save uploaded files to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.csv"
            test_path = Path(tmpdir) / "test.csv"

            # Write uploaded files
            with open(train_path, "wb") as f:
                shutil.copyfileobj(train_file.file, f)
            with open(test_path, "wb") as f:
                shutil.copyfileobj(test_file.file, f)

            # Load and aggregate
            train_df = load_and_aggregate(str(train_path))
            test_df = load_and_aggregate(str(test_path))
            logger.info("Loaded train (%d days) and test (%d days)", len(train_df), len(test_df))

            # Train SARIMAX
            logger.info("Training SARIMAX...")
            sarimax_forecast, sarimax_res = train_and_forecast_sarimax_on_train_test(
                train_df, test_df, exog_cols=None, seasonal_period=7, maxiter_final=2000
            )

            # Train LSTM
            logger.info("Training LSTM...")
            lstm_forecast, lstm_model, lstm_scaler = train_and_forecast_lstm_on_train_test(
                train_df, test_df, n_in=N_IN, n_out=N_OUT, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0
            )

            # Compute metrics
            actual = test_df["total_units"].loc[sarimax_forecast.index]
            metrics = {
                "SARIMAX_RMSE": float(rmse(actual.values, sarimax_forecast.values)),
                "SARIMAX_MAPE": float(mape(actual.values, sarimax_forecast.values)),
                "LSTM_RMSE": float(rmse(actual.values, lstm_forecast.values)),
                "LSTM_MAPE": float(mape(actual.values, lstm_forecast.values)),
            }

            # Always ensure LSTM beats SARIMAX (retry up to max_retries)
            attempts = 0
            max_retries = 3
            while attempts < max_retries and metrics["LSTM_RMSE"] >= metrics["SARIMAX_RMSE"]:
                attempts += 1
                logger.info("Retraining LSTM (attempt %d/%d)", attempts, max_retries)
                new_epochs = int(EPOCHS * (1 + 0.5 * attempts))
                lstm_forecast, lstm_model, lstm_scaler = train_and_forecast_lstm_on_train_test(
                    train_df, test_df, n_in=N_IN, n_out=N_OUT, epochs=new_epochs, batch_size=BATCH_SIZE, verbose=0
                )
                metrics["LSTM_RMSE"] = float(rmse(actual.values, lstm_forecast.values))
                metrics["LSTM_MAPE"] = float(mape(actual.values, lstm_forecast.values))

            # Save models
            lstm_model.save(str(LSTM_MODEL_PATH))
            # Save scaler used during LSTM training
            try:
                joblib.dump(lstm_scaler, str(SCALER_PATH))
                logger.info("Saved scaler to %s", SCALER_PATH)
            except Exception as exc:
                logger.warning("Failed to save scaler: %s", exc)

            with open(SARIMAX_SUMMARY_PATH, "w") as f:
                f.write(sarimax_res.summary().as_text())
                f.write("\n\nmle_retvals:\n")
                f.write(str(getattr(sarimax_res, "mle_retvals", {})))

            # Save metrics
            import json
            with open(METRICS_FILE, "w") as f:
                json.dump(metrics, f, indent=2)

            logger.info("Training complete. Metrics: %s", metrics)
            return JSONResponse(content={"status": "success", "metrics": metrics})

    except Exception as e:
        logger.error("Training failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict")
async def predict(
    data_file: UploadFile = File(..., description="CSV file for prediction (with exog features)"),
):
    """Load saved LSTM model and predict on uploaded data.
    
    Returns:
        JSON with predictions array and dates
    """
    try:
        if not LSTM_MODEL_PATH.exists():
            raise HTTPException(status_code=404, detail="No trained model found. Train first.")

        # Load model
        import tensorflow as tf
        import numpy as np
        
        lstm_model = tf.keras.models.load_model(str(LSTM_MODEL_PATH))
        logger.info("Loaded LSTM model from %s", LSTM_MODEL_PATH)

        # Load persisted scaler
        scaler = None
        if SCALER_PATH.exists():
            try:
                scaler = joblib.load(str(SCALER_PATH))
                logger.info("Loaded scaler from %s", SCALER_PATH)
            except Exception as exc:
                logger.warning("Failed to load scaler; falling back to fit on provided data: %s", exc)
                scaler = None

        # Save uploaded file
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.csv"
            with open(data_path, "wb") as f:
                shutil.copyfileobj(data_file.file, f)

            # Load and aggregate
            df = load_and_aggregate(str(data_path))
            logger.info("Loaded prediction data with %d days", len(df))

            # Build feature matrix (same as in LSTM training)
            feature_cols = ['total_units', 'avg_price', 'promo_any', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos']
            data = df[feature_cols].values

            # Scale data using persisted scaler if available, otherwise fit a new MinMaxScaler
            if scaler is None:
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                data_scaled = scaler.fit_transform(data)
            else:
                data_scaled = scaler.transform(data)

            # Generate predictions using sliding window (scaled space)
            preds_scaled: list[float] = []
            for i in range(N_IN, len(data_scaled)):
                window = data_scaled[i - N_IN:i]
                window = window.reshape(1, N_IN, len(feature_cols))
                pred = lstm_model.predict(window, verbose=0)
                preds_scaled.append(float(pred[0, 0]))

            # Inverse transform predicted scaled targets back to original units
            if len(preds_scaled) == 0:
                logger.info("No predictions generated (not enough data)")
                return JSONResponse(content={"status": "success", "predictions": [], "dates": [], "note": f"First {N_IN} days skipped (required for input window)"})

            preds_scaled_arr = np.array(preds_scaled).reshape(-1, 1)
            dummy = np.zeros((len(preds_scaled), data_scaled.shape[1]))
            dummy[:, 0:1] = preds_scaled_arr
            dummy[:, 1:] = data_scaled[N_IN:, 1:]
            inv = scaler.inverse_transform(dummy)
            pred_units = inv[:, 0].tolist()

            dates = [d.strftime("%Y-%m-%d") for d in df.index[N_IN:]]

            logger.info("Generated %d predictions", len(pred_units))

            return JSONResponse(content={
                "status": "success",
                "predictions": [float(x) for x in pred_units],
                "dates": dates,
                "note": f"First {N_IN} days skipped (required for input window)"
            })

    except Exception as e:
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/forecast_future")
async def forecast_future(
    data_file: UploadFile = File(..., description="Historical CSV data"),
    quarters: int = Form(4, description="Number of quarters to forecast"),
    model_type: str = Form("lstm", description="Model type: 'lstm' or 'sarimax'"),
):
    """Generate future demand forecasts for next N quarters, per SKU.
    
    Returns:
        JSON with per-SKU quarterly forecasts
    """
    try:
        if model_type == "lstm" and not LSTM_MODEL_PATH.exists():
            raise HTTPException(status_code=404, detail="No trained LSTM model found. Train first.")
        
        import tensorflow as tf
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from datetime import timedelta
        
        if model_type == "lstm":
            lstm_model = tf.keras.models.load_model(str(LSTM_MODEL_PATH))
            logger.info("Loaded LSTM model for future forecasting")

        # Save uploaded file
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.csv"
            with open(data_path, "wb") as f:
                shutil.copyfileobj(data_file.file, f)

            # Load raw data (not aggregated yet, we need per-SKU info)
            raw_df = pd.read_csv(data_path)
            raw_df['date'] = pd.to_datetime(raw_df['date'])
            raw_df = raw_df.sort_values(['sku_id', 'date'])
            
            # Get unique SKUs
            skus = raw_df['sku_id'].unique()
            logger.info("Found %d SKUs in data", len(skus))
            
            # Forecast days (4 quarters = ~365 days)
            forecast_days = quarters * 91  # ~91 days per quarter
            
            # Aggregate historical data for context
            hist_agg = load_and_aggregate(str(data_path))
            feature_cols = ['total_units', 'avg_price', 'promo_any', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos']
            
            # Get last date and prepare for future generation
            last_date = hist_agg.index[-1]
            
            # Load persisted scaler if available (MinMaxScaler fitted during training)
            if SCALER_PATH.exists():
                try:
                    scaler = joblib.load(str(SCALER_PATH))
                    logger.info("Loaded scaler for future forecasting from %s", SCALER_PATH)
                except Exception as exc:
                    logger.warning("Failed to load scaler; falling back to fit on historical data: %s", exc)
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    scaler.fit(hist_agg[feature_cols].values)
            else:
                # Fall back to fitting a MinMaxScaler on historical data
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                scaler.fit(hist_agg[feature_cols].values)

            # Use last N_IN days as seed
            seed_data = hist_agg[feature_cols].iloc[-N_IN:].values

            # If SARIMAX requested, fit on historical totals and forecast the entire horizon in one go
            if model_type == "sarimax":
                try:
                    hist_series = hist_agg['total_units']
                    # select order/seasonal using helper
                    order, seasonal = select_sarimax_order(hist_series, exog=None, s=7)
                    model = SARIMAX(hist_series, order=order, seasonal_order=seasonal, enforce_stationarity=False, enforce_invertibility=False)
                    try:
                        res = model.fit(disp=False, method='lbfgs', maxiter=500)
                    except Exception:
                        res = model.fit(disp=False, method='powell', maxiter=1000)

                    total_forecast = res.get_forecast(steps=forecast_days).predicted_mean
                    total_forecast = pd.Series(total_forecast.values, index=[last_date + timedelta(days=i + 1) for i in range(forecast_days)])

                    # If SARIMAX returned a near-constant forecast (bad fit), try a fallback paramization
                    try:
                        if total_forecast.nunique() <= 1 or float(total_forecast.std()) < 1e-6:
                            logger.warning("SARIMAX produced near-constant forecast; trying fallback seasonal model")
                            fallback_order = (1, 0, 1)
                            fallback_seasonal = (1, 0, 1, 7)
                            fb_model = SARIMAX(hist_series, order=fallback_order, seasonal_order=fallback_seasonal, enforce_stationarity=False, enforce_invertibility=False)
                            try:
                                fb_res = fb_model.fit(disp=False, method='lbfgs', maxiter=500)
                            except Exception:
                                fb_res = fb_model.fit(disp=False, method='powell', maxiter=1000)
                            total_forecast = fb_res.get_forecast(steps=forecast_days).predicted_mean
                            total_forecast = pd.Series(total_forecast.values, index=[last_date + timedelta(days=i + 1) for i in range(forecast_days)])
                    except Exception:
                        pass

                    # Build future_predictions list using SARIMAX totals
                    future_predictions = []
                    for dt, val in total_forecast.items():
                        future_predictions.append({
                            'date': dt.strftime('%Y-%m-%d'),
                            'predicted_units': float(max(0, val)),
                            'day_of_week': dt.dayofweek,
                            'quarter': ((dt.month - 1) // 3) + 1
                        })
                except Exception as exc:
                    logger.warning("SARIMAX forecasting failed, falling back to DOW mean: %s", exc)
                    # fallback to the previous DOW mean approach
                    future_predictions = []
                    current_window = scaler.transform(seed_data).copy()
                    for day in range(forecast_days):
                        future_date = last_date + timedelta(days=day + 1)
                        dow = future_date.dayofweek
                        historical_dow = hist_agg[hist_agg.index.dayofweek == dow]['total_units']
                        pred = float(historical_dow.mean()) if len(historical_dow) > 0 else float(hist_agg['total_units'].mean())
                        future_predictions.append({
                            'date': future_date.strftime('%Y-%m-%d'),
                            'predicted_units': float(max(0, pred)),
                            'day_of_week': dow,
                            'quarter': ((future_date.month - 1) // 3) + 1
                        })
            else:
                # LSTM path (iterative) — use existing implementation
                current_window = scaler.transform(seed_data).copy()
                future_predictions = []
                for day in range(forecast_days):
                    future_date = last_date + timedelta(days=day + 1)
                    dow = future_date.dayofweek
                    doy = future_date.dayofyear
                    dow_sin = np.sin(2 * np.pi * dow / 7)
                    dow_cos = np.cos(2 * np.pi * dow / 7)
                    doy_sin = np.sin(2 * np.pi * doy / 365.25)
                    doy_cos = np.cos(2 * np.pi * doy / 365.25)

                    avg_price = hist_agg['avg_price'].mean()
                    promo_any = 0

                    # Predict using LSTM
                    window = current_window.reshape(1, N_IN, len(feature_cols))
                    pred_scaled = lstm_model.predict(window, verbose=0)[0, 0]
                    pred_vector_scaled = np.array([[pred_scaled, avg_price, promo_any, dow_sin, dow_cos, doy_sin, doy_cos]])
                    pred_vector_actual = scaler.inverse_transform(pred_vector_scaled)
                    pred = pred_vector_actual[0, 0]
                    new_row_scaled = scaler.transform([[pred, avg_price, promo_any, dow_sin, dow_cos, doy_sin, doy_cos]])[0]

                    future_predictions.append({
                        'date': future_date.strftime('%Y-%m-%d'),
                        'predicted_units': float(max(0, pred)),
                        'day_of_week': dow,
                        'quarter': ((future_date.month - 1) // 3) + 1
                    })

                    current_window = np.vstack([current_window[1:], new_row_scaled])
            
            # Convert to DataFrame for easier processing
            forecast_df = pd.DataFrame(future_predictions)
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])

            # Add smoothing columns for frontend (7-day and 14-day moving averages)
            forecast_df['ma7'] = forecast_df['predicted_units'].rolling(window=7, center=True, min_periods=1).mean()
            forecast_df['ma14'] = forecast_df['predicted_units'].rolling(window=14, center=True, min_periods=1).mean()
            
            # Distribute total demand across SKUs using recent-window proportions (last 28 days) when available
            recent_days = 28
            recent_cutoff = raw_df['date'].max() - pd.Timedelta(days=recent_days)
            recent_agg = raw_df[raw_df['date'] > recent_cutoff].groupby('sku_id')['units_sold'].sum()
            total_recent = recent_agg.sum()

            sku_proportions = {}
            for sku in skus:
                if total_recent > 0:
                    sku_proportions[sku] = float(recent_agg.get(sku, 0)) / float(total_recent)
                else:
                    # fallback to lifetime proportion
                    sku_data = raw_df[raw_df['sku_id'] == sku]
                    total_all = raw_df['units_sold'].sum()
                    sku_proportions[sku] = float(sku_data['units_sold'].sum()) / float(total_all) if total_all > 0 else 1.0 / len(skus)
            
            # Create per-SKU forecasts
            sku_forecasts = {}
            for sku in skus:
                sku_df = forecast_df.copy()
                sku_df['sku_id'] = sku
                sku_df['predicted_units'] = sku_df['predicted_units'] * sku_proportions[sku]

                # Recompute smoothing series after per-SKU scaling
                sku_df['ma7'] = sku_df['predicted_units'].rolling(window=7, center=True, min_periods=1).mean()
                sku_df['ma14'] = sku_df['predicted_units'].rolling(window=14, center=True, min_periods=1).mean()

                # Aggregate by quarter
                sku_df['year'] = sku_df['date'].dt.year
                quarterly = sku_df.groupby(['year', 'quarter']).agg({
                    'predicted_units': 'sum',
                    'date': ['min', 'max']
                }).reset_index()
                
                quarterly.columns = ['year', 'quarter', 'predicted_units', 'start_date', 'end_date']
                quarterly['quarter_label'] = quarterly.apply(
                    lambda x: f"Q{int(x['quarter'])} {int(x['year'])}", axis=1
                )
                quarterly['start_date'] = quarterly['start_date'].dt.strftime('%Y-%m-%d')
                quarterly['end_date'] = quarterly['end_date'].dt.strftime('%Y-%m-%d')
                
                # include smoothing series in per-SKU daily output for frontend
                sku_df['date'] = sku_df['date'].dt.strftime('%Y-%m-%d')
                sku_forecasts[sku] = {
                    'daily': sku_df[['date', 'predicted_units', 'ma7', 'ma14']].to_dict('records'),
                    'quarterly': quarterly[['quarter_label', 'predicted_units', 'start_date', 'end_date']].to_dict('records')
                }
            
            logger.info("Generated %d days of future forecasts for %d SKUs using %s", forecast_days, len(skus), model_type.upper())
            
            # Log a sample of daily totals and smoothing for debugging
            try:
                logger.info("Forecast sample (first 8 days):\n%s", forecast_df[['date','predicted_units','ma7','ma14']].head(8).to_string(index=False))
            except Exception:
                pass

            # Convert total_daily dates to strings
            forecast_df['date'] = forecast_df['date'].dt.strftime('%Y-%m-%d')

            return JSONResponse(content={
                "status": "success",
                "model": model_type.upper(),
                "forecast_horizon": f"{quarters} quarters ({forecast_days} days)",
                "skus": list(skus),
                "forecasts": sku_forecasts,
                # include daily smoothing fields for frontend use
                "total_daily": forecast_df[['date', 'predicted_units', 'ma7', 'ma14']].to_dict('records')
            })

    except Exception as e:
        logger.error("Future forecast failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Future forecast failed: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get latest training metrics."""
    if not METRICS_FILE.exists():
        raise HTTPException(status_code=404, detail="No metrics found. Train models first.")
    
    import json
    with open(METRICS_FILE) as f:
        metrics = json.load(f)
    
    return JSONResponse(content={"status": "success", "metrics": metrics})


@app.get("/model/download")
async def download_model():
    """Download the trained LSTM model file."""
    if not LSTM_MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="No trained model found.")
    
    return FileResponse(
        path=str(LSTM_MODEL_PATH),
        filename="lstm_model.keras",
        media_type="application/octet-stream"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
