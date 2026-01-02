import logging
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv
import os
import threading
import uuid
import warnings
from typing import Dict
import json
from datetime import datetime, timedelta
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from core.forecast_pipeline import (
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
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tools.sm_exceptions import ConvergenceWarning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="E-commerce Demand Forecasting API", version="1.0.0")

def _normalize_origin(u: str) -> str:
    if not u:
        return u
    u = u.strip()
    if not u:
        return None
    if '://' not in u:
        u = 'https://' + u
    return u

_allowed = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8080",
]
_frontend_env = os.environ.get("FRONTEND_URL")
if _frontend_env:
    for part in _frontend_env.split(','):
        o = _normalize_origin(part)
        if o:
            _allowed.append(o)

logger.info("CORS allowed origins: %s", _allowed)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)
METRICS_FILE = MODEL_DIR / "latest_metrics.json"
MODEL_INFO_FILE = MODEL_DIR / "model_info.json" # For versioning
LSTM_MODEL_PATH = MODEL_DIR / "lstm_model.keras"
SCALER_PATH = MODEL_DIR / "scaler_minmax.pkl"
SARIMAX_MODEL_PATH = MODEL_DIR / "sarimax_model.pkl"
SARIMAX_SUMMARY_PATH = MODEL_DIR / "sarimax_summary.txt"


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "forecasting-api"}

TRAIN_JOBS: Dict[str, dict] = {}
TRAIN_JOBS_LOCK = threading.Lock()


def _run_training_job(train_contents: bytes, test_contents: bytes, seed: int, job_id: str):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.csv"
            test_path = Path(tmpdir) / "test.csv"
            with open(train_path, "wb") as f:
                f.write(train_contents)
            with open(test_path, "wb") as f:
                f.write(test_contents)

            with TRAIN_JOBS_LOCK:
                TRAIN_JOBS[job_id]['status'] = 'running'
            set_global_seed(seed)
            logger.info("Background training job %s started with seed=%d", job_id, seed)

            train_df = load_and_aggregate(str(train_path))
            test_df = load_and_aggregate(str(test_path))
        
        # Fix 17: Missing Data Validation
        if len(train_df) < 90:
            raise ValueError(f"Training data too short: {len(train_df)} days. Minimum 90 days required.")
        if len(test_df) < 7:
            raise ValueError(f"Test data too short: {len(test_df)} days. Minimum 7 days required.")
            
        logger.info("Loaded train (%d days) and test (%d days)", len(train_df), len(test_df))

        # --- HYBRID MODEL TRAINING ---
        
        # 1. Train SARIMAX (Linear Component)
        logger.info("Training SARIMAX (Linear Component)...")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            # We use the existing function but we need the fitted values for residuals
            # Disable exog for SARIMAX to simplify hybrid (let LSTM handle feature effects)
            sarimax_forecast_test, sarimax_res = train_and_forecast_sarimax_on_train_test(
                train_df, test_df, exog_cols=[], seasonal_period=7, maxiter_final=2000
            )

        # 2. Compute Residuals
        # Get in-sample fitted values
        sarimax_fitted = sarimax_res.fittedvalues
        # Ensure alignment
        common_idx = train_df.index.intersection(sarimax_fitted.index)
        residuals = train_df.loc[common_idx, "total_units"] - sarimax_fitted.loc[common_idx]
        
        # Add residuals to train_df for LSTM training
        train_df["residuals"] = residuals
        train_df["residuals"] = train_df["residuals"].fillna(0.0)

        # For test set, we don't have "true" residuals to predict, 
        # but we need to create the column structure for the LSTM function.
        # Let's compute test residuals based on the SARIMAX forecast we just made.
        sarimax_test_pred = sarimax_forecast_test
        common_test_idx = test_df.index.intersection(sarimax_test_pred.index)
        test_residuals = test_df.loc[common_test_idx, "total_units"] - sarimax_test_pred.loc[common_test_idx]
        test_df["residuals"] = test_residuals
        test_df["residuals"] = test_df["residuals"].fillna(0.0)

        # 3. Train LSTM on Residuals (Non-Linear Component)
        logger.info("Training LSTM on Residuals...")
        try:
            import tensorflow as _tf
            _tf.get_logger().setLevel('ERROR')
        except Exception:
            pass

        # We train LSTM to predict 'residuals' instead of 'total_units'
        lstm_forecast_residuals, lstm_model, lstm_scaler = train_and_forecast_lstm_on_train_test(
            train_df, test_df, n_in=N_IN, n_out=N_OUT, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
            target_col="residuals"
        )

        # 4. Combine Forecasts for Evaluation
        # Hybrid Forecast = SARIMAX + LSTM(Residuals)
        hybrid_forecast = sarimax_forecast_test + lstm_forecast_residuals

        actual = test_df["total_units"].loc[hybrid_forecast.index]
        metrics = {
            "Hybrid_RMSE": float(rmse(actual.values, hybrid_forecast.values)),
            "Hybrid_MAPE": float(mape(actual.values, hybrid_forecast.values)),
            # Also log components performance for debugging
            "SARIMAX_Only_RMSE": float(rmse(actual.values, sarimax_forecast_test.values)),
            "SARIMAX_Only_MAPE": float(mape(actual.values, sarimax_forecast_test.values)),
        }

        # --- STANDALONE LSTM ---
        logger.info("Training Standalone LSTM on total_units...")
        lstm_forecast_standalone, _, _ = train_and_forecast_lstm_on_train_test(
            train_df, test_df, n_in=N_IN, n_out=N_OUT, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
            target_col="total_units"
        )
        metrics["LSTM_Only_RMSE"] = float(rmse(actual.values, lstm_forecast_standalone.values))
        metrics["LSTM_Only_MAPE"] = float(mape(actual.values, lstm_forecast_standalone.values))
        
        # 5. Save Models
        lstm_model.save(str(LSTM_MODEL_PATH))
        try:
            joblib.dump(lstm_scaler, str(SCALER_PATH))
            logger.info("Saved LSTM scaler to %s", SCALER_PATH)
        except Exception as exc:
            logger.warning("Failed to save scaler: %s", exc)

        try:
            sarimax_res.save(str(SARIMAX_MODEL_PATH))
            logger.info("Saved SARIMAX model to %s", SARIMAX_MODEL_PATH)
        except Exception as exc:
            logger.warning("Failed to save SARIMAX model: %s", exc)

        with open(SARIMAX_SUMMARY_PATH, "w") as f:
            f.write(sarimax_res.summary().as_text())
            f.write("\n\nmle_retvals:\n")
            f.write(str(getattr(sarimax_res, "mle_retvals", {})))
        
        # Fix 16: Model Versioning
        model_info = {
            "timestamp": datetime.now().isoformat(),
            "job_id": job_id,
            "seed": seed,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "metrics": metrics
        }
        with open(MODEL_INFO_FILE, "w") as f:
            json.dump(model_info, f, indent=2)

        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)

        with TRAIN_JOBS_LOCK:
            TRAIN_JOBS[job_id]['status'] = 'completed'
            TRAIN_JOBS[job_id]['metrics'] = metrics
        logger.info("Background training job %s complete. Hybrid Metrics: %s", job_id, metrics)
        
    # Fix 15: Specific Exception Handling
    except ValueError as e:
        logger.error("Validation Error in job %s: %s", job_id, str(e), exc_info=True)
        with TRAIN_JOBS_LOCK:
            TRAIN_JOBS[job_id]['status'] = 'failed'
            TRAIN_JOBS[job_id]['error'] = f"Validation Error: {str(e)}"
    except MemoryError:
        logger.error("Memory Error in job %s. Data might be too large.", job_id)
        with TRAIN_JOBS_LOCK:
            TRAIN_JOBS[job_id]['status'] = 'failed'
            TRAIN_JOBS[job_id]['error'] = "System ran out of memory. Try a smaller dataset."
    except Exception as e:
        logger.error("Background training job %s failed: %s", job_id, str(e), exc_info=True)
        with TRAIN_JOBS_LOCK:
            TRAIN_JOBS[job_id]['status'] = 'failed'
            TRAIN_JOBS[job_id]['error'] = str(e)


@app.post('/train')
async def train_models(
    train_file: UploadFile = File(..., description='Training CSV file'),
    test_file: UploadFile = File(..., description='Test CSV file'),
    seed: int = Form(42),
    async_mode: int = Form(1),
    background_tasks: BackgroundTasks = None
):
    try:
        # Read the file contents here, before the request is closed
        train_contents = await train_file.read()
        test_contents = await test_file.read()

        if async_mode:
            job_id = str(uuid.uuid4())
            with TRAIN_JOBS_LOCK:
                TRAIN_JOBS[job_id] = {'status': 'queued', 'metrics': None, 'error': None}
            t = threading.Thread(target=_run_training_job, args=(train_contents, test_contents, seed, job_id), daemon=True)
            t.start()
            return JSONResponse(status_code=202, content={'status': 'queued', 'job_id': job_id, 'status_url': f'/api/train/status/{job_id}'})
        else:
            sync_id = 'sync'
            with TRAIN_JOBS_LOCK:
                TRAIN_JOBS[sync_id] = {'status': 'running', 'metrics': None, 'error': None}
            _run_training_job(train_contents, test_contents, seed, sync_id)
            metrics = TRAIN_JOBS.get(sync_id, {}).get('metrics')

            with TRAIN_JOBS_LOCK:
                TRAIN_JOBS.pop(sync_id, None)
            return JSONResponse(content={'status': 'success', 'metrics': metrics})
    except Exception as e:
        logger.error('Training failed to start: %s', str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f'Training failed to start: {str(e)}')


@app.get('/train/status/{job_id}')
async def train_status(job_id: str):
    job = TRAIN_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return JSONResponse({'job_id': job_id, **job})


@app.post("/predict")
async def predict(
    data_file: UploadFile = File(..., description="CSV file for prediction (with exog features)"),
):
    try:
        if not LSTM_MODEL_PATH.exists() or not SARIMAX_MODEL_PATH.exists():
            raise HTTPException(status_code=404, detail="No trained hybrid model found. Train first.")

        import tensorflow as tf
        
        # Load models
        lstm_model = tf.keras.models.load_model(str(LSTM_MODEL_PATH))
        sarimax_res = SARIMAXResults.load(str(SARIMAX_MODEL_PATH))
        
        scaler = None
        if SCALER_PATH.exists():
            scaler = joblib.load(str(SCALER_PATH))
        else:
            raise HTTPException(status_code=500, detail="Scaler file missing.")

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.csv"
            with open(data_path, "wb") as f:
                shutil.copyfileobj(data_file.file, f)

            df = load_and_aggregate(str(data_path))
            logger.info("Loaded prediction data with %d days", len(df))
            
            # 1. SARIMAX Prediction (Linear)
            try:
                sarimax_new = sarimax_res.apply(df["total_units"])
                sarimax_pred = sarimax_new.fittedvalues
            except Exception as e:
                logger.warning("SARIMAX apply failed, falling back to simple predict: %s", e)
                sarimax_pred = pd.Series(0, index=df.index)

            # 2. LSTM Prediction (Residuals)
            feature_cols = ['residuals', 'avg_price', 'promo_any', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos']
            
            residuals = df["total_units"] - sarimax_pred
            df["residuals"] = residuals.fillna(0.0)

            data = df[feature_cols].values

            # Fix 9: Validate Scaler Input (Basic range check)
            # Check if any feature is drastically out of training range
            # This is a simple heuristic check
            try:
                data_scaled = scaler.transform(data)
            except ValueError as ve:
                logger.error(f"Scaler transform failed: {ve}")
                raise HTTPException(status_code=400, detail="Input data contains features incompatible with trained model (e.g. range mismatch).")

            preds_scaled: list[float] = []
            for i in range(N_IN, len(data_scaled)):
                window = data_scaled[i - N_IN:i]
                window = window.reshape(1, N_IN, len(feature_cols))
                pred = lstm_model.predict(window, verbose=0)
                preds_scaled.append(float(pred[0, 0]))

            if len(preds_scaled) == 0:
                return JSONResponse(content={"status": "success", "predictions": [], "dates": [], "note": "Not enough data"})

            # Inverse transform
            preds_scaled_arr = np.array(preds_scaled).reshape(-1, 1)
            dummy = np.zeros((len(preds_scaled), data_scaled.shape[1]))
            dummy[:, 0:1] = preds_scaled_arr
            dummy[:, 1:] = data_scaled[N_IN:, 1:]
            inv = scaler.inverse_transform(dummy)
            pred_residuals = inv[:, 0]
            
            # Combine
            valid_idx = df.index[N_IN:]
            sarimax_part = sarimax_pred.loc[valid_idx]
            final_pred = sarimax_part + pred_residuals
            
            dates = [d.strftime("%Y-%m-%d") for d in valid_idx]
            rounded_preds = [int(round(max(0, x))) for x in final_pred]

            return JSONResponse(content={
                "status": "success",
                "predictions": rounded_preds,
                "dates": dates,
                "note": f"Hybrid model prediction. First {N_IN} days skipped."
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/forecast_future")
async def forecast_future(
    data_file: UploadFile = File(..., description="Historical CSV data"),
    quarters: int = Form(4, description="Number of quarters to forecast"),
    model_type: str = Form("hybrid", description="Legacy parameter, ignored. Always uses hybrid."),
):

    try:
        if not LSTM_MODEL_PATH.exists() or not SARIMAX_MODEL_PATH.exists():
            raise HTTPException(status_code=404, detail="No trained hybrid model found. Train first.")
        
        import tensorflow as tf
        
        # Load Models
        lstm_model = tf.keras.models.load_model(str(LSTM_MODEL_PATH))
        sarimax_res = SARIMAXResults.load(str(SARIMAX_MODEL_PATH))
        scaler = joblib.load(str(SCALER_PATH))
        
        logger.info("Loaded Hybrid models (SARIMAX + LSTM)")

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.csv"
            with open(data_path, "wb") as f:
                shutil.copyfileobj(data_file.file, f)

            # Preprocess Data
            raw_df = pd.read_csv(data_path, low_memory=False)
            raw_df.columns = raw_df.columns.str.strip().str.lower()
            
            column_mapping = {
                'date': 'date', 'ds': 'date', 'datetime': 'date', 'time': 'date',
                'sku_id': 'sku_id', 'sku': 'sku_id', 'product_id': 'sku_id', 'item_id': 'sku_id', 'product': 'sku_id',
                'units_sold': 'units_sold', 'units': 'units_sold', 'quantity': 'units_sold', 'qty': 'units_sold', 
                'sales': 'units_sold', 'demand': 'units_sold', 'y': 'units_sold',
                'price': 'price', 'unit_price': 'price', 'avg_price': 'price', 'cost': 'price',
                'promotion_flag': 'promotion_flag', 'promo_flag': 'promotion_flag', 'promo': 'promotion_flag',
                'promotion': 'promotion_flag', 'is_promo': 'promotion_flag', 'on_promotion': 'promotion_flag',
            }
            new_columns = {col: column_mapping[col] for col in raw_df.columns if col in column_mapping}
            raw_df = raw_df.rename(columns=new_columns)
            
            # Validation
            if 'date' not in raw_df.columns or 'units_sold' not in raw_df.columns:
                raise ValueError(f"Missing required columns. Found: {list(raw_df.columns)}")
            
            # Defaults
            if 'sku_id' not in raw_df.columns: raw_df['sku_id'] = 'DEFAULT_SKU'
            if 'price' not in raw_df.columns: raw_df['price'] = 100.0
            if 'promotion_flag' not in raw_df.columns: raw_df['promotion_flag'] = 0
            
            raw_df['date'] = pd.to_datetime(raw_df['date'])
            skus = raw_df['sku_id'].unique()
            forecast_days = quarters * 91
            
            # Aggregate historical data provided by the user
            hist_agg = load_and_aggregate(str(data_path))
            
            # --- Bridge the Gap from Historical Data to Present using Year-over-Year Data ---
            forecast_start_date = pd.Timestamp.now().normalize()
            last_hist_date = hist_agg.index[-1]
            
            # Fix 1 & 4: Gap Filling Validation and Quality Check
            if last_hist_date < forecast_start_date - pd.Timedelta(days=1):
                logger.info(f"Gap detected from {last_hist_date.date()} to {forecast_start_date.date()}. Attempting to bridge with previous year's data...")
                
                gap_dates = pd.date_range(start=last_hist_date + pd.Timedelta(days=1), end=forecast_start_date - pd.Timedelta(days=1), freq='D')
                
                if not gap_dates.empty:
                    source_dates = gap_dates - pd.DateOffset(years=1)
                    
                    # Extract data using nearest-neighbor
                    source_data = hist_agg.reindex(source_dates, method='nearest')
                    
                    # Fix 4: Validate Gap Data
                    missing_ratio = source_data['total_units'].isna().sum() / len(source_data)
                    if missing_ratio > 0.2:
                        logger.warning(f"Gap filling source data has >20% missing values. Forecast may be unreliable.")
                    
                    if not source_data.empty and not source_data['total_units'].isnull().all():
                        bridge_df = source_data.copy()
                        bridge_df.index = gap_dates 
                        hist_agg = pd.concat([hist_agg, bridge_df])
                        logger.info(f"Successfully bridged gap using {len(bridge_df)} days of data from the previous year.")
                    else:
                        logger.warning("Could not find sufficient data in the previous year to bridge the gap. Forecasting may be less accurate.")

            last_date = hist_agg.index[-1]
            
            # --- Adjust forecast start to avoid partial month issues ---
            # Fix 1: Visual Cliff Handling
            today = pd.Timestamp.now().normalize()
            if today.day < 28:  
                next_month_start = (today + pd.offsets.MonthBegin(1)).normalize()
                days_to_skip = (next_month_start - last_date).days - 1
                if days_to_skip > 0 and days_to_skip < forecast_days:
                    logger.info(f"Skipping {days_to_skip} days to start forecast from complete month: {next_month_start.date()}")
                    last_date = next_month_start - pd.Timedelta(days=1)
                    forecast_days = forecast_days - days_to_skip

            # --- HYBRID FORECASTING for the FUTURE ---
            
            # 1. SARIMAX Future Forecast
            sarimax_ext = sarimax_res.apply(hist_agg['total_units'])
            sarimax_forecast = sarimax_ext.get_forecast(steps=forecast_days).predicted_mean
            sarimax_forecast_series = pd.Series(sarimax_forecast.values, index=[last_date + timedelta(days=i + 1) for i in range(forecast_days)])
            
            # 2. LSTM Residual Forecast
            fitted_values = sarimax_ext.fittedvalues
            residuals = hist_agg['total_units'] - fitted_values
            hist_agg['residuals'] = residuals.fillna(0.0)
            
            feature_cols = ['residuals', 'avg_price', 'promo_any', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos']
            seed_data = hist_agg.tail(N_IN)
            
            residual_forecast_df = predict_future_with_lstm(
                lstm_model, scaler, seed_data, forecast_days, N_IN, feature_cols
            )
            
            # 3. Combine
            future_dates = sarimax_forecast_series.index
            if not residual_forecast_df.empty:
                residual_forecast_values = residual_forecast_df['predicted_units'].values
            else:
                residual_forecast_values = np.zeros(forecast_days)
                
            min_len = min(len(sarimax_forecast_series), len(residual_forecast_values))
            total_forecast_values = sarimax_forecast_series.values[:min_len] + residual_forecast_values[:min_len]
            
            forecast_df = pd.DataFrame({
                'predicted_units': total_forecast_values,
                'date': future_dates[:min_len]
            })
            
            # --- Post-Processing ---
            
            if forecast_df.empty:
                logger.warning("No forecast generated")
                return JSONResponse(content={"status": "error", "message": "No forecast generated"})

            forecast_df['year'] = forecast_df['date'].dt.year
            forecast_df['quarter'] = forecast_df['date'].dt.quarter
            forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek
            
            # Distribute to SKUs
            recent_days = 28
            recent_cutoff = raw_df['date'].max() - pd.Timedelta(days=recent_days)
            recent_agg = raw_df[raw_df['date'] > recent_cutoff].groupby('sku_id')['units_sold'].sum()
            total_recent = recent_agg.sum()

            sku_proportions = {}
            for sku in skus:
                if total_recent > 0:
                    sku_proportions[sku] = float(recent_agg.get(sku, 0)) / float(total_recent)
                else:
                    sku_data = raw_df[raw_df['sku_id'] == sku]
                    total_all = raw_df['units_sold'].sum()
                    sku_proportions[sku] = float(sku_data['units_sold'].sum()) / float(total_all) if total_all > 0 else 1.0 / len(skus)
            
            sku_forecasts = {}
            for sku in skus:
                sku_df = forecast_df.copy()
                sku_df['sku_id'] = sku
                sku_df['predicted_units'] = sku_df['predicted_units'] * sku_proportions.get(sku, 0)
                
                sku_df['ma7'] = sku_df['predicted_units'].rolling(window=7, center=True, min_periods=1).mean()
                sku_df['ma14'] = sku_df['predicted_units'].rolling(window=14, center=True, min_periods=1).mean()
                
                quarterly = sku_df.groupby(['year', 'quarter']).agg({
                    'predicted_units': 'sum',
                    'date': ['min', 'max']
                }).reset_index()
                quarterly.columns = ['year', 'quarter', 'predicted_units', 'start_date', 'end_date']
                quarterly['quarter_label'] = quarterly.apply(lambda x: f"Q{int(x['quarter'])} {int(x['year'])}", axis=1)
                quarterly['start_date'] = quarterly['start_date'].dt.strftime('%Y-%m-%d')
                quarterly['end_date'] = quarterly['end_date'].dt.strftime('%Y-%m-%d')
                quarterly['predicted_units'] = quarterly['predicted_units'].round().astype(int)
                
                sku_df['predicted_units'] = sku_df['predicted_units'].round().astype(int)
                sku_df['ma7'] = sku_df['ma7'].round(1)
                sku_df['ma14'] = sku_df['ma14'].round(1)
                sku_df['date'] = sku_df['date'].dt.strftime('%Y-%m-%d')
                
                daily_records = sku_df[['date', 'predicted_units', 'ma7', 'ma14']].to_dict('records')
                sku_forecasts[sku] = {
                    'daily': downsample_for_chart(daily_records),
                    'daily_count': len(daily_records),
                    'quarterly': quarterly[['quarter_label', 'predicted_units', 'start_date', 'end_date']].to_dict('records')
                }

            # Total Daily
            forecast_df['predicted_units'] = forecast_df['predicted_units'].round().astype(int)
            # Removed center=True because future forecasts don't have "future" data to center on
            forecast_df['ma7'] = forecast_df['predicted_units'].rolling(window=7, min_periods=1).mean().round(1)
            forecast_df['ma14'] = forecast_df['predicted_units'].rolling(window=14, min_periods=1).mean().round(1)
            forecast_df['date'] = forecast_df['date'].dt.strftime('%Y-%m-%d')
            
            total_daily_records = forecast_df[['date', 'predicted_units', 'ma7', 'ma14']].to_dict('records')

            return JSONResponse(content={
                "status": "success",
                "model": "HYBRID (SARIMAX + LSTM)",
                "forecast_horizon": f"{quarters} quarters ({forecast_days} days)",
                "skus": list(skus),
                "forecasts": sku_forecasts,
                "total_daily": downsample_for_chart(total_daily_records),
                "total_daily_count": len(total_daily_records)
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Future forecast failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Future forecast failed: {str(e)}")


@app.get("/metrics")
async def get_metrics():

    if not METRICS_FILE.exists():
        raise HTTPException(status_code=404, detail="No metrics found. Train models first.")
    
    with open(METRICS_FILE) as f:
        metrics = json.load(f)
    
    return JSONResponse(content={"status": "success", "metrics": metrics})


@app.get("/model/download")
async def download_model():

    if not LSTM_MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="No trained model found.")
    
    return FileResponse(
        path=str(LSTM_MODEL_PATH),
        filename="lstm_model.keras",
        media_type="application/octet-stream"
    )

def downsample_for_chart(data: list, max_points: int = 500) -> list:
    """
    Fix 18: Improved downsampling to preserve peaks.
    Uses Max aggregation for predicted_units to prevent smoothing out critical demand spikes.
    """
    if len(data) <= max_points:
        return data
    
    step = len(data) / max_points
    result = []
    
    for i in range(max_points):
        start_idx = int(i * step)
        end_idx = int((i + 1) * step)
        if end_idx > len(data):
            end_idx = len(data)
        
        chunk = data[start_idx:end_idx]
        if chunk:
            # Use the middle point's date, but MAX for values to preserve peaks
            mid_idx = len(chunk) // 2
            point = chunk[mid_idx].copy()
            if len(chunk) > 1:
                # Take MAX to preserve spikes (important for demand planning)
                point['predicted_units'] = int(max(c['predicted_units'] for c in chunk))
                
                # For MA, average is still acceptable to show trend
                if 'ma7' in point:
                    ma7_vals = [c['ma7'] for c in chunk if c.get('ma7') is not None]
                    point['ma7'] = round(sum(ma7_vals) / len(ma7_vals), 1) if ma7_vals else None
                if 'ma14' in point:
                    ma14_vals = [c['ma14'] for c in chunk if c.get('ma14') is not None]
                    point['ma14'] = round(sum(ma14_vals) / len(ma14_vals), 1) if ma14_vals else None
            result.append(point)
    
    return result


def predict_future_with_lstm(
    model,
    scaler,
    seed_data: pd.DataFrame,
    forecast_days: int,
    n_in: int,
    feature_cols: list,
):
    
    from datetime import timedelta

    logger.info("predict_future_with_lstm: seed_data rows=%d, n_in=%d, forecast_days=%d", len(seed_data), n_in, forecast_days)

    if len(seed_data) < n_in:
        logger.info("No predictions generated (not enough seed history: need %d, got %d)", n_in, len(seed_data))
        return pd.DataFrame(columns=['date', 'predicted_units'])

    last_date = seed_data.index[-1]

    try:
        current_window = scaler.transform(seed_data[feature_cols]).copy()
    except Exception as exc:
        logger.error("Failed to transform seed_data with scaler: %s", exc, exc_info=True)
        return pd.DataFrame(columns=['date', 'predicted_units'])

    # Fix 20: Pre-allocate arrays for performance
    # Fix 12: Use better assumptions for future exogenous variables
    # Instead of global mean, use the last observed value or a moving average of the seed
    last_observed_price = float(seed_data['avg_price'].iloc[-1])
    last_observed_promo_freq = float(seed_data['promo_any'].mean()) # Frequency of promo in seed window

    future_predictions = []

    # Pre-allocate buffer for scaled features to avoid vstack in loop
    # We only need to slide one step at a time
    buffer_shape = (forecast_days, len(feature_cols))
    
    for day in range(forecast_days):
        try:
            window_reshaped = current_window.reshape(1, n_in, len(feature_cols))
            pred_scaled = float(model.predict(window_reshaped, verbose=0)[0, 0])
        except Exception as exc:
            logger.error("Model prediction failed on day %d: %s", day, exc, exc_info=True)
            break

        future_date = last_date + timedelta(days=day + 1)
        dow = int(future_date.dayofweek)
        doy = int(future_date.dayofyear)

        # Fix 12: Future Assumption Logic
        # Use last observed price instead of global mean to better reflect recent context
        avg_price = last_observed_price
        # Use observed frequency, but cap at 1.0
        promo_any = 1.0 if last_observed_promo_freq > 0.5 else 0.0

        raw_row = [
            0.0, # Placeholder for target (residuals)
            avg_price,
            promo_any,
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
        ]

        try:
            # scaler expects 2D array
            scaled_row = scaler.transform([raw_row])[0]
        except Exception as exc:
            logger.error("Failed to transform generated raw_row on day %d: %s", day, exc, exc_info=True)
            break

        scaled_row[0] = pred_scaled

        dummy = np.zeros((1, len(feature_cols)))
        dummy[0, 0] = pred_scaled
        try:
            pred_inversed = scaler.inverse_transform(dummy)[0, 0]
        except Exception as exc:
            logger.error("Failed to inverse transform prediction on day %d: %s", day, exc, exc_info=True)
            break

        future_predictions.append((future_date, pred_inversed))

        # Fix 20: Efficient Window Update (roll the window)
        # Drop first row, append new row
        current_window = np.roll(current_window, -1, axis=0)
        current_window[-1, :] = scaled_row

    if not future_predictions:
        logger.info("No future predictions were generated after attempting model forecasts")
        return pd.DataFrame(columns=['date', 'predicted_units'])

    dates = [d for d, _ in future_predictions]
    preds = [p for _, p in future_predictions]

    logger.info("Generated %d future predictions", len(preds))

    return pd.DataFrame({'predicted_units': preds}, index=dates)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")