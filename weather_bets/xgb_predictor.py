"""XGBoost Intraday Temperature Predictor.

Uses trajectory features (rate of climb, morning low, hourly temps) + seasonal
features (day of year, daylight hours) to predict the daily high earlier than
the simple lookup model.

Trains 3 quantile models (p10, p50, p90) for confidence intervals.
"""

from __future__ import annotations

import csv
import logging
import math
import pickle
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb

from weather_bets.iem_fetcher import (
    DAILY_SUMMARY_FILE,
    HISTORY_FILE,
    compute_daylight_hours,
    load_daily_summary,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "data" / "models"

FEATURE_NAMES = [
    "current_temp",
    "morning_low",
    "total_rise_so_far",
    "rise_rate_1h",
    "rise_rate_2h",
    "rise_rate_3h",
    "temp_at_8am",
    "temp_at_10am",
    "current_hour",
    "day_of_year",
    "daylight_hours",
    "pct_daylight_elapsed",
    "hours_sun_remaining",
    "sky_cover_enc",
    "wind_speed_kt",
]

SKY_ENCODE = {"CLR": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4}


def _sunrise_hour(day_of_year: int, lat: float = 30.1975) -> float:
    """Approximate sunrise hour (CDT) for Austin."""
    declination = 23.45 * math.sin(math.radians(360 / 365 * (day_of_year - 81)))
    dec_rad = math.radians(declination)
    lat_rad = math.radians(lat)
    cos_ha = -math.tan(lat_rad) * math.tan(dec_rad)
    cos_ha = max(-1.0, min(1.0, cos_ha))
    ha = math.degrees(math.acos(cos_ha))
    # Solar noon in Austin CDT is roughly 13:30 (1:30 PM)
    sunrise = 13.5 - ha / 15.0
    return sunrise


def _sunset_hour(day_of_year: int, lat: float = 30.1975) -> float:
    """Approximate sunset hour (CDT) for Austin."""
    declination = 23.45 * math.sin(math.radians(360 / 365 * (day_of_year - 81)))
    dec_rad = math.radians(declination)
    lat_rad = math.radians(lat)
    cos_ha = -math.tan(lat_rad) * math.tan(dec_rad)
    cos_ha = max(-1.0, min(1.0, cos_ha))
    ha = math.degrees(math.acos(cos_ha))
    sunset = 13.5 + ha / 15.0
    return sunset


def build_training_data(
    hourly_path: Path = HISTORY_FILE,
    daily_path: Path = DAILY_SUMMARY_FILE,
    prediction_hours: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Build feature matrix and target vector from historical data.

    For each day, creates one training row per prediction_hour.
    Returns (X, y, metadata) where metadata contains date/hour for analysis.
    """
    if prediction_hours is None:
        prediction_hours = [10, 11, 12, 13, 14, 15, 16]

    # Load daily summaries
    daily_data = load_daily_summary(daily_path)
    daily_by_date = {d["date"]: d for d in daily_data}

    # Load hourly data, indexed by date -> hour -> readings
    hourly_by_date: dict[str, dict[int, dict]] = defaultdict(dict)
    with open(hourly_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row["date"]
            h = int(row["hour"])
            hourly_by_date[d][h] = {
                "temp_f": int(float(row["temp_f"])),
                "wind_kt": float(row["wind_kt"]),
                "sky_cover": row["sky_cover"],
                "daylight_hours": float(row["daylight_hours"]),
            }

    X_rows = []
    y_rows = []
    meta = []

    for date_str, day_info in sorted(daily_by_date.items()):
        hours = hourly_by_date.get(date_str, {})
        if len(hours) < 8:  # Need enough hourly data
            continue

        daily_high = day_info["daily_high"]
        dt = date.fromisoformat(date_str)
        doy = dt.timetuple().tm_yday
        daylight = day_info["daylight_hours"]
        sunrise = _sunrise_hour(doy)
        sunset = _sunset_hour(doy)

        # Find morning low (min temp from midnight to 10 AM)
        morning_temps = [hours[h]["temp_f"] for h in range(0, 11) if h in hours]
        if not morning_temps:
            continue
        morning_low = min(morning_temps)

        for pred_hour in prediction_hours:
            if pred_hour not in hours:
                continue

            reading = hours[pred_hour]
            current_temp = reading["temp_f"]

            # Rate of change features
            rise_1h = 0.0
            rise_2h = 0.0
            rise_3h = 0.0
            if pred_hour - 1 in hours:
                rise_1h = current_temp - hours[pred_hour - 1]["temp_f"]
            if pred_hour - 2 in hours:
                rise_2h = (current_temp - hours[pred_hour - 2]["temp_f"]) / 2.0
            if pred_hour - 3 in hours:
                rise_3h = (current_temp - hours[pred_hour - 3]["temp_f"]) / 3.0

            # Reference temps
            temp_8am = hours[8]["temp_f"] if 8 in hours else current_temp
            temp_10am = hours[10]["temp_f"] if 10 in hours else current_temp

            # Daylight position
            total_sun = sunset - sunrise
            elapsed_sun = max(0, pred_hour - sunrise)
            pct_elapsed = min(1.0, elapsed_sun / total_sun) if total_sun > 0 else 0.5
            hours_remaining = max(0, sunset - pred_hour)

            # Sky cover encoding
            sky_enc = SKY_ENCODE.get(reading["sky_cover"], 2)

            features = [
                current_temp,
                morning_low,
                current_temp - morning_low,      # total_rise_so_far
                rise_1h,
                rise_2h,
                rise_3h,
                temp_8am,
                temp_10am,
                pred_hour,
                doy,
                daylight,
                pct_elapsed,
                hours_remaining,
                sky_enc,
                reading["wind_kt"],
            ]

            X_rows.append(features)
            y_rows.append(daily_high)
            meta.append({"date": date_str, "hour": pred_hour, "month": dt.month})

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)

    logger.info(f"[XGB] Built training data: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, meta


class XGBoostPredictor:
    """
    XGBoost-based intraday temperature predictor.
    
    Trains 3 models:
    - p10 (10th percentile) — lower bound of CI
    - p50 (median) — best estimate
    - p90 (90th percentile) — upper bound of CI
    """

    def __init__(self):
        self.model_p10: xgb.XGBRegressor | None = None
        self.model_p50: xgb.XGBRegressor | None = None
        self.model_p90: xgb.XGBRegressor | None = None
        self._trained = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict:
        """Train the 3 quantile models."""
        common_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "random_state": 42,
            "verbosity": 0,
        }

        # Median model (main prediction)
        self.model_p50 = xgb.XGBRegressor(
            objective="reg:squarederror",
            **common_params,
        )
        self.model_p50.fit(X_train, y_train)

        # Lower bound (10th percentile)
        self.model_p10 = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.10,
            **common_params,
        )
        self.model_p10.fit(X_train, y_train)

        # Upper bound (90th percentile)
        self.model_p90 = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.90,
            **common_params,
        )
        self.model_p90.fit(X_train, y_train)

        self._trained = True

        # Training metrics
        train_pred = self.model_p50.predict(X_train)
        train_mae = np.mean(np.abs(train_pred - y_train))

        result = {"train_mae": round(float(train_mae), 2), "train_n": len(y_train)}

        if X_val is not None and y_val is not None:
            val_pred = self.model_p50.predict(X_val)
            val_mae = np.mean(np.abs(val_pred - y_val))
            result["val_mae"] = round(float(val_mae), 2)
            result["val_n"] = len(y_val)

        # Feature importance
        importance = self.model_p50.feature_importances_
        feat_imp = sorted(
            zip(FEATURE_NAMES, importance),
            key=lambda x: x[1],
            reverse=True,
        )
        result["feature_importance"] = feat_imp

        logger.info(f"[XGB] Training complete: train MAE={result['train_mae']}°F")
        if "val_mae" in result:
            logger.info(f"[XGB] Validation MAE={result['val_mae']}°F")

        return result

    def predict(self, features: dict) -> dict:
        """
        Predict daily high from current conditions.

        Args:
            features: dict with keys matching FEATURE_NAMES

        Returns:
            predicted_high, ci_80 (p10-p90), confidence info
        """
        if not self._trained:
            raise RuntimeError("Model not trained yet")

        x = np.array([[
            features["current_temp"],
            features["morning_low"],
            features["current_temp"] - features["morning_low"],
            features.get("rise_rate_1h", 0),
            features.get("rise_rate_2h", 0),
            features.get("rise_rate_3h", 0),
            features.get("temp_at_8am", features["current_temp"]),
            features.get("temp_at_10am", features["current_temp"]),
            features["current_hour"],
            features["day_of_year"],
            features["daylight_hours"],
            features.get("pct_daylight_elapsed", 0.5),
            features.get("hours_sun_remaining", 4.0),
            SKY_ENCODE.get(features.get("sky_cover", "CLR"), 0),
            features.get("wind_speed_kt", 5.0),
        ]], dtype=np.float32)

        pred_high = round(float(self.model_p50.predict(x)[0]))
        pred_low = int(float(self.model_p10.predict(x)[0]))
        pred_upper = int(float(self.model_p90.predict(x)[0]))

        ci_width = pred_upper - pred_low

        return {
            "predicted_high": pred_high,
            "confidence_80": (pred_low, pred_upper),
            "ci_width": ci_width,
        }

    def save(self, directory: Path = MODEL_DIR) -> None:
        """Save trained models to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        self.model_p10.save_model(str(directory / "xgb_p10.json"))
        self.model_p50.save_model(str(directory / "xgb_p50.json"))
        self.model_p90.save_model(str(directory / "xgb_p90.json"))
        logger.info(f"[XGB] Models saved to {directory}")

    def load(self, directory: Path = MODEL_DIR) -> None:
        """Load trained models from disk."""
        self.model_p10 = xgb.XGBRegressor()
        self.model_p10.load_model(str(directory / "xgb_p10.json"))
        self.model_p50 = xgb.XGBRegressor()
        self.model_p50.load_model(str(directory / "xgb_p50.json"))
        self.model_p90 = xgb.XGBRegressor()
        self.model_p90.load_model(str(directory / "xgb_p90.json"))
        self._trained = True
        logger.info(f"[XGB] Models loaded from {directory}")
