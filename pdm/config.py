"""Central configuration: paths, sensor definitions and thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
RAW_DATA_PATH = DATA_DIR / "vehicle_sensor_data.csv"

SEED = 42

# The nine input signals from the project brief.
SENSORS = [
    "engine_temp",        # deg C
    "rpm",                # rev/min
    "oil_pressure",       # psi
    "vibration",          # mm/s RMS
    "battery_voltage",    # V
    "coolant_temp",       # deg C
    "fuel_consumption",   # L/h
    "vehicle_speed",      # km/h
    "operating_hours",    # h (cumulative)
]

# Human-readable names used in reports and alerts.
SENSOR_LABELS = {
    "engine_temp": "engine temperature",
    "rpm": "RPM",
    "oil_pressure": "oil pressure",
    "vibration": "vibration",
    "battery_voltage": "battery voltage",
    "coolant_temp": "coolant temperature",
    "fuel_consumption": "fuel consumption",
    "vehicle_speed": "vehicle speed",
    "operating_hours": "operating hours",
}

# Recommended maintenance action per sensor, used for alerts.
MAINTENANCE_ACTIONS = {
    "engine_temp": "Check cooling system, thermostat and engine load",
    "rpm": "Inspect idle control, throttle body and transmission",
    "oil_pressure": "Check oil level, oil pump and filter; look for leaks",
    "vibration": "Inspect wheel/engine bearings, mounts and wheel balance",
    "battery_voltage": "Test battery health, alternator and charging circuit",
    "coolant_temp": "Inspect radiator, coolant level, water pump and fan",
    "fuel_consumption": "Inspect injectors, air filter and O2 sensors",
    "vehicle_speed": "Review driving profile / speed sensor",
    "operating_hours": "Schedule periodic service based on hours",
}

# A reading is labelled positive if the vehicle fails within this many hours.
FAILURE_HORIZON = 48
# Rolling window (hours) for engineered features and for sequence models.
WINDOW = 24
ROLL = 12

# Status thresholds on failure probability.
WARNING_THRESHOLD = 0.40
CRITICAL_THRESHOLD = 0.80

# Model used for probability + SHAP explanations by default.
PRIMARY_MODEL = "xgboost"

ML_MODELS = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
DL_MODELS = ["lstm", "gru", "transformer"]
ALL_MODELS = ML_MODELS + DL_MODELS
