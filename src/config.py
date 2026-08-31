"""Központi konfiguráció.

Minden útvonal és beállítás egy helyen van, hogy a kísérletek
reprodukálhatóak legyenek. A RANDOM_STATE-et SOHA ne írd át futás közben.
"""

from pathlib import Path

# --- Útvonalak ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_CSV = DATA_RAW / "telco_churn.csv"
TRAIN_CSV = DATA_PROCESSED / "train.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"

# --- Adatforrás --------------------------------------------------------------
# IBM Telco Customer Churn (7043 ügyfél, 20 jellemző + célváltozó)
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

# --- Kísérleti beállítások ---------------------------------------------------
RANDOM_STATE = 42          # fix seed: enélkül nem reprodukálható a kutatás
TEST_SIZE = 0.2            # 20% teszthalmaz, rétegzett bontással

TARGET = "Churn"           # célváltozó (1 = lemorzsolódott)
ID_COLUMN = "customerID"   # azonosító, modellezésre nem használjuk

# A jellemzők típus szerinti besorolása (a 3. héten a SHAP ezeket a
# neveket fogja megjeleníteni, ezért fontos, hogy beszédesek legyenek)
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def ensure_dirs() -> None:
    """Létrehozza a szükséges mappákat, ha még nem léteznek."""
    for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
