"""One place for every setting. Change here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # Yahoo downloads (not in git)
OUT_DIR = ROOT / "outputs"
MASTER_PATH = DATA_DIR / "master_v1.csv"
TOP50_PATH = DATA_DIR / "top50_tickers.csv"

# ---- Data period ----
START_DATE = "2021-01-01"
END_DATE = "2026-09-23"          # exclusive: last day is 2026-09-22
INDEX_TICKER, VIX_TICKER = "^GSPC", "^VIX"
TOP_N = 50
SHARE_CLASS_DUPES = [("GOOGL", "GOOG")]   # count Alphabet once
VIX_STRESS = 20                          # VIX >= 20 = "stressed" day

# Top-50 S&P 500 by index weight, Dec 2024 (Kaggle). Fixed so the list never changes.
TICKER_WEIGHTS = {
    "AAPL": 0.069209, "NVDA": 0.059350, "MSFT": 0.058401, "AMZN": 0.042550,
    "GOOGL": 0.042309, "GOOG": 0.042309, "META": 0.026581, "TSLA": 0.024317,
    "AVGO": 0.018553, "BRK-B": 0.017609, "WMT": 0.013332, "LLY": 0.012422,
    "JPM": 0.012035, "V": 0.011069, "MA": 0.008719, "ORCL": 0.008537,
    "XOM": 0.008371, "UNH": 0.008281, "COST": 0.007619, "PG": 0.007121,
    "HD": 0.007016, "NFLX": 0.006991, "JNJ": 0.006258, "BAC": 0.006097,
    "CRM": 0.005917, "ABBV": 0.005582, "KO": 0.004848, "TMUS": 0.004600,
    "CVX": 0.004582, "MRK": 0.004462, "WFC": 0.004213, "CSCO": 0.004193,
    "ACN": 0.004123, "NOW": 0.004051, "AXP": 0.003785, "MCD": 0.003773,
    "PEP": 0.003771, "BX": 0.003728, "IBM": 0.003716, "DIS": 0.003650,
    "LIN": 0.003635, "TMO": 0.003606, "MS": 0.003578, "ABT": 0.003565,
    "ADBE": 0.003541, "AMD": 0.003481, "PM": 0.003475, "ISRG": 0.003361,
    "PLTR": 0.003301, "GE": 0.003278, "INTU": 0.003240, "GS": 0.003197,
    # spares (used only if a ticker above fails to download)
    "CAT": 0.003180, "TXN": 0.003067, "QCOM": 0.003056, "VZ": 0.003024,
    "BKNG": 0.003006, "DHR": 0.002970, "T": 0.002937, "BLK": 0.002866,
    "RTX": 0.002789,
}

# ---- Walk-forward (3 folds, expanding window) ----
FOLDS = [  # (name, first test day, last test day)
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-09-21"),
]
WINDOW = 5            # last 5 trading days as model input
VAL_FRACTION = 0.10   # last 10% of training period = validation slice (early stopping)
SEED = 42
TARGET = "target_next_ret"

# ---- Frozen-master checks (update after the first real build) ----
EXPECTED_N_FEATURES = 58

# ---- Ablation: do VIX and technical indicators help? (src/ablation.py) ----
GROUP_VIX = ["VIX", "VIX_chg"]
GROUP_TECH = ["RSI_14", "MACD_diff", "BB_pctB"]
GROUP_IDXVOL = ["vol_chg", "hl_range"]
ABLATION_SEEDS = [42, 43, 44]
