from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 数据文件
ROLLING_SALES_CSV = BASE_DIR / "data" / "rollingsales_queens_clean.csv"

APP_TITLE = "Queens Housing Trends & Buy-Timing Advisor"

DEFAULT_LAT = 40.728
DEFAULT_LON = -73.85

# 权重
TREND_WEIGHT = 0.444
RATIO_WEIGHT = 0.333
AFFORD_WEIGHT = 0.222

# 地理编码
GEOCODE_CACHE = BASE_DIR / "data" / "geocode_cache.parquet"
GEOCODE_CITY_HINT = "Queens, NY"
GEOCODE_BATCH_LIMIT = 600
GEOCODE_CALLS_PER_PERIOD = 1
GEOCODE_PERIOD_SECONDS = 1

def ensure_data_exists():
    if not ROLLING_SALES_CSV.exists():
        raise FileNotFoundError("Missing data/rollingsales_queens_clean.csv in ./data")
