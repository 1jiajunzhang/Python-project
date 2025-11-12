import pandas as pd


def compute_trend_score(ts: pd.Series) -> float:
    ts = ts.dropna().sort_values()
    if ts.shape[0] < 5:
        return 0.0
    short_ma = ts.rolling(3).mean()
    long_ma = ts.rolling(6).mean()
    if long_ma.iloc[-1] == 0 or pd.isna(long_ma.iloc[-1]):
        return 0.0
    slope = (short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1]
    return max(-1.0, min(1.0, float(slope * 5)))
