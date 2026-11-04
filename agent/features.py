"""Indicadores técnicos que el modelo usa para decidir.

Todo se calcula solo con información disponible ese día (sin mirar el futuro).
Objetivo (target): ¿la acción le gana al mercado (SPY) en los próximos `horizon` días?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Indicadores de cada acción por separado
BASE_FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_60",
    "vol_10", "vol_20", "vol_ratio",
    "rsi_14", "macd_hist",
    "dist_sma20", "dist_sma50", "dist_sma200",
    "range_20", "volume_z",
    "rel_bench_20", "bench_ret_20", "bench_dist_sma200",
]
# Cómo va cada acción comparada con las demás ese mismo día (0 = la peor, 1 = la mejor)
CS_SOURCES = ["ret_5", "ret_20", "ret_60", "vol_20", "rsi_14", "dist_sma50", "volume_z"]
CS_FEATURES = [f"{c}_rank" for c in CS_SOURCES]
FEATURES = BASE_FEATURES + CS_FEATURES


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def make_features(df: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for n in (1, 5, 10, 20, 60):
        f[f"ret_{n}"] = c.pct_change(n)
    f["vol_10"] = r.rolling(10).std()
    f["vol_20"] = r.rolling(20).std()
    f["vol_ratio"] = f["vol_10"] / f["vol_20"]
    f["rsi_14"] = _rsi(c) / 100
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c
    for n in (20, 50, 200):
        f[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1
    f["range_20"] = (df["High"].rolling(20).max() - df["Low"].rolling(20).min()) / c
    lv = np.log(df["Volume"].replace(0, np.nan))
    f["volume_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()

    bc = bench["Close"].reindex(df.index).ffill()
    f["bench_ret_20"] = bc.pct_change(20)
    f["rel_bench_20"] = f["ret_20"] - f["bench_ret_20"]
    f["bench_dist_sma200"] = bc / bc.rolling(200).mean() - 1
    return f.replace([np.inf, -np.inf], np.nan)


def add_cross_sectional(ds: pd.DataFrame) -> pd.DataFrame:
    """Agrega el lugar (0 a 1) de cada acción contra las demás en la misma fecha."""
    g = ds.groupby("date")
    for c in CS_SOURCES:
        ds[f"{c}_rank"] = g[c].rank(pct=True)
    return ds


def build_dataset(data: dict[str, pd.DataFrame], tickers: list[str], bench: str,
                  horizon: int) -> pd.DataFrame:
    """Une todas las acciones en una tabla larga: (fecha, ticker) -> features + target."""
    bc = data[bench]["Close"]
    bench_fwd = bc.shift(-horizon) / bc - 1
    rows = []
    for t in tickers:
        if t not in data:
            continue
        df = data[t]
        f = make_features(df, data[bench])
        fwd = df["Close"].shift(-horizon) / df["Close"] - 1
        rel = fwd - bench_fwd.reindex(df.index)          # cuánto le ganó (o perdió) al mercado
        f["fwd_ret"] = fwd
        f["next_ret"] = df["Close"].pct_change().shift(-1)   # para el backtest
        f["target"] = (rel > 0).astype(float).where(rel.notna())
        f["ticker"] = t
        rows.append(f)
    ds = pd.concat(rows)
    ds.index.name = "date"
    return add_cross_sectional(ds.reset_index())
