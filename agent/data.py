"""Descarga de precios históricos.

- Modo normal: usa yfinance (gratis, datos diarios de Yahoo Finance).
- Modo demo (--demo): genera precios sintéticos para probar sin internet.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["Open", "High", "Low", "Close", "Volume"]
CACHE = Path(__file__).resolve().parent.parent / "data" / "precios.csv.gz"


def save_cache(data: dict[str, pd.DataFrame]) -> None:
    """Guarda una copia de los precios para poder re-entrenar sin volver a bajarlos."""
    CACHE.parent.mkdir(exist_ok=True)
    long = pd.concat(data, names=["ticker", "date"]).reset_index()
    long.to_csv(CACHE, index=False, compression="gzip", float_format="%.6g")


def load_cache() -> dict[str, pd.DataFrame]:
    long = pd.read_csv(CACHE, parse_dates=["date"])
    return {t: g.drop(columns="ticker").set_index("date").sort_index()
            for t, g in long.groupby("ticker")}


def _pick(raw: pd.DataFrame, t: str) -> pd.DataFrame | None:
    """Saca las columnas de un ticker, venga como venga el formato de yfinance."""
    if raw is None or raw.empty:
        return None
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw
    for lvl in range(raw.columns.nlevels):
        if t in raw.columns.get_level_values(lvl):
            return raw.xs(t, axis=1, level=lvl)
    return None


def download(tickers: list[str], start: str, end: str | None = None) -> dict[str, pd.DataFrame]:
    """Regresa {ticker: DataFrame con columnas Open, High, Low, Close, Volume}."""
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    pending = list(tickers)
    for intento in range(3):          # reintenta los que fallen (Yahoo a veces limita)
        if not pending:
            break
        if intento:
            print(f"  reintentando {len(pending)} acciones...")
            time.sleep(5 * intento)
        try:
            raw = yf.download(pending, start=start, end=end, auto_adjust=True,
                              group_by="ticker", progress=False, threads=True)
        except Exception as e:  # noqa: BLE001
            print(f"  ! error al descargar: {e}")
            continue
        for t in list(pending):
            df = _pick(raw, t)
            if df is None or not set(COLS) <= set(df.columns):
                continue
            df = df[COLS].dropna()
            if len(df) > 300:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                out[t] = df
                pending.remove(t)
    for t in pending:
        print(f"  ! sin datos suficientes para {t}, lo salto")
    if not out:
        raise SystemExit("No pude bajar datos del mercado. Revisa tu internet e inténtalo en unos minutos.")
    return out


def demo_data(tickers: list[str], start: str = "2012-01-01", seed: int = 7) -> dict[str, pd.DataFrame]:
    """Precios sintéticos con tendencias y regímenes de volatilidad (solo para pruebas)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, pd.Timestamp.today().normalize())
    n = len(dates)
    market = rng.normal(0.0003, 0.010, n)
    out = {}
    for i, t in enumerate(tickers):
        beta = 0.7 + 0.6 * rng.random()
        drift = rng.normal(0.0002, 0.0002)
        vol = np.where(rng.random(n) < 0.15, 0.03, 0.013)
        # un poco de momentum para que haya algo que aprender
        noise = rng.normal(0, 1, n) * vol
        ret = np.zeros(n)
        for k in range(1, n):
            ret[k] = drift + beta * market[k] + 0.04 * ret[k - 1] + noise[k]
        close = 50 * (1 + i * 0.3) * np.exp(np.cumsum(ret))
        high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
        openp = close * (1 + rng.normal(0, 0.003, n))
        volume = rng.lognormal(15, 0.4, n) * (1 + 3 * (vol > 0.02))
        out[t] = pd.DataFrame({"Open": openp, "High": high, "Low": low,
                               "Close": close, "Volume": volume}, index=dates)
    return out


def get_data(cfg: dict, demo: bool = False, cache: bool = False) -> dict[str, pd.DataFrame]:
    tickers = list(dict.fromkeys(cfg["tickers"] + [cfg["benchmark"]]))
    if demo:
        print("Usando datos DEMO sintéticos (no son precios reales).")
        return demo_data(tickers, cfg["data"]["start"])
    if cache:
        if not CACHE.exists():
            raise SystemExit("Todavía no hay precios guardados. Entrena una vez sin --cache.")
        print(f"Usando los precios guardados ({CACHE.name}).")
        data = {t: df for t, df in load_cache().items() if t in tickers}
    else:
        print(f"Descargando {len(tickers)} acciones desde {cfg['data']['start']}...")
        data = download(tickers, cfg["data"]["start"])
        try:
            save_cache(data)
        except Exception as e:  # noqa: BLE001
            print(f"  (no pude guardar la copia de los precios: {e})")
    if cfg["benchmark"] not in data:
        raise SystemExit(f"No pude bajar {cfg['benchmark']} (el mercado de referencia). Inténtalo en unos minutos.")
    return data
