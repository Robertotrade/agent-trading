"""Velas de 5 minutos de Alpaca: AÑOS de experiencia para el agente de day trading.

Yahoo solo da 60 días de velas de 5 minutos. Alpaca las da gratis desde 2016 con su
plan Basic (solo pide no consultar los últimos 15 minutos). Con eso el agente aprende
de cientos de miles de rupturas en vez de unos cuantos miles.

Hace falta una cuenta de Alpaca (la demo es gratis) y sus llaves en las variables
ALPACA_API_KEY y ALPACA_SECRET_KEY (en GitHub van en Settings → Secrets → Actions;
en tu compu, en el archivo .env). Lo que baja se guarda en data/alpaca_5m/ y la
siguiente vez solo baja lo que falta.

Uso para solo bajar (sin entrenar):  python -m agent.alpaca_datos
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import ROOT
from .data import COLS
from .velas import NY, _horario_normal

CARPETA = ROOT / "data" / "alpaca_5m"


def hay_llaves() -> bool:
    return bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))


def _archivo(t: str):
    return CARPETA / f"{t}.csv.gz"


def _cargar(t: str) -> pd.DataFrame | None:
    p = _archivo(t)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
    return df.set_index("ts").sort_index()


def _guardar(t: str, df: pd.DataFrame) -> None:
    CARPETA.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = out.index.tz_convert("UTC")
    out.to_csv(_archivo(t), index_label="ts", compression="gzip", float_format="%.6g")


def cargar_todo(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Lo que ya está guardado, sin bajar nada."""
    out = {}
    for t in tickers:
        df = _cargar(t)
        if df is not None and len(df):
            out[t] = df
    return out


def _bajar(cliente, simbolo: str, inicio: datetime, fin: datetime) -> pd.DataFrame | None:
    """Una consulta a Alpaca (con reintentos si pide esperar)."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    req = StockBarsRequest(symbol_or_symbols=simbolo, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                           start=inicio, end=fin, adjustment=Adjustment.ALL, feed=DataFeed.SIP)
    for intento in range(6):
        try:
            barras = cliente.get_stock_bars(req)
            break
        except Exception as e:  # noqa: BLE001
            if intento == 5:
                raise
            espera = 20 * (intento + 1)
            print(f"    (Alpaca pidió esperar: {str(e)[:80]}; reintento en {espera} s)")
            time.sleep(espera)
    df = barras.df
    if df is None or df.empty:
        return None
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})[COLS]
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(NY)
    return _horario_normal(df.astype(float))


def actualizar(tickers: list[str], desde: str = "2016-01-01") -> dict[str, pd.DataFrame]:
    """Baja lo que falte de cada acción (por años) y regresa {ticker: velas en hora de NY}."""
    from alpaca.data.historical import StockHistoricalDataClient

    cliente = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    fin = datetime.now(timezone.utc) - timedelta(minutes=20)   # el plan gratis no da los últimos 15 min
    inicio_total = pd.Timestamp(desde, tz="UTC").to_pydatetime()
    out, t0 = {}, time.time()
    for n, t in enumerate(tickers, 1):
        viejo = _cargar(t)
        if viejo is not None and len(viejo):
            a = (viejo.index.max() + pd.Timedelta(minutes=5)).tz_convert("UTC").to_pydatetime()
        else:
            a = inicio_total
        partes = [] if viejo is None else [viejo]
        nuevas = 0
        while a < fin:
            b = min(a + timedelta(days=366), fin)
            try:
                df = _bajar(cliente, t.replace("-", "."), a, b)   # BRK-B en Yahoo es BRK.B en Alpaca
            except Exception as e:  # noqa: BLE001
                print(f"  ! {t}: no pude bajar {a:%Y-%m-%d} a {b:%Y-%m-%d} ({str(e)[:80]})")
                break
            if df is not None and len(df):
                partes.append(df)
                nuevas += len(df)
            a = b
        if partes:
            todo = pd.concat(partes)
            todo = todo[~todo.index.duplicated(keep="last")].sort_index()
            if nuevas:
                _guardar(t, todo)
            out[t] = todo
        if nuevas or n % 10 == 0 or n == len(tickers):
            print(f"  [{n}/{len(tickers)}] {t}: {len(out.get(t, [])):,} velas ({nuevas:,} nuevas) "
                  f"· {int(time.time() - t0)} s")
    return out


def main() -> None:
    from . import load_config, load_env
    from .daytrade import universo

    load_env()
    if not hay_llaves():
        raise SystemExit("No encontré las llaves de Alpaca (ALPACA_API_KEY y ALPACA_SECRET_KEY). "
                         "Revisa los Secrets del repositorio en GitHub.")
    cfg = load_config()
    tickers = universo(cfg)
    print(f"Bajando años de velas de 5 minutos de Alpaca para {len(tickers)} acciones...")
    velas = actualizar(tickers, desde=cfg["day_trading"].get("alpaca_desde", "2016-01-01"))
    total = sum(len(v) for v in velas.values())
    dias = {d for v in velas.values() for d in v.index.date}
    print(f"Listo: {len(velas)} acciones, {total:,} velas, {len(dias):,} días de bolsa "
          f"({min(dias) if dias else '-'} → {max(dias) if dias else '-'}).")
    if len(velas) < 10:
        raise SystemExit("Alpaca casi no regresó velas: revisa que las llaves sean de Paper Trading y estén bien copiadas.")


if __name__ == "__main__":
    main()
