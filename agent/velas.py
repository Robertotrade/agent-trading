"""Velas de 5 minutos: descarga, copia local y "lectura" de velas y gráfica.

Yahoo Finance da gratis velas de 5 minutos de los últimos 60 días. Cada vez que
entrenas, las nuevas se suman a la copia en data/velas_5m.csv.gz, así que con el
tiempo el agente junta más historia para aprender.
"""
from __future__ import annotations

import time
from datetime import time as hora
from pathlib import Path

import numpy as np
import pandas as pd

from .data import COLS, _pick
from .features import _rsi

NY = "America/New_York"
CACHE = Path(__file__).resolve().parent.parent / "data" / "velas_5m.csv.gz"

# Patrones clásicos de velas japonesas: nombre y lo que "dicen los libros"
PATRONES = {
    "p_doji": ("Doji", "indecisión"),
    "p_martillo": ("Martillo", "sube"),
    "p_estrella_fugaz": ("Estrella fugaz", "baja"),
    "p_envolvente_alcista": ("Envolvente alcista", "sube"),
    "p_envolvente_bajista": ("Envolvente bajista", "baja"),
    "p_vela_interior": ("Vela interior", "pausa"),
    "p_tres_verdes": ("Tres velas verdes", "sube"),
    "p_tres_rojas": ("Tres velas rojas", "baja"),
}
# Cómo es la vela (cuerpo, mechas, tamaño) y las 2 anteriores
VELA_FEATURES = ["cuerpo", "rango", "mecha_sup", "mecha_inf", "cuerpo_rel", "rango_rel",
                 "cuerpo_1", "cuerpo_2", *PATRONES]
# Qué dice la gráfica alrededor: tendencia, VWAP, rango del día, apertura, volumen
GRAFICA_FEATURES = ["ret_1", "ret_3", "ret_6", "ret_12", "volatilidad_12", "rsi_14",
                    "dist_ema9", "dist_ema20", "vol_rel", "dist_vwap", "ret_dia", "gap_dia",
                    "pos_rango_dia", "ruptura_apertura", "minuto"]
# Qué está haciendo el mercado (SPY) en ese momento
MERCADO_FEATURES = ["mkt_ret_3", "mkt_ret_12", "mkt_dist_vwap", "mkt_ret_dia"]
FEATURES = VELA_FEATURES + GRAFICA_FEATURES + MERCADO_FEATURES


# ---------------------------------------------------------------- datos

def _horario_normal(df: pd.DataFrame) -> pd.DataFrame:
    t = df.index.time
    return df[(t >= hora(9, 30)) & (t <= hora(15, 55))]


def quitar_vela_incompleta(df: pd.DataFrame) -> pd.DataFrame:
    """La vela que todavía se está formando no cuenta."""
    ahora = pd.Timestamp.now(tz=NY)
    return df[df.index + pd.Timedelta(minutes=5) <= ahora]


def descargar(tickers: list[str], periodo: str = "60d") -> dict[str, pd.DataFrame]:
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    pending = list(tickers)
    for intento in range(3):
        if not pending:
            break
        if intento:
            print(f"  reintentando {len(pending)} acciones...")
            time.sleep(5 * intento)
        de_quien = {}
        for i in range(0, len(pending), 40):          # de 40 en 40 para no saturar a Yahoo
            grupo = pending[i:i + 40]
            try:
                raw = yf.download(grupo, period=periodo, interval="5m", group_by="ticker",
                                  auto_adjust=False, prepost=False, progress=False, threads=True)
            except Exception as e:  # noqa: BLE001
                print(f"  ! error al descargar velas: {e}")
                continue
            de_quien.update({t: raw for t in grupo})
        for t in list(pending):
            df = _pick(de_quien[t], t) if t in de_quien else None
            if df is None or not set(COLS) <= set(df.columns):
                continue
            df = df[COLS].dropna()
            if len(df) < 200:
                continue
            idx = pd.to_datetime(df.index)
            df.index = idx.tz_localize(NY) if idx.tz is None else idx.tz_convert(NY)
            out[t] = quitar_vela_incompleta(_horario_normal(df))
            pending.remove(t)
    for t in pending:
        print(f"  ! sin velas para {t}, la salto")
    if not out:
        raise SystemExit("No pude bajar velas de 5 minutos. Revisa tu internet e inténtalo en unos minutos.")
    return out


def guardar_cache(velas: dict[str, pd.DataFrame]) -> int:
    """Suma las velas nuevas a la copia local. Regresa cuántas velas hay guardadas."""
    CACHE.parent.mkdir(exist_ok=True)
    nuevo = pd.concat(velas, names=["ticker", "ts"]).reset_index()
    nuevo["ts"] = nuevo["ts"].dt.tz_convert("UTC")
    if CACHE.exists():
        viejo = pd.read_csv(CACHE)
        viejo["ts"] = pd.to_datetime(viejo["ts"], utc=True)
        nuevo = pd.concat([viejo, nuevo]).drop_duplicates(["ticker", "ts"], keep="last")
    nuevo = nuevo.sort_values(["ticker", "ts"])
    nuevo.to_csv(CACHE, index=False, compression="gzip", float_format="%.6g")
    return len(nuevo)


def cargar_cache() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(CACHE)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
    return {t: g.drop(columns="ticker").set_index("ts").sort_index() for t, g in df.groupby("ticker")}


def velas_demo(tickers: list[str], dias: int = 60, seed: int = 11) -> dict[str, pd.DataFrame]:
    """Velas sintéticas de 5 minutos (solo para probar sin internet)."""
    rng = np.random.default_rng(seed)
    fechas = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=dias)
    minutos = np.arange(78) * 5
    forma_u = 1.6 - np.sin(np.linspace(0, np.pi, 78))      # más movimiento al abrir y al cerrar
    mercado = rng.normal(0, 0.0008, (dias, 78)) * forma_u
    out = {}
    for k, t in enumerate(tickers):
        beta, vol = 0.6 + 0.8 * rng.random(), 0.0010 + 0.0010 * rng.random()
        filas, precio = [], 50.0 + 20 * k
        for d, fecha in enumerate(fechas):
            precio *= 1 + rng.normal(0, 0.006)               # hueco de apertura
            r_prev = 0.0
            for b in range(78):
                r = beta * mercado[d, b] + rng.normal(0, vol) * forma_u[b] - 0.05 * r_prev
                o = precio
                c = o * (1 + r)
                h = max(o, c) * (1 + abs(rng.normal(0, vol / 2)))
                low = min(o, c) * (1 - abs(rng.normal(0, vol / 2)))
                ts = pd.Timestamp(fecha.date()).tz_localize(NY) + pd.Timedelta(hours=9, minutes=30 + int(minutos[b]))
                filas.append((ts, o, h, low, c, rng.lognormal(10, 0.3) * forma_u[b]))
                precio, r_prev = c, r
        df = pd.DataFrame(filas, columns=["ts", *COLS]).set_index("ts")
        out[t] = df
    return out


# ---------------------------------------------------------------- lectura de velas

def leer_velas(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte cada vela (y su contexto) en números que el modelo entiende."""
    o, h, l, c, v = (df[k] for k in COLS)
    f = pd.DataFrame(index=df.index)
    rango = h - l
    seguro = rango.where(rango > 0)
    # anatomía de la vela
    f["cuerpo"] = (c - o) / o
    f["rango"] = rango / o
    f["mecha_sup"] = (h - np.maximum(o, c)) / seguro
    f["mecha_inf"] = (np.minimum(o, c) - l) / seguro
    f["cuerpo_rel"] = (c - o).abs() / seguro
    f["rango_rel"] = rango / rango.rolling(20).mean()
    f["cuerpo_1"] = f["cuerpo"].shift(1)
    f["cuerpo_2"] = f["cuerpo"].shift(2)
    # patrones clásicos
    po, pc, ph, pl = o.shift(1), c.shift(1), h.shift(1), l.shift(1)
    sube, baja = c > o, c < o
    f["p_doji"] = f["cuerpo_rel"] < 0.1
    f["p_martillo"] = (f["mecha_inf"] > 0.6) & (f["mecha_sup"] < 0.15)
    f["p_estrella_fugaz"] = (f["mecha_sup"] > 0.6) & (f["mecha_inf"] < 0.15)
    f["p_envolvente_alcista"] = (pc < po) & sube & (c >= po) & (o <= pc)
    f["p_envolvente_bajista"] = (pc > po) & baja & (c <= po) & (o >= pc)
    f["p_vela_interior"] = (h < ph) & (l > pl)
    f["p_tres_verdes"] = sube & sube.shift(1, fill_value=False) & sube.shift(2, fill_value=False)
    f["p_tres_rojas"] = baja & baja.shift(1, fill_value=False) & baja.shift(2, fill_value=False)
    for p in PATRONES:
        f[p] = f[p].astype(float)
    # contexto de la gráfica
    for n in (1, 3, 6, 12):
        f[f"ret_{n}"] = c.pct_change(n)
    f["volatilidad_12"] = c.pct_change().rolling(12).std()
    f["rsi_14"] = _rsi(c) / 100
    f["dist_ema9"] = c / c.ewm(span=9, adjust=False).mean() - 1
    f["dist_ema20"] = c / c.ewm(span=20, adjust=False).mean() - 1
    f["vol_rel"] = v / v.rolling(20).mean()
    # dentro del día: VWAP, apertura, rango del día, primeros 30 minutos
    dia = pd.Series(df.index.date, index=df.index)
    tp = (h + l + c) / 3
    vwap = (tp * v).groupby(dia).cumsum() / v.groupby(dia).cumsum().replace(0, np.nan)
    f["dist_vwap"] = c / vwap - 1
    apertura = o.groupby(dia).transform("first")
    f["ret_dia"] = c / apertura - 1
    f["gap_dia"] = apertura / dia.map(c.groupby(dia).last().shift(1)) - 1
    alto, bajo = h.groupby(dia).cummax(), l.groupby(dia).cummin()
    f["pos_rango_dia"] = (c - bajo) / (alto - bajo).where(alto > bajo)
    n_vela = c.groupby(dia).cumcount()
    f["minuto"] = n_vela * 5
    or_alto = h.where(n_vela < 6).groupby(dia).transform("max")
    or_bajo = l.where(n_vela < 6).groupby(dia).transform("min")
    ruptura = np.where(c > or_alto, 1.0, np.where(c < or_bajo, -1.0, 0.0))
    f["ruptura_apertura"] = np.where(n_vela >= 6, ruptura, np.nan)
    return f.replace([np.inf, -np.inf], np.nan)
