"""Backtest: simula al agente usando SOLO predicciones fuera de muestra
(el modelo nunca vio esos días al entrenar).

Estrategia: cada `rebalance_days` días de bolsa compra las `top_n` acciones con
más probabilidad de ganarle al mercado, en partes iguales, y las mantiene hasta
el siguiente cambio. Cobra `cost_bps` por cada compra/venta.

Se compara contra:
 - el mercado (SPY), y
 - comprar TODAS las acciones del universo en partes iguales. Esta es la prueba
   justa de si el agente sabe escoger: si no le gana a esto, escoger no sirvió.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(preds: pd.DataFrame, bench_close: pd.Series, cfg: dict) -> dict:
    s, risk = cfg["strategy"], cfg["risk"]
    p = preds.pivot(index="date", columns="ticker", values="prob").sort_index()
    nxt = preds.pivot(index="date", columns="ticker", values="next_ret").reindex_like(p)
    n, every = s["top_n"], s["rebalance_days"]
    size = risk["invest_pct"] / n

    # cartera: se decide en los días de rebalanceo y se mantiene hasta el siguiente
    w = pd.DataFrame(np.nan, index=p.index, columns=p.columns)
    for k, d in enumerate(p.index):
        if k % every == 0:
            picks = p.loc[d].dropna().nlargest(n).index
            w.loc[d] = 0.0
            w.loc[d, picks] = size
    w = w.ffill().fillna(0.0)

    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    curves = {
        "agente": (w * nxt.fillna(0.0)).sum(axis=1) - turnover * s["cost_bps"] / 1e4,
        "iguales": nxt.where(p.notna()).mean(axis=1).fillna(0.0),
        "benchmark": bench_close.reindex(p.index).ffill().pct_change().shift(-1).fillna(0.0),
    }
    equity = {k: (1 + v).cumprod() for k, v in curves.items()}
    por_anio = [{"anio": int(y), **{k: round(float((1 + v[p.index.year == y]).prod() - 1), 4)
                                     for k, v in curves.items()}}
                for y in sorted(set(p.index.year))]
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in p.index],
        "equity": equity["agente"].round(5).tolist(),
        "iguales": equity["iguales"].round(5).tolist(),
        "benchmark": equity["benchmark"].round(5).tolist(),
        "metrics": {k: _metrics(v) for k, v in curves.items()},
        "por_anio": por_anio,
        "exposure_avg": float(w.sum(axis=1).mean()),
        "top_n": n,
        "universo": int(p.shape[1]),
    }


def _metrics(daily: pd.Series) -> dict:
    eq = (1 + daily).cumprod()
    years = max(len(daily) / 252, 1e-9)
    cagr = eq.iloc[-1] ** (1 / years) - 1
    sd = daily.std()
    sharpe = daily.mean() / sd * np.sqrt(252) if sd > 0 else 0.0
    max_dd = (eq / eq.cummax() - 1).min()
    return {
        "retorno_total": round(float(eq.iloc[-1] - 1), 4),
        "cagr": round(float(cagr), 4),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown": round(float(max_dd), 4),
        "dias_ganadores": round(float((daily > 0).mean()), 4),
    }
