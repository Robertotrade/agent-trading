"""Historial PÚBLICO y EN VIVO del agente de day trading.

Uso (cada tarde, después del cierre; en GitHub corre solo):
    python -m agent.historial          # anota los días nuevos
    python -m agent.historial --cache  # con las velas ya guardadas

Cómo se mantiene honesto:
 - Cada día se evalúa con el modelo que YA estaba guardado antes de que abriera la bolsa
   (el que se entrenó la noche anterior). Nunca con uno que haya visto ese día.
 - El agente decide cada ruptura solo con las velas que ya habían cerrado antes de ella.
 - Los días se agregan al final y no se borran. En GitHub cada anotación queda con fecha
   y hora en el historial del repositorio, así que nadie puede cambiar los días malos.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np

from . import MODELS_DIR, WEB_DATA_DIR, load_config, write_web_bundle
from .apertura import FEATURES_AP, elegir, setups
from .daytrade import VERSION_DT, obtener_velas

ARCHIVO = WEB_DATA_DIR / "historial_dt.json"


def cargar() -> dict:
    try:
        return json.loads(ARCHIVO.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"empezo": None, "dias": []}


def resumen(h: dict) -> dict:
    out = {"dias": len(h["dias"]), "fechas": [d["dia"] for d in h["dias"]]}
    for modo in ("agente", "regla"):
        r = np.array([d[modo]["resultado_dia"] for d in h["dias"]])
        netos = np.array([o["neto"] for d in h["dias"] for o in d[modo]["operaciones"]])
        eq = np.cumprod(1 + r) if len(r) else np.array([1.0])
        out[modo] = {"retorno_total": round(float(eq[-1] - 1), 5), "operaciones": int(len(netos)),
                     "ganadoras": round(float((netos > 0).mean()), 4) if len(netos) else None,
                     "promedio_bps": round(float(netos.mean() * 1e4), 2) if len(netos) else None,
                     "curva": eq.round(5).tolist()}
    spy = np.array([d["spy"] for d in h["dias"]])
    eq = np.cumprod(1 + spy) if len(spy) else np.array([1.0])
    out["spy"] = {"retorno_total": round(float(eq[-1] - 1), 5), "curva": eq.round(5).tolist()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true", help="usar las velas ya guardadas")
    args = ap.parse_args()
    cfg = load_config()
    dt = cfg["day_trading"]

    path = MODELS_DIR / "daytrade.joblib"
    if not path.exists():
        raise SystemExit("Primero entrena al agente: python -m agent.daytrade")
    b = joblib.load(path)
    if b.get("version") != VERSION_DT or b.get("demo") or not b.get("datos_hasta"):
        raise SystemExit("El modelo guardado no sirve para el historial (es de prueba o de otra versión). "
                         "Entrénalo con datos reales: python -m agent.daytrade")
    hasta = b["datos_hasta"]

    velas = obtener_velas(cfg, demo=False, cache=args.cache)
    s = setups(velas, dt, dt["mercado"])
    h = cargar()
    ya = {d["dia"] for d in h["dias"]}
    nuevos = sorted(d for d in s.loc[s["completo"], "dia"].unique() if str(d) > hasta and str(d) not in ya)
    if not nuevos:
        print("No hay días nuevos para anotar.")
        write_web_bundle()
        return

    s["espera"] = b["model"].predict(s[FEATURES_AP])
    mk = velas[dt["mercado"]]["Close"]
    spy_ret = mk.groupby(mk.index.date).last().pct_change()
    w = 1.0 / dt["max_operaciones_dia"]
    ahora = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    for d in nuevos:
        reg = {"dia": str(d), "anotado": ahora, "modelo_entrenado": b["trained_at"], "modelo_vio_hasta": hasta}
        for modo in ("agente", "regla"):
            ops = elegir(s, [d], modo, dt)
            reg[modo] = {
                "resultado_dia": round(float(ops["neto"].sum() * w), 6),
                "operaciones": [{"ticker": o["ticker"], "lado": "COMPRA" if o["lado"] > 0 else "CORTO",
                                 "hora": o["hora_entrada"], "entrada": round(float(o["entrada"]), 4),
                                 "salida": round(float(o["salida"]), 4), "neto": round(float(o["neto"]), 6),
                                 "motivo": o["motivo"], "espera_bps": round(float(o["espera"]), 1)}
                                for _, o in ops.iterrows()]}
        reg["spy"] = round(float(spy_ret.get(d, 0.0)), 6)
        h["dias"].append(reg)
        a = reg["agente"]
        print(f"  {d}: agente {len(a['operaciones'])} operaciones, {a['resultado_dia']:+.3%} · "
              f"regla {reg['regla']['resultado_dia']:+.3%} · SPY {reg['spy']:+.2%}")
    h["empezo"] = h["empezo"] or h["dias"][0]["dia"]
    h["resumen"] = resumen(h)
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVO.write_text(json.dumps(h, ensure_ascii=False, indent=1), encoding="utf-8")
    write_web_bundle()
    r = h["resumen"]
    print(f"Historial en vivo desde el {h['empezo']} ({r['dias']} días): agente {r['agente']['retorno_total']:+.2%} "
          f"· regla {r['regla']['retorno_total']:+.2%} · SPY {r['spy']['retorno_total']:+.2%}")


if __name__ == "__main__":
    main()
