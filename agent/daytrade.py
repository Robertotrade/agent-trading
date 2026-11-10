"""Agente de DAY TRADING: qué acciones comprar, cuándo comprarlas y leer las velas.

Uso:
    python -m agent.daytrade          # baja velas reales (Yahoo, últimos 60 días) y entrena
    python -m agent.daytrade --cache  # entrena con las velas ya guardadas
    python -m agent.daytrade --demo   # velas sintéticas, para probar
    python -m agent.daytrade --solo-plan  # solo el plan del día, con el modelo ya entrenado

Qué hace:
 1. QUÉ acciones: cada día busca las que están "en juego" (mucho más volumen que normal
    en los primeros 5 minutos).
 2. CUÁNDO: entra si el precio rompe la primera vela (máximo si fue verde -> compra;
    mínimo si fue roja -> venta en corto). Stop a 10% del ATR; si no, sale al cierre.
 3. El AGENTE lee las velas y la gráfica justo antes de cada ruptura y decide si la toma.
 4. Se examina en días que no vio y se compara contra la regla sin agente y contra SPY.
 5. Arma el plan del día: qué acciones vigilar, en qué precio entrar y dónde va el stop.
 No manda órdenes.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from . import MODELS_DIR, WEB_DATA_DIR, alpaca_datos, load_config, load_env, write_web_bundle
from .apertura import FEATURES_AP, elegir, resultados, setups
from .velas import (CACHE, NY, PATRONES, cargar_cache, descargar, guardar_cache, leer_velas,
                    velas_demo)

VERSION_DT = 2


def universo(cfg: dict) -> list[str]:
    dt = cfg["day_trading"]
    return list(dict.fromkeys(cfg["tickers"] + dt["extras_en_juego"] + dt["acciones"] + [dt["mercado"]]))


def obtener_velas(cfg: dict, demo: bool, cache: bool, fuente: str = "auto") -> dict[str, pd.DataFrame]:
    """Velas de 5 minutos: años desde Alpaca si hay llaves; si no, 60 días de Yahoo."""
    dt, tickers = cfg["day_trading"], universo(cfg)
    if demo:
        print("Usando velas DEMO sintéticas (no son precios reales).")
        return velas_demo(tickers)
    if fuente == "auto" and (alpaca_datos.hay_llaves() or (cache and any(alpaca_datos.CARPETA.glob("*.csv.gz")))):
        if cache:
            print("Usando los años de velas de Alpaca ya guardados.")
            velas = alpaca_datos.cargar_todo(tickers)
        else:
            print(f"Bajando años de velas de 5 minutos de Alpaca ({len(tickers)} acciones). "
                  "La primera vez tarda (hasta una hora); después solo baja lo nuevo.")
            try:
                velas = alpaca_datos.actualizar(tickers, desde=dt.get("alpaca_desde", "2016-01-01"))
            except SystemExit as e:      # llaves mal puestas, etc.: el día no se pierde
                print(f"  ! {e}")
                velas = {}
        if dt["mercado"] in velas and len(velas) > 10:
            return velas
        print("  No pude usar Alpaca (¿llaves correctas?); sigo con los 60 días de Yahoo.")
    if not cache:
        print(f"Bajando velas de 5 minutos de {len(tickers)} acciones (últimos 60 días)...")
        try:
            total = guardar_cache(descargar(tickers))
            print(f"  Velas guardadas en total: {total:,}")
        except SystemExit:
            if not CACHE.exists():
                raise
            print("  No pude bajar velas nuevas; uso las que ya tenía guardadas.")
    elif not CACHE.exists():
        raise SystemExit("Todavía no hay velas guardadas. Entrena una vez sin --cache.")
    velas = {t: df for t, df in cargar_cache().items() if t in tickers}
    if dt["mercado"] not in velas:
        raise SystemExit(f"No tengo velas de {dt['mercado']} (el mercado de referencia).")
    return velas


def modelo() -> HistGradientBoostingRegressor:
    """Aprende cuánto se espera ganar (en pb, después de comisiones) en cada ruptura."""
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=150, max_depth=3,
                                         l2_regularization=1.0, min_samples_leaf=40, random_state=42)


def entrenables(s: pd.DataFrame) -> pd.DataFrame:
    return s[s["entrada"].notna() & s["completo"] & s["pasa_filtros"]]


def examen(s: pd.DataFrame, dt: dict) -> tuple[list[dict], list]:
    """Entrena con días anteriores y se examina en días que no vio. Llena s['espera']."""
    base = entrenables(s)
    dias = sorted(base["dia"].unique())
    k = dt["examenes"] if len(dias) < 250 else dt.get("examenes_con_historia", 8)   # con años, más exámenes
    edges = np.linspace(max(len(dias) // 2, 8), len(dias), k + 1, dtype=int)
    s["espera"] = np.nan
    folds, dias_prueba = [], []
    for j in range(k):
        prueba = dias[edges[j]:edges[j + 1]]
        tr = base[base["dia"] < prueba[0]] if prueba else base.iloc[0:0]
        if not prueba or len(tr) < 50 or tr["target"].nunique() < 2:
            continue
        m = modelo().fit(tr[FEATURES_AP], tr["y_bps"])
        mask = s["dia"].isin(prueba)
        s.loc[mask, "espera"] = m.predict(s.loc[mask, FEATURES_AP])
        te = s.loc[base[base["dia"].isin(prueba)].index]
        auc = roc_auc_score(te["target"], te["espera"]) if te["target"].nunique() == 2 else 0.5
        fold = {"examen": j + 1, "desde": str(prueba[0]), "hasta": str(prueba[-1]), "dias": len(prueba),
                "rupturas": int(len(te)), "auc": round(float(auc), 4),
                "aciertos": round(float(((te["espera"] > 0) == (te["target"] == 1)).mean()), 4)}
        folds.append(fold)
        dias_prueba += list(prueba)
        print(f"  examen {fold['examen']}: {fold['desde']} → {fold['hasta']} ({fold['dias']} días, "
              f"{fold['rupturas']} rupturas)  AUC={fold['auc']:.3f}")
    return folds, dias_prueba


def plan_del_dia(s: pd.DataFrame, m, dt: dict) -> dict:
    """Qué acciones vigilar el último día con velas, dónde entrar y si el agente la tomaría."""
    ult = s["dia"].max()
    hoy = s[(s["dia"] == ult) & s["pasa_filtros"]].copy()
    if hoy.empty:
        return {"dia": str(ult), "filas": []}
    hoy["espera"] = m.predict(hoy[FEATURES_AP])
    filas = []
    for _, r in hoy.sort_values("rel_vol", ascending=False).head(12).iterrows():
        if pd.notna(r["entrada"]):
            como = {"cierre": "al cierre", "stop": "en el stop", "dudoso": "stop dudoso"}.get(r["motivo"], "")
            estado = f"rompió a las {r['hora_entrada']}" + (
                f" · terminó {r['neto']:+.2%} ({como})" if r["completo"] else " · sigue abierta")
        else:
            estado = "no rompió" if r["completo"] else "esperando la ruptura"
        filas.append({"ticker": r["ticker"], "en_juego": round(float(r["rel_vol"]), 2),
                      "lado": "COMPRA" if r["lado"] > 0 else "CORTO",
                      "entrar_en": round(float(r["nivel"]), 2), "stop": round(float(r["stop"]), 2),
                      "espera_bps": round(float(r["espera"]), 1),
                      "decision": "ENTRAR" if r["espera"] >= dt["umbral_agente_bps"] else "NO ENTRAR",
                      "estado": estado})
    hoy_ny = pd.Timestamp.now(tz=NY).date()
    return {"dia": str(ult), "es_hoy": bool(ult == hoy_ny),
            "completo": bool(hoy["completo"].all()), "filas": filas}


# ---------------------------------------------------------------- ¿sirven las velas?

def tabla_velas(velas: dict, cfg: dict) -> tuple[list[dict], dict]:
    """Qué pasó 30 min después de cada patrón clásico de velas japonesas."""
    dt = cfg["day_trading"]
    H = dt["horizonte_velas"]
    filas = []
    for t in dt["acciones"]:
        if t not in velas:
            continue
        df = velas[t]
        dias_t = sorted(set(df.index.date))
        df = df[df.index.date >= dias_t[max(len(dias_t) - 250, 0)]]   # el último año basta
        f = leer_velas(df)[list(PATRONES)]
        o, c = df["Open"].to_numpy(float), df["Close"].to_numpy(float)
        pos = np.arange(len(df))
        ultimo = pd.Series(pos, index=df.index).groupby(df.index.date).transform("max").to_numpy()
        ok = pos + 1 <= ultimo
        fwd = np.full(len(df), np.nan)
        fwd[ok] = c[np.minimum(pos + H, ultimo)[ok]] / o[(pos + 1)[ok]] - 1
        f["fwd"] = fwd
        filas.append(f)
    base = pd.concat(filas).dropna(subset=["fwd"])
    base["sube"] = (base["fwd"] > 0).astype(float)
    p0 = float(base["sube"].mean())
    out = []
    for col, (nombre, libros) in PATRONES.items():
        m = base[col] == 1
        n = int(m.sum())
        if n < 30:
            continue
        sube = float(base.loc[m, "sube"].mean())
        dif = (sube - p0) * (1 if libros == "sube" else -1 if libros == "baja" else 0)
        se = np.sqrt(p0 * (1 - p0) / n)
        veredicto = ("no predice dirección" if libros not in ("sube", "baja") else
                     "sí ayuda, un poco" if dif > 2 * se and dif >= 0.01 else
                     "pasa lo contrario" if dif < -2 * se and dif <= -0.01 else "no se nota")
        out.append({"patron": nombre, "libros": libros, "veces": n, "sube_pct": round(sube, 4),
                    "promedio_bps": round(float(base.loc[m, "fwd"].mean() * 1e4), 2), "veredicto": veredicto})
    return out, {"veces": int(len(base)), "sube_pct": round(p0, 4),
                 "promedio_bps": round(float(base["fwd"].mean() * 1e4), 2)}


def solo_plan(s: pd.DataFrame, dt: dict) -> None:
    """Actualiza solo el plan del día en la página, sin re-entrenar (el modelo no cambia)."""
    path = MODELS_DIR / "daytrade.joblib"
    if not path.exists():
        raise SystemExit("Primero entrena al agente: python -m agent.daytrade")
    b = joblib.load(path)
    if b.get("version") != VERSION_DT:
        raise SystemExit("El modelo es de otra versión. Vuelve a entrenar: python -m agent.daytrade")
    plan = plan_del_dia(s, b["model"], dt)
    archivo = WEB_DATA_DIR / "daytrading.json"
    out = json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else {}
    out["plan"] = plan
    out["plan_actualizado"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    archivo.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    write_web_bundle()
    print(f"Plan del {plan['dia']}:")
    for f in plan["filas"][:8]:
        print(f"  {f['ticker']:<6} en juego {f['en_juego']:.1f}x · {f['lado']:<6} si rompe {f['entrar_en']} "
              f"· stop {f['stop']} · agente: {f['decision']} · {f['estado']}")


def imprimir_examen(est: dict, spy: dict, folds: list, dias_prueba: list, dt: dict) -> None:
    nombres = {"agente": "Agente (lee las velas)", "regla": f"Regla: {dt['en_juego_top']} más en juego",
               "todas": "Cualquier ruptura"}
    print(f"\nSimulación en {len(dias_prueba)} días que no vio (con comisiones):")
    for k, nombre in nombres.items():
        x = est[k]
        if x["operaciones"]:
            print(f"  {nombre:<26} {x['operaciones']:>5} operaciones · ganadoras {x['ganadoras']:.0%} · "
                  f"promedio {x['promedio_bps']:+6.1f} pb · R {x['R_promedio']:+.2f} · resultado {x['retorno_total']:+.2%} "
                  f"· Sharpe {x['sharpe']:+.2f} · peor caída {x['peor_caida']:.1%}")
        else:
            print(f"  {nombre:<26} sin operaciones")
    print(f"  {'Mercado (SPY)':<26} resultado {spy['retorno_total']:+.2%} · Sharpe {spy['sharpe']:+.2f}")
    auc = float(np.mean([f["auc"] for f in folds]))
    print(f"Calificación del agente (AUC promedio en {len(folds)} exámenes): {auc:.3f}")


# ---------------------------------------------------------------- principal

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="velas sintéticas")
    ap.add_argument("--cache", action="store_true", help="usar las velas ya guardadas")
    ap.add_argument("--solo-plan", action="store_true", help="solo el plan del día con el modelo ya guardado")
    ap.add_argument("--solo-examen", action="store_true",
                    help="solo examinar (no guarda modelo ni cambia la página ni el historial)")
    args = ap.parse_args()
    cfg = load_config()
    dt = cfg["day_trading"]

    load_env()
    # el plan de la mañana usa Yahoo (casi en vivo); entrenar y anotar usa años de Alpaca si hay llaves
    velas = obtener_velas(cfg, demo=args.demo, cache=args.cache, fuente="yahoo" if args.solo_plan else "auto")
    s = setups(velas, dt, dt["mercado"])
    if s.empty:
        raise SystemExit("No encontré oportunidades: hacen falta más días de velas.")
    if args.solo_plan:
        solo_plan(s, dt)
        return
    dias = sorted(s["dia"].unique())
    base = entrenables(s)
    print(f"Oportunidades: {len(s):,} (acción, día) en {len(dias)} días · rupturas que se pueden "
          f"examinar: {len(base):,} · acciones: {s['ticker'].nunique()}")

    print("Examen por días (entrena con días anteriores y se prueba en días que no vio):")
    folds, dias_prueba = examen(s, dt)
    if not folds:
        raise SystemExit("Muy pocos días de velas para examinar al agente.")

    est, curvas, ops_agente = {}, {}, None
    for modo in ("agente", "regla", "todas"):
        ops = elegir(s, dias_prueba, modo, dt)
        est[modo], curvas[modo] = resultados(ops, dias_prueba, dt)
        if modo == "agente":
            ops_agente = ops
    mk = velas[dt["mercado"]]["Close"]
    spy_d = mk.groupby(mk.index.date).last().pct_change().reindex(dias_prueba).fillna(0.0)
    eq_spy = (1 + spy_d).cumprod()
    spy = {"retorno_total": round(float(eq_spy.iloc[-1] - 1), 4),
           "sharpe": round(float(spy_d.mean() / spy_d.std() * np.sqrt(252)), 2) if spy_d.std() > 0 else 0.0,
           "peor_caida": round(float((eq_spy / eq_spy.cummax() - 1).min()), 4)}

    # El plan del último día lo decide el modelo que YA existía antes de ese día (el mismo que
    # usa el historial), nunca el que se va a entrenar ahora viendo ese día.
    previo = None
    if (MODELS_DIR / "daytrade.joblib").exists():
        try:
            pb = joblib.load(MODELS_DIR / "daytrade.joblib")
            if pb.get("version") == VERSION_DT and str(pb.get("datos_hasta", "9999")) < str(s["dia"].max()):
                previo = pb["model"]
        except Exception:  # noqa: BLE001
            previo = None

    if args.solo_examen:
        imprimir_examen(est, spy, folds, dias_prueba, dt)
        return

    print("Entrenando el modelo final con todas las rupturas...")
    final = modelo().fit(base[FEATURES_AP], base["y_bps"])
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES_AP, "version": VERSION_DT, "demo": args.demo,
                 "datos_hasta": str(base["dia"].max()),
                 "trained_at": datetime.now(timezone.utc).isoformat()}, MODELS_DIR / "daytrade.joblib")
    plan = plan_del_dia(s, previo if previo is not None else final, dt)
    plan["modelo"] = "anterior" if previo is not None else "nuevo"
    tabla, tbase = tabla_velas(velas, cfg)

    auc = round(float(np.mean([f["auc"] for f in folds])), 4)
    cols = ["dia", "hora_entrada", "ticker", "lado", "entrada", "salida", "neto", "R", "motivo", "espera", "rel_vol"]
    ops_out = []
    if ops_agente is not None and len(ops_agente):
        for _, o in ops_agente.sort_values(["dia", "k_ent"]).tail(30)[cols].iloc[::-1].iterrows():
            ops_out.append({"dia": str(o["dia"]), "hora": o["hora_entrada"], "ticker": o["ticker"],
                            "lado": "COMPRA" if o["lado"] > 0 else "CORTO",
                            "entrada": round(float(o["entrada"]), 2), "salida": round(float(o["salida"]), 2),
                            "resultado": round(float(o["neto"]), 5), "R": round(float(o["R"]), 2),
                            "motivo": o["motivo"], "espera_bps": round(float(o["espera"]), 1),
                            "en_juego": round(float(o["rel_vol"]), 2)})
    out = {
        "version": VERSION_DT, "entrenado": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "datos_demo": args.demo, "dias_datos": len(dias), "desde": str(dias[0]), "hasta": str(dias[-1]),
        "universo": int(s["ticker"].nunique()), "rupturas": int(len(base)), "auc_promedio": auc,
        "examenes": folds, "horizonte_min": dt["horizonte_velas"] * 5,
        "reglas": {k: dt[k] for k in ("en_juego_top", "max_operaciones_dia", "umbral_agente_bps", "stop_atr",
                                      "ultima_entrada", "costo_bps_por_lado")},
        "estrategias": est, "spy": spy,
        "curva": {"fechas": [str(d) for d in dias_prueba],
                  **{k: (1 + v).cumprod().round(5).tolist() for k, v in curvas.items()},
                  "spy": eq_spy.round(5).tolist()},
        "operaciones": ops_out, "plan": plan,
        "velas_patrones": tabla, "velas_base": tbase,
    }
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (WEB_DATA_DIR / "daytrading.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    write_web_bundle()

    # resumen para la terminal
    nombres = {"agente": "Agente (lee las velas)", "regla": f"Regla: {dt['en_juego_top']} más en juego",
               "todas": "Cualquier ruptura"}
    print(f"\nSimulación en {len(dias_prueba)} días que no vio (con comisiones):")
    for k, nombre in nombres.items():
        x = est[k]
        if x["operaciones"]:
            print(f"  {nombre:<26} {x['operaciones']:>4} operaciones · ganadoras {x['ganadoras']:.0%} · "
                  f"promedio {x['promedio_bps']:+6.1f} pb · resultado {x['retorno_total']:+.2%}")
        else:
            print(f"  {nombre:<26} sin operaciones")
    print(f"  {'Mercado (SPY)':<26} resultado {spy['retorno_total']:+.2%}")
    nivel = ("no encontró nada útil (es como echar volado)" if auc < 0.51 else
             "encontró algo, pero muy débil" if auc < 0.53 else "encontró algo interesante")
    print(f"\nCalificación del agente (AUC en días que no vio): {auc:.3f} → {nivel}")
    print(f"\nPlan del {plan['dia']}" + (" (hoy)" if plan.get("es_hoy") else " (último día con velas)") + ":")
    for f in plan["filas"][:6]:
        print(f"  {f['ticker']:<6} en juego {f['en_juego']:.1f}x · {f['lado']:<6} si rompe {f['entrar_en']} "
              f"· stop {f['stop']} · agente: {f['decision']} (espera {f['espera_bps']:+.0f} pb) · {f['estado']}")


if __name__ == "__main__":
    main()
