"""Entrena el agente.

Uso:
    python -m agent.train          # con datos reales (yfinance)
    python -m agent.train --cache  # con los precios que ya bajó la última vez
    python -m agent.train --demo   # con datos sintéticos, para probar

Qué hace:
 1. Baja históricos y calcula indicadores.
 2. Validación walk-forward: entrena con el pasado, prueba en el futuro,
    avanza y repite. Deja un hueco (embargo) para no filtrar información.
 3. Corre el backtest con esas predicciones fuera de muestra.
 4. Entrena el modelo final con todo y lo guarda en models/model.joblib.
 5. Escribe docs/data/training.json y backtest.json para la página.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from . import MODEL_VERSION, MODELS_DIR, WEB_DATA_DIR, load_config, write_web_bundle
from .backtest import run_backtest
from .data import get_data
from .features import FEATURES, build_dataset


def new_model(m: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=m["learning_rate"], max_iter=m["max_iter"], max_depth=m["max_depth"],
        l2_regularization=1.0, min_samples_leaf=200, random_state=42)


def walk_forward(ds: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    m = cfg["model"]
    dates = np.sort(ds["date"].unique())
    n_splits, emb = m["n_splits"], m["horizon_days"]
    # primera mitad solo para entrenar; la segunda se divide en bloques de prueba
    start = len(dates) // 2
    edges = np.linspace(start, len(dates), n_splits + 1, dtype=int)
    preds, folds = [], []
    for i in range(n_splits):
        test_dates = dates[edges[i]:edges[i + 1]]
        train_end = dates[max(edges[i] - emb, 1)]
        tr = ds[ds["date"] < train_end].dropna(subset=FEATURES + ["target"])
        te = ds[ds["date"].isin(test_dates)].dropna(subset=FEATURES)
        model = new_model(m).fit(tr[FEATURES], tr["target"])
        te = te.assign(prob=model.predict_proba(te[FEATURES])[:, 1])
        scored = te.dropna(subset=["target"])
        fold = {
            "fold": i + 1,
            "prueba_desde": pd.Timestamp(test_dates[0]).strftime("%Y-%m-%d"),
            "prueba_hasta": pd.Timestamp(test_dates[-1]).strftime("%Y-%m-%d"),
            "auc": round(float(roc_auc_score(scored["target"], scored["prob"])), 4),
            "accuracy": round(float(accuracy_score(scored["target"], scored["prob"] > 0.5)), 4),
            "tasa_base": round(float(scored["target"].mean()), 4),
        }
        folds.append(fold)
        print(f"  examen {fold['fold']}: {fold['prueba_desde']} → {fold['prueba_hasta']}  "
              f"AUC={fold['auc']:.3f}  aciertos={fold['accuracy']:.1%}")
        preds.append(te[["date", "ticker", "prob", "next_ret"]])
    return pd.concat(preds), folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="usar datos sintéticos")
    ap.add_argument("--cache", action="store_true", help="usar los precios guardados la última vez")
    args = ap.parse_args()

    cfg = load_config()
    data = get_data(cfg, demo=args.demo, cache=args.cache)
    tickers = [t for t in cfg["tickers"] if t in data]
    ds = build_dataset(data, tickers, cfg["benchmark"], cfg["model"]["horizon_days"])
    print(f"Dataset: {len(ds):,} filas, {len(tickers)} acciones")

    print("Validación walk-forward (entrena con el pasado y se examina en años que no vio):")
    preds, folds = walk_forward(ds, cfg)

    print("Backtest con predicciones fuera de muestra...")
    bt = run_backtest(preds, data[cfg["benchmark"]]["Close"], cfg)
    m = bt["metrics"]
    nombres = {"agente": "Agente",
               "iguales": f"Las {bt['universo']} en partes iguales",
               "benchmark": f"Mercado ({cfg['benchmark']})"}
    for k, nombre in nombres.items():
        x = m[k]
        print(f"  {nombre:<28} total {x['retorno_total']:+7.1%}  al año {x['cagr']:+6.1%}  "
              f"Sharpe {x['sharpe']:.2f}  peor caída {x['max_drawdown']:.1%}")

    print("Entrenando modelo final con todos los datos...")
    full = ds.dropna(subset=FEATURES + ["target"])
    model = new_model(cfg["model"]).fit(full[FEATURES], full["target"])
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "version": MODEL_VERSION,
                 "trained_at": datetime.now(timezone.utc).isoformat(),
                 "sklearn": sklearn.__version__,
                 "demo": args.demo}, MODELS_DIR / "model.joblib")

    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    training = {
        "entrenado": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "datos_demo": args.demo,
        "objetivo": f"ganarle al mercado ({cfg['benchmark']}) en {cfg['model']['horizon_days']} días",
        "acciones": tickers,
        "filas": int(len(full)),
        "horizonte_dias": cfg["model"]["horizon_days"],
        "auc_promedio": round(float(np.mean([f["auc"] for f in folds])), 4),
        "folds": folds,
    }
    (WEB_DATA_DIR / "training.json").write_text(json.dumps(training, indent=2, ensure_ascii=False))
    (WEB_DATA_DIR / "backtest.json").write_text(json.dumps(bt, ensure_ascii=False))
    write_web_bundle()

    auc = training["auc_promedio"]
    nivel = ("no encontró nada útil (es como echar volado)" if auc < 0.51 else
             "encontró algo, pero muy débil" if auc < 0.53 else
             "encontró algo interesante; revisa que el backtest lo confirme")
    a, i, b = m["agente"], m["iguales"], m["benchmark"]
    print(f"\nCalificación del agente (AUC en años que no vio): {auc:.3f} → {nivel}")
    print(f"¿Sabe escoger?    agente {a['cagr']:+.1%} al año vs las {bt['universo']} en partes "
          f"iguales {i['cagr']:+.1%} → {'sí, escogió mejor' if a['cagr'] > i['cagr'] else 'no, escoger no ayudó'}")
    print(f"¿Le gana a {cfg['benchmark']}? agente {a['cagr']:+.1%} al año vs {b['cagr']:+.1%} → "
          f"{'sí' if a['cagr'] > b['cagr'] else 'no'}")


if __name__ == "__main__":
    main()
