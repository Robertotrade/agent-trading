"""El agente operando: lee el mercado, decide y manda órdenes a Alpaca.

Uso:
    python -m agent.trade               # opera en tu cuenta DEMO de Alpaca
    python -m agent.trade --dry-run     # solo calcula señales, no manda órdenes
    python -m agent.trade --solo-riesgo # solo revisa stop loss y freno diario
    python -m agent.trade --demo        # datos sintéticos, sin broker (prueba)

Variables de entorno (en GitHub van como Secrets):
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER=true

Qué hace, en este orden:
 1. Freno diario: si la cuenta perdió más de max_daily_loss_pct hoy, cierra todo
    y espera hasta el siguiente rebalanceo (una semana).
 2. Stop loss: vende cualquier acción que vaya perdiendo más de stop_loss_pct.
 3. Rebalanceo semanal: se queda con las top_n acciones con más probabilidad de
    ganarle al mercado, en partes iguales (invest_pct del capital en total).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import date, datetime, timezone

import joblib
import pandas as pd

from . import MODEL_VERSION, MODELS_DIR, WEB_DATA_DIR, load_config, load_env, write_web_bundle
from .data import demo_data, download
from .features import BASE_FEATURES, add_cross_sectional, make_features

STATE = WEB_DATA_DIR / "estado.json"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def to_alpaca(t: str) -> str:   # Yahoo usa BRK-B, Alpaca usa BRK.B
    return t.replace("-", ".")


def from_alpaca(t: str) -> str:
    return t.replace(".", "-")


def compute_signals(cfg: dict, demo: bool) -> pd.DataFrame:
    path = MODELS_DIR / "model.joblib"
    if not path.exists():
        raise SystemExit("No hay modelo entrenado. Corre primero: python -m agent.train")
    bundle = joblib.load(path)
    if bundle.get("version") != MODEL_VERSION:
        raise SystemExit("El modelo es de una versión anterior del agente. "
                         "Vuelve a entrenarlo: python -m agent.train")
    if bundle.get("demo") and not demo:
        raise SystemExit("ALTO: el modelo se entrenó con datos DEMO sintéticos. "
                         "Entrénalo con datos reales (python -m agent.train) antes de operar.")
    model, feats = bundle["model"], bundle["features"]
    tickers = list(dict.fromkeys(cfg["tickers"] + [cfg["benchmark"]]))
    start = (pd.Timestamp.today() - pd.Timedelta(days=600)).strftime("%Y-%m-%d")
    data = demo_data(tickers, "2012-01-01") if demo else download(tickers, start)
    if cfg["benchmark"] not in data:
        raise SystemExit(f"No pude bajar {cfg['benchmark']} (el mercado de referencia). "
                         "Inténtalo en unos minutos.")

    rows = []
    for t in cfg["tickers"]:
        if t not in data:
            continue
        f = make_features(data[t], data[cfg["benchmark"]]).dropna(subset=BASE_FEATURES)
        if f.empty:
            continue
        rows.append({**f.iloc[-1].to_dict(), "ticker": t, "fecha": f.index[-1],
                     "precio": float(data[t]["Close"].iloc[-1])})
    lat = pd.DataFrame(rows)
    # solo acciones con datos recientes, y todas se comparan como si fueran el mismo día
    lat = lat[lat["fecha"] >= lat["fecha"].max() - pd.Timedelta(days=4)].copy()
    lat["date"] = lat["fecha"].max()
    lat = add_cross_sectional(lat)
    lat["prob"] = model.predict_proba(lat[feats])[:, 1]
    lat["fecha_datos"] = lat["fecha"].dt.strftime("%Y-%m-%d")
    sig = (lat[["ticker", "prob", "precio", "fecha_datos"]]
           .sort_values("prob", ascending=False).reset_index(drop=True))
    top = set(sig["ticker"].head(cfg["strategy"]["top_n"]))
    sig["senal"] = sig["ticker"].apply(lambda t: "COMPRA" if t in top else "NADA")
    return sig


def get_client(cfg: dict):
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        return None, "sin_broker"
    paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
    if not paper and not cfg["broker"].get("live_trading_confirmed"):
        raise SystemExit("ALTO: ALPACA_PAPER=false pero live_trading_confirmed no está en true. "
                         "No se opera con dinero real.")
    from alpaca.trading.client import TradingClient
    return TradingClient(key, secret, paper=paper), ("paper" if paper else "REAL")


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def execute(client, cfg: dict, sig: pd.DataFrame, dry_run: bool,
            risk_only: bool = False) -> tuple[dict, list, list[dict]]:
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    risk, s = cfg["risk"], cfg["strategy"]
    acct = client.get_account()
    equity, last_eq = float(acct.equity), float(acct.last_equity)
    day_pl = equity / last_eq - 1 if last_eq else 0.0
    market_open = client.get_clock().is_open
    positions = {from_alpaca(p.symbol): p for p in client.get_all_positions()}
    trades: list[dict] = []
    state = load_state()
    last = state.get("ultimo_rebalanceo")
    days_since = (date.today() - date.fromisoformat(last)).days if last else 999
    rebalance_due = days_since >= math.ceil(s["rebalance_days"] * 7 / 5)

    def log(ticker, lado, motivo, monto=None):
        trades.append({"fecha": now(), "ticker": ticker, "lado": lado,
                       "motivo": motivo, "monto_usd": monto, "simulada": dry_run})

    def sell(sym, motivo):
        log(sym, "VENTA", motivo, float(positions[sym].market_value))
        if not dry_run:
            client.close_position(to_alpaca(sym))
        positions.pop(sym)

    kill = day_pl <= -risk["max_daily_loss_pct"]
    if not market_open:
        print("Mercado cerrado: no mando órdenes.")
    elif kill:
        print(f"FRENO DIARIO: la cuenta va {day_pl:.2%} hoy. Cierro todo y espero una semana.")
        for sym in list(positions):
            log(sym, "VENTA", "freno diario (pérdida máxima)")
        if not dry_run:
            client.close_all_positions(cancel_orders=True)
            state["ultimo_rebalanceo"] = date.today().isoformat()
        positions = {}
    else:
        stopped = set()
        for sym, p in list(positions.items()):
            plpc = float(p.unrealized_plpc)
            if plpc <= -risk["stop_loss_pct"]:
                sell(sym, f"stop loss ({plpc:.1%})")
                stopped.add(sym)   # no la vuelve a comprar hoy
        if risk_only or not rebalance_due:
            print("Hoy no toca rebalancear: solo reviso el riesgo."
                  + (f" Último rebalanceo: {last}." if last else ""))
        else:
            wanted = list(sig.loc[sig["senal"] == "COMPRA", "ticker"])
            for sym in [x for x in positions if x not in wanted]:
                sell(sym, f"ya no está entre las {s['top_n']} mejores")
            notional = round(equity * risk["invest_pct"] / s["top_n"], 2)
            for sym in [x for x in wanted if x not in positions and x not in stopped]:
                log(sym, "COMPRA", f"entre las {s['top_n']} mejores", notional)
                if not dry_run:
                    client.submit_order(MarketOrderRequest(
                        symbol=to_alpaca(sym), notional=notional,
                        side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
            if not dry_run:
                state["ultimo_rebalanceo"] = date.today().isoformat()
    if not dry_run:
        STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    account = {"capital": round(equity, 2), "efectivo": round(float(acct.cash), 2),
               "pl_dia_pct": round(day_pl, 4), "mercado_abierto": market_open,
               "kill_switch": kill, "ultimo_rebalanceo": state.get("ultimo_rebalanceo")}
    pos_out = [{"ticker": t, "cantidad": float(p.qty), "precio_entrada": float(p.avg_entry_price),
                "precio_actual": float(p.current_price), "pl_pct": round(float(p.unrealized_plpc), 4),
                "valor": round(float(p.market_value), 2)} for t, p in positions.items()]
    return account, pos_out, trades


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no mandar órdenes")
    ap.add_argument("--demo", action="store_true", help="datos sintéticos, sin broker")
    ap.add_argument("--solo-riesgo", action="store_true",
                    help="solo stop loss y freno diario; no compra ni rebalancea")
    args = ap.parse_args()
    load_env()
    cfg = load_config()

    sig = compute_signals(cfg, demo=args.demo)
    top_n = cfg["strategy"]["top_n"]
    print(f"Las {top_n} con más probabilidad de ganarle al mercado (de {len(sig)}):")
    print(sig.head(top_n).to_string(index=False))

    client, mode = (None, "demo") if args.demo else get_client(cfg)
    account, positions, trades = None, [], []
    if client is not None:
        try:
            account, positions, trades = execute(client, cfg, sig, dry_run=args.dry_run,
                                                 risk_only=args.solo_riesgo)
        except Exception as e:  # noqa: BLE001
            if "401" not in str(e) and "403" not in str(e) and "nauthorized" not in str(e):
                raise
            print(f"Alpaca rechazó las llaves ({str(e)[:90]}). Solo publico señales, sin órdenes.")
            client, mode = None, "sin_broker"
    if client is None:
        print(f"Modo {mode}: solo señales, sin órdenes.")

    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    status = {"actualizado": now(), "modo": mode, "simulacion": args.dry_run or client is None,
              "top_n": top_n, "rebalanceo_dias": cfg["strategy"]["rebalance_days"],
              "cuenta": account, "posiciones": positions,
              "senales": sig.round(4).to_dict(orient="records")}
    (WEB_DATA_DIR / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False))

    for t in trades:
        print(f"  {'(simulada) ' if t['simulada'] else ''}{t['lado']} {t['ticker']}: {t['motivo']}")
    path = WEB_DATA_DIR / "trades.json"
    history = json.loads(path.read_text()) if path.exists() else []
    if not args.dry_run:   # las simulaciones no se guardan en el historial
        history = (trades + history)[:500]
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False))

    if account:
        eq_path = WEB_DATA_DIR / "equity_live.json"
        eq = json.loads(eq_path.read_text()) if eq_path.exists() else []
        eq.append({"fecha": now(), "capital": account["capital"]})
        eq_path.write_text(json.dumps(eq[-2000:], ensure_ascii=False))
    write_web_bundle()
    print(f"Órdenes {'simuladas' if args.dry_run or client is None else 'enviadas'}: {len(trades)}. "
          "Estado guardado para la página.")


if __name__ == "__main__":
    main()
