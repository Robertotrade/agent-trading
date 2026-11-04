"""Day trading: QUÉ acciones comprar y CUÁNDO comprarlas.

QUÉ: las acciones "en juego", las que en sus primeros 5 minutos traen mucho más
volumen de lo normal (casi siempre por una noticia). Ahí está el movimiento del día.

CUÁNDO: ruptura de la primera vela de 5 minutos (la idea de un estudio de 2016-2023):
 - si la primera vela es verde, compra cuando el precio rompa su máximo;
 - si es roja, vende en corto cuando rompa su mínimo;
 - stop a 10% del rango diario promedio (ATR); si no lo toca, sale al cierre.

El AGENTE lee las velas y la gráfica justo antes de la ruptura (solo velas ya
cerradas, nunca el futuro) y decide si la toma o la deja pasar.
"""
from __future__ import annotations

from datetime import time as hora

import numpy as np
import pandas as pd

from .data import COLS

ETFS = {"SPY", "QQQ", "IWM", "DIA"}

FEATURES_AP = [
    # qué tan "en juego" está
    "rel_vol", "rel_vol_rank", "vol_a_ruptura_rel",
    # cómo abrió y cómo fue la primera vela
    "lado", "gap_a_favor", "gap_atr", "primera_cuerpo", "primera_rango_atr",
    "primera_mecha_favor", "primera_mecha_contra",
    # las velas justo antes de la ruptura
    "previa_cuerpo_a_favor", "previa_cuerpo_rel", "previa_rango_atr", "previa2_cuerpo_a_favor",
    "dist_vwap_a_favor", "dist_nivel_atr", "minutos_a_ruptura",
    # tendencia de días anteriores y el mercado
    "ret_5d_a_favor", "ret_20d_a_favor", "dist_ma20_a_favor", "spy_primera_a_favor", "spy_gap_a_favor",
    # tipo de acción
    "atr_pct", "precio_log", "vol_prom_log", "dia_semana",
]


def _minutos(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)


def resumen_diario(df: pd.DataFrame) -> pd.DataFrame:
    """Un renglón por día: precios del día, primera vela, ATR y volumen relativo."""
    dia = pd.Series(df.index.date, index=df.index)
    g = df.groupby(dia)
    d = pd.DataFrame({"o": g["Open"].first(), "h": g["High"].max(), "l": g["Low"].min(),
                      "c": g["Close"].last(), "v": g["Volume"].sum(), "n": g["Close"].size(),
                      "h1": g["High"].first(), "l1": g["Low"].first(), "c1": g["Close"].first(),
                      "v1": g["Volume"].first()})
    d["empieza_bien"] = pd.Series(df.index.time, index=df.index).groupby(dia).first() == hora(9, 30)
    pc = d["c"].shift(1)
    tr = pd.concat([d["h"] - d["l"], (d["h"] - pc).abs(), (d["l"] - pc).abs()], axis=1).max(axis=1)
    d["pc"] = pc
    d["atr"] = tr.rolling(14).mean().shift(1)                  # sin contar el día de hoy
    d["vol_prom"] = d["v"].rolling(14).mean().shift(1)
    d["rel_vol"] = d["v1"] / d["v1"].rolling(14).mean().shift(1)
    d["gap"] = d["o"] / pc - 1
    d["ret_5d"] = pc / d["c"].shift(6) - 1
    d["ret_20d"] = pc / d["c"].shift(21) - 1
    d["dist_ma20"] = pc / d["c"].rolling(20).mean().shift(1) - 1
    return d


def setups(velas: dict[str, pd.DataFrame], dt: dict, mercado: str) -> pd.DataFrame:
    """Todas las oportunidades (acción, día) con su ruptura, resultado y lectura de velas."""
    spy = resumen_diario(velas[mercado])
    spy_r1 = (spy["c1"] / spy["o"] - 1)
    corte, costo, k_atr = _minutos(dt["ultima_entrada"]), dt["costo_bps_por_lado"] / 1e4, dt["stop_atr"]
    filas = []
    for t, df in velas.items():
        if t in ETFS or len(df) < 200:
            continue
        d = resumen_diario(df)
        O, H, L, C, V = (df[k].to_numpy(dtype=float) for k in COLS)
        TP = (H + L + C) / 3
        minutos = (df.index.hour * 60 + df.index.minute).to_numpy()
        dias, inicios, cuantas = np.unique(np.array(df.index.date), return_index=True, return_counts=True)
        for dia, i0, n in zip(dias, inicios, cuantas):
            r = d.loc[dia]
            if not r["empieza_bien"] or not (r["atr"] > 0) or not np.isfinite(r["rel_vol"]):
                continue
            o1, h1, l1, c1 = O[i0], H[i0], L[i0], C[i0]
            lado = 1 if c1 > o1 else -1 if c1 < o1 else 0
            if lado == 0:
                continue
            atr, nivel = r["atr"], (h1 if lado > 0 else l1)
            k_ent = next((k for k in range(1, n) if minutos[i0 + k] <= corte and
                          ((lado > 0 and H[i0 + k] >= nivel) or (lado < 0 and L[i0 + k] <= nivel))), None)
            prev = (k_ent if k_ent is not None else n) - 1      # última vela cerrada antes de entrar
            j = i0 + prev
            vw = (TP[i0:j + 1] * V[i0:j + 1]).sum() / max(V[i0:j + 1].sum(), 1e-9)
            rango1 = max(h1 - l1, 1e-9)
            mecha_arr, mecha_ab = (h1 - max(o1, c1)) / rango1, (min(o1, c1) - l1) / rango1
            f = {
                "dia": dia, "ticker": t, "lado": lado, "nivel": nivel, "atr": atr,
                "stop": nivel - lado * k_atr * atr, "pc": r["pc"], "vol_prom": r["vol_prom"],
                "completo": n >= 75, "ultima_vela": df.index[i0 + n - 1],
                "rel_vol": r["rel_vol"],
                "vol_a_ruptura_rel": V[i0:j + 1].sum() / max(r["vol_prom"] / 78 * (prev + 1), 1e-9),
                "gap_a_favor": r["gap"] * lado, "gap_atr": (r["o"] - r["pc"]) / atr * lado,
                "primera_cuerpo": abs(c1 - o1) / rango1, "primera_rango_atr": (h1 - l1) / atr,
                "primera_mecha_favor": mecha_ab if lado > 0 else mecha_arr,
                "primera_mecha_contra": mecha_arr if lado > 0 else mecha_ab,
                "previa_cuerpo_a_favor": (C[j] / O[j] - 1) * lado,
                "previa_cuerpo_rel": abs(C[j] - O[j]) / max(H[j] - L[j], 1e-9),
                "previa_rango_atr": (H[j] - L[j]) / atr,
                "previa2_cuerpo_a_favor": (C[j - 1] / O[j - 1] - 1) * lado if prev >= 1 else np.nan,
                "dist_vwap_a_favor": (C[j] / vw - 1) * lado, "dist_nivel_atr": (C[j] - nivel) / atr * lado,
                "minutos_a_ruptura": (prev + 1) * 5,
                "ret_5d_a_favor": r["ret_5d"] * lado, "ret_20d_a_favor": r["ret_20d"] * lado,
                "dist_ma20_a_favor": r["dist_ma20"] * lado,
                "spy_primera_a_favor": spy_r1.get(dia, np.nan) * lado,
                "spy_gap_a_favor": spy["gap"].get(dia, np.nan) * lado,
                "atr_pct": atr / r["pc"], "precio_log": np.log(r["pc"]),
                "vol_prom_log": np.log(max(r["vol_prom"], 1.0)), "dia_semana": pd.Timestamp(dia).weekday(),
                "k_ent": np.nan, "hora_entrada": None, "entrada": np.nan, "salida": np.nan,
                "neto": np.nan, "bruto": np.nan, "R": np.nan, "motivo": None,
            }
            if k_ent is not None:
                e = i0 + k_ent
                abre_pasado = (O[e] >= nivel) if lado > 0 else (O[e] <= nivel)
                entrada = O[e] if abre_pasado else nivel
                stop = entrada - lado * k_atr * atr
                # Vela de entrada: con velas de 5 min no se sabe si el precio tocó el stop antes o
                # después de entrar. Si abrió ya pasada la ruptura, fue después (cuenta como stop).
                # Si cerró del otro lado del stop, también. Si solo lo tocó con la mecha, no se
                # sabe: cuenta mitad y mitad (ni optimista ni pesimista).
                toca = (L[e] <= stop) if lado > 0 else (H[e] >= stop)
                cierra_tras_stop = (C[e] <= stop) if lado > 0 else (C[e] >= stop)
                peso_stop = 1.0 if (toca and (abre_pasado or cierra_tras_stop)) else 0.5 if toca else 0.0
                salida_sigue, motivo = C[i0 + n - 1], "cierre"
                for k in range(e + 1, i0 + n):
                    if (lado > 0 and L[k] <= stop) or (lado < 0 and H[k] >= stop):
                        salida_sigue, motivo = stop, "stop"
                        break
                salida = peso_stop * stop + (1 - peso_stop) * salida_sigue
                motivo = "stop" if peso_stop == 1 else "dudoso" if peso_stop else motivo
                # costo por lado: % fijo + medio centavo por acción (pesa más en acciones baratas)
                costo_lado = costo + dt.get("costo_por_accion", 0.0) / entrada
                bruto = (salida / entrada - 1) * lado
                f.update({"k_ent": k_ent, "hora_entrada": df.index[e].strftime("%H:%M"),
                          "entrada": entrada, "stop": stop, "salida": salida, "bruto": bruto,
                          "neto": bruto - 2 * costo_lado, "R": (salida - entrada) * lado / (k_atr * atr),
                          "motivo": motivo})
            filas.append(f)
    s = pd.DataFrame(filas)
    if s.empty:
        return s
    s["rel_vol_rank"] = s.groupby("dia")["rel_vol"].rank(pct=True)
    # filtros del estudio: precio > $5, ATR > $0.50, volumen diario promedio > 1 millón
    s["pasa_filtros"] = (s["pc"] > 5) & (s["atr"] > 0.5) & (s["vol_prom"] > 1e6)
    s["target"] = np.where(s["neto"].notna(), (s["neto"] > 0).astype(float), np.nan)
    s["y_bps"] = (s["neto"] * 1e4).clip(-150, 1000)       # lo que el agente aprende a predecir
    return s


def elegir(s: pd.DataFrame, dias: list, modo: str, dt: dict) -> pd.DataFrame:
    """Qué operaciones habría tomado cada día, decidiendo en orden de hora (sin ver el futuro).

    modo = "agente": de las N más en juego, solo las rupturas donde el agente espera ganar
                     (después de comisiones) al leer las velas.
    modo = "regla":  todas las rupturas de las N más en juego (el estudio, sin agente).
    modo = "todas":  cualquier ruptura, la primera que salga.
    """
    elegidas = []
    for d in dias:
        dia = s[(s["dia"] == d) & s["pasa_filtros"]]
        if modo in ("regla", "agente"):     # primero QUÉ acciones: las más en juego al abrir
            dia = dia[dia["rel_vol"] >= 1].nlargest(dt["en_juego_top"], "rel_vol")
        tomadas = dia[dia["entrada"].notna() & dia["completo"]]
        if modo == "agente":
            tomadas = tomadas[tomadas["espera"] >= dt["umbral_agente_bps"]]
        orden = ["k_ent", "espera" if modo == "agente" else "rel_vol"]
        elegidas.append(tomadas.sort_values(orden, ascending=[True, False]).head(dt["max_operaciones_dia"]))
    return pd.concat(elegidas) if elegidas else s.iloc[0:0]


def resultados(ops: pd.DataFrame, dias: list, dt: dict) -> tuple[dict, pd.Series]:
    w = 1.0 / dt["max_operaciones_dia"]      # cada operación usa la misma parte del capital
    diario = pd.Series(0.0, index=pd.Index(dias))
    if len(ops):
        diario = diario.add(ops.groupby("dia")["neto"].sum() * w, fill_value=0.0).reindex(dias).fillna(0.0)
    eq = (1 + diario).cumprod()
    sd = diario.std()
    m = {"operaciones": int(len(ops)), "dias": len(dias),
         "por_dia": round(len(ops) / max(len(dias), 1), 1),
         "retorno_total": round(float(eq.iloc[-1] - 1), 4) if len(eq) else 0.0,
         "sharpe": round(float(diario.mean() / sd * np.sqrt(252)), 2) if sd > 0 else 0.0,
         "peor_caida": round(float((eq / eq.cummax() - 1).min()), 4) if len(eq) else 0.0}
    if len(ops):
        r = ops["neto"].to_numpy()
        perd = -r[r < 0].sum()
        m.update({"ganadoras": round(float((r > 0).mean()), 4),
                  "promedio_bps": round(float(r.mean() * 1e4), 2),
                  "promedio_bruto_bps": round(float(ops["bruto"].mean() * 1e4), 2),
                  "R_promedio": round(float(ops["R"].mean()), 3),
                  "factor_ganancia": round(float(r[r > 0].sum() / perd), 2) if perd > 0 else None,
                  "stops": int((ops["motivo"] == "stop").sum())})
    return m, diario
