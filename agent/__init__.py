"""Agente de trading para acciones: datos -> features -> modelo -> backtest -> paper trading."""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
WEB_DATA_DIR = ROOT / "docs" / "data"
WEB_FILES = {"status": "status.json", "backtest": "backtest.json",
             "training": "training.json", "trades": "trades.json",
             "daytrading": "daytrading.json", "historial_dt": "historial_dt.json"}

# Sube este número cuando cambie lo que el modelo aprende: obliga a re-entrenar.
MODEL_VERSION = 2


def load_env() -> None:
    """Lee el archivo .env (si existe) para correr el agente en tu compu."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def load_config(path: Path | None = None) -> dict:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "sectores" in cfg:  # junta las listas por sector en una sola
        cfg["tickers"] = list(dict.fromkeys(t for lista in cfg["sectores"].values() for t in lista))
    return cfg


def modelo_compatible() -> bool:
    """¿Hay un modelo entrenado con datos reales que sirva con el código actual?"""
    path = MODELS_DIR / "model.joblib"
    if not path.exists():
        return False
    try:
        import joblib
        bundle = joblib.load(path)
        return bundle.get("version") == MODEL_VERSION and not bundle.get("demo")
    except Exception:  # noqa: BLE001
        return False


def write_web_bundle() -> None:
    """Junta los .json de la página en docs/data/data.js.

    Así la página abre con doble clic (los navegadores no dejan leer .json
    sueltos desde un archivo local) y también funciona en GitHub Pages.
    """
    bundle = {}
    for key, name in WEB_FILES.items():
        p = WEB_DATA_DIR / name
        if p.exists():
            bundle[key] = json.loads(p.read_text(encoding="utf-8"))
    (WEB_DATA_DIR / "data.js").write_text(
        "window.AGENTE = " + json.dumps(bundle, ensure_ascii=False) + ";\n", encoding="utf-8")
