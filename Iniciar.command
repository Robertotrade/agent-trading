#!/bin/bash
# =============================================================
#  Agente de Trading: haz DOBLE CLIC en este archivo (Mac).
#   - La 1ª vez prepara todo (incluso Python, si no lo tienes)
#     y entrena al agente.
#   - Después te deja ver las señales de hoy o volver a entrenarlo.
#  No manda órdenes de compra/venta: solo calcula y te enseña.
# =============================================================
cd "$(dirname "$0")" || exit 1
VENV="$HOME/.venvs/agente-trading"
PY="$VENV/bin/python"
UV_DIR="$HOME/.venvs/uv"

pausa() { echo; read -r -p "Presiona Enter para cerrar esta ventana... " _; }
falla() { echo; echo "❌ $1"; pausa; exit 1; }

# Un Python 3.10–3.14 que de verdad funcione
python_ok() {
  [ -n "$1" ] && [ -x "$1" ] &&
    "$1" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 14))' >/dev/null 2>&1
}

buscar_uv() {
  for c in "$UV_DIR/uv" "$UV_DIR/bin/uv"; do
    [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

entorno_ok() { [ -x "$PY" ] && "$PY" -c 'import sys' >/dev/null 2>&1; }

crear_entorno() {
  rm -rf "$VENV"
  # a) Un Python que ya tengas (de python.org o Homebrew). Nunca el "python3" de Apple:
  #    sin las herramientas de desarrollador abre un aviso para instalarlas.
  candidatos="/usr/local/bin/python3 /opt/homebrew/bin/python3"
  candidatos="$candidatos /Library/Frameworks/Python.framework/Versions/Current/bin/python3"
  [ "$(uname)" != "Darwin" ] && candidatos="$candidatos $(command -v python3)"
  for p in $candidatos; do
    if python_ok "$p"; then
      echo "   Usando tu Python: $p"
      "$p" -m venv "$VENV" && return 0
    fi
  done
  # b) Si no hay, bajo un Python solo para el agente con uv (herramienta gratuita de Astral).
  #    Queda guardado en tu carpeta de usuario y no cambia nada más de tu Mac.
  echo "   No encontré Python en tu Mac: bajo uno solo para el agente..."
  UV="$(buscar_uv)"
  if [ -z "$UV" ]; then
    curl -LsSf https://astral.sh/uv/install.sh \
      | env UV_INSTALL_DIR="$UV_DIR" UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1
    UV="$(buscar_uv)" || return 1
  fi
  UV_PYTHON_PREFERENCE=only-managed "$UV" venv --quiet --seed --python 3.12 "$VENV"
}

clear 2>/dev/null
echo "==============================================="
echo "   🤖  Agente de Trading"
echo "==============================================="
echo

# 1) Python y librerías (solo la primera vez, o si cambia requirements.txt)
if ! entorno_ok || ! cmp -s requirements.txt "$VENV/requirements.txt"; then
  echo "📦 Preparando todo (5 a 10 minutos, solo pasa la primera vez)..."
  if ! entorno_ok; then
    crear_entorno || falla "No pude preparar Python. Revisa tu internet y vuelve a intentarlo."
  fi
  "$PY" -m pip install -q --upgrade pip >/dev/null 2>&1
  NO_SQLITE=1 "$PY" -m pip install -q -r requirements.txt \
    || falla "No se pudieron instalar las librerías. Revisa tu internet y vuelve a intentarlo."
  cp requirements.txt "$VENV/requirements.txt"
  echo "   Listo."
  echo
fi

# Certificados para conexiones seguras (algunos Python no los traen)
CERT="$("$PY" -c 'import certifi; print(certifi.where())' 2>/dev/null)"
[ -n "$CERT" ] && export SSL_CERT_FILE="$CERT"

# 2) ¿Entrenar? (siempre, si no hay modelo o si el agente cambió desde la última vez)
entrenar=0
if ! "$PY" -c 'import sys, agent; sys.exit(0 if agent.modelo_compatible() else 1)' >/dev/null 2>&1; then
  if [ -f models/model.joblib ]; then
    echo "El agente cambió desde la última vez: hay que entrenarlo de nuevo."
  else
    echo "Es la primera vez: voy a entrenar al agente."
  fi
  entrenar=1
else
  echo "¿Qué quieres hacer?"
  echo "  [1] Ver la página con las señales de hoy   (solo presiona Enter)"
  echo "  [2] Entrenar al agente semanal otra vez con datos nuevos"
  echo "  [3] Day trading: qué acciones comprar hoy y cuándo (después de las 8:36 am)"
  read -r -p "Escribe 1, 2 o 3: " opcion
  [ "$opcion" = "2" ] && entrenar=1
fi
echo

# Day trading (opción 3)
if [ "$opcion" = "3" ]; then
  echo "🕯️  Bajando velas de 5 minutos de ~110 acciones y entrenando al agente de day trading (2 a 5 minutos)..."
  echo
  "$PY" -m agent.daytrade \
    || falla "El entrenamiento de day trading falló. Copia el mensaje de arriba y pásaselo a Claude."
  echo
  echo "✅ Listo. Abriendo la página de day trading (si ya la tenías abierta, recárgala con Cmd + R)..."
  open docs/daytrading.html 2>/dev/null || echo "Abre el archivo docs/daytrading.html en tu navegador."
  pausa
  exit 0
fi

if [ "$entrenar" = 1 ]; then
  echo "🧠 Entrenando con datos reales de 100 acciones (2 a 5 minutos)..."
  echo
  "$PY" -m agent.train \
    || falla "El entrenamiento falló. Copia el mensaje de arriba y pásaselo a Claude."
  echo
fi

# 3) Señales de hoy (no manda órdenes)
echo "📈 Calculando las señales de hoy..."
echo
"$PY" -m agent.trade --dry-run \
  || falla "No pude calcular las señales. Copia el mensaje de arriba y pásaselo a Claude."

# 4) Abrir la página
echo
echo "✅ Listo. Abriendo la página (si ya la tenías abierta, recárgala con Cmd + R)..."
open docs/index.html 2>/dev/null || echo "Abre el archivo docs/index.html en tu navegador."
pausa
