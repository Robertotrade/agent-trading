# 🤖 Agente de Trading de Acciones

**Página pública:** https://robertotrade.github.io/agent-trading/ ·
**Day trading e historial en vivo:** https://robertotrade.github.io/agent-trading/daytrading.html

Un agente que **aprende de datos históricos** del mercado de acciones de EE. UU.,
se **prueba solo** con backtesting honesto, **opera en una cuenta demo** de Alpaca
y publica todo en una **página web** (GitHub Pages).

> ⚠️ **Proyecto educativo.** Ningún modelo garantiza ganancias. El agente puede
> perder dinero. Úsalo en cuenta demo (paper) durante meses antes de pensar en
> dinero real, y nunca metas dinero que no puedas perder.

## Cómo funciona

```
 Yahoo Finance ──► indicadores ──► modelo (Gradient Boosting) ──► probabilidad de ganarle al mercado
                    (25 features)          │                              │
                                   walk-forward + backtest          reglas de riesgo
                                           │                              │
                                     docs/data/*.json  ◄──────────  órdenes en Alpaca (demo)
                                           │
                                     página web (GitHub Pages)
```

1. **Datos** (`agent/data.py`): precios diarios de 100 empresas grandes de EE. UU. + el
   S&P 500 (SPY). Guarda una copia en `data/precios.csv.gz`.
2. **Indicadores** (`agent/features.py`): momentum, RSI, MACD, volatilidad, distancia a
   medias móviles, volumen, fuerza contra el mercado y **cómo va cada acción comparada
   con las otras 99 ese mismo día**.
3. **Entrenamiento** (`agent/train.py`): el modelo aprende a predecir si cada acción **le
   gana al mercado** en los próximos 5 días. Se valida **walk-forward**: entrena con el
   pasado, se examina en años que nunca vio, avanza y repite.
4. **Backtest** (`agent/backtest.py`): cada semana compra las 10 con más probabilidad, en
   partes iguales, con costos. Lo compara contra el mercado (SPY) y contra **comprar las
   100 en partes iguales**, que es la prueba justa de si sabe escoger.
5. **Operar** (`agent/trade.py`): cada día hábil revisa el riesgo y una vez por semana
   cambia la cartera en Alpaca.
6. **Página** (`docs/index.html`): capital, señales, posiciones, operaciones y el backtest.

### Reglas de seguridad (en `config.yaml`)
| Regla | Valor por defecto |
|---|---|
| Cartera | Las 10 mejores, 9.5% del capital cada una (95% invertido) |
| Cambio de cartera | Una vez por semana |
| Stop loss por acción | -10% → la vende |
| Freno diario (pérdida de la cuenta en un día) | -5% → cierra todo y espera una semana |
| Dinero real | **Bloqueado** (requiere 2 confirmaciones) |

## Puesta en marcha

### 0. Lo más fácil (Mac): doble clic
Haz doble clic en **`Iniciar.command`**. La primera vez prepara todo (si tu Mac no tiene
Python, baja uno solo para el agente) y lo entrena (5 a 10 minutos); después te deja ver las señales de hoy o volver a entrenarlo. Al final
abre la página sola. No manda órdenes.

Si tu Mac no te deja abrirlo, abre la app **Terminal** y pega esto:
```bash
bash ~/Desktop/agente-trading/Iniciar.command
```

### 1. Probar en tu compu (sin cuentas ni internet de mercado)
```bash
pip install -r requirements.txt
python -m agent.train --demo     # entrena con datos sintéticos
python -m agent.trade --demo     # calcula señales
python -m http.server -d docs 8000   # abre http://localhost:8000
```

### 2. Entrenar con datos reales
```bash
python -m agent.train
```
Fíjate en el **AUC** (0.50 = azar; arriba de 0.53 sostenido ya es algo) y, sobre todo,
en si el agente le gana a **las 100 en partes iguales**: esa es la prueba de que sabe escoger.
Para re-entrenar sin volver a bajar precios: `python -m agent.train --cache`.

### 3. Cuenta demo de Alpaca (gratis)
1. Crea cuenta en [alpaca.markets](https://alpaca.markets) → **Paper Trading** → genera API Keys.
2. Copia `.env.example` como `.env` y pega tus llaves.
3. `python -m agent.trade --dry-run` (ve qué haría sin mandar órdenes), luego `python -m agent.trade`.

### 4. Subirlo a GitHub y dejarlo operando solo
> Lo más fácil: pídele a Claude que lo suba por ti. Si lo haces a mano, primero mueve
> los archivos de `github-actions/` a una carpeta nueva llamada `.github/workflows/`.

1. Crea un repo nuevo y sube esta carpeta:
   ```bash
   git init && git add . && git commit -m "Agente de trading"
   git branch -M main
   git remote add origin https://github.com/Robertotrade/agent-trading.git
   git push -u origin main
   ```
2. **Settings → Secrets and variables → Actions → New repository secret**:
   `ALPACA_API_KEY` y `ALPACA_SECRET_KEY`.
3. **Settings → Pages** → Source: *Deploy from a branch* → `main` / carpeta `/docs`.
4. **Actions** → corre **"Entrenar agente"** una vez a mano (botón *Run workflow*).

Listo: el agente re-entrena cada sábado, opera de lunes a viernes y la página se
actualiza sola en `https://robertotrade.github.io/agent-trading/`.

| Cuándo | Qué hace |
|---|---|
| Sábado | `Entrenar agente`: baja datos nuevos, valida, re-entrena y actualiza el backtest |
| Lun–Vie en la mañana | `Agente operando`: revisa señales, vende lo que ya no sirve y compra lo mejor |
| Lun–Vie antes del cierre | `Agente operando --solo-riesgo`: solo stop loss y kill switch |

> El agente se niega a operar si el modelo se entrenó con datos demo, y se niega a usar
> dinero real a menos que cambies **dos** cosas a propósito (`ALPACA_PAPER=false` y
> `live_trading_confirmed: true`).

> 🔒 Nunca subas tu `.env` ni pongas llaves dentro del código. Si el repo es público,
> cualquiera verá tus señales y operaciones (no tus llaves, que van en Secrets).

## Agente de day trading (velas de 5 minutos)

Un segundo agente, aparte del semanal. En `Iniciar.command` es la **opción 3**, o:
`python -m agent.daytrade` (`--cache` para no volver a bajar velas). No manda órdenes.

1. **Qué acciones:** cada mañana ordena ~110 acciones por el volumen de sus primeros
   5 minutos contra lo normal y se fija en las 10 más **en juego**.
2. **Cuándo:** si la primera vela fue verde, entra en compra cuando el precio rompe su
   máximo; si fue roja, en venta en corto cuando rompe su mínimo. Stop a 10% del ATR;
   si no lo toca, sale al cierre (la idea viene de un estudio de 2016-2023).
3. **Leer las velas:** justo antes de cada ruptura el agente lee la primera vela, las
   velas previas, el VWAP, el hueco de apertura, el volumen y el mercado, y calcula
   cuánto espera ganar. Solo toma las que valen la pena después de comisiones.
4. **Examen:** entrena con días anteriores y se prueba en días que no vio. Se compara
   contra la regla sin agente y contra SPY. Si no le gana a la regla, leer las velas no
   está ayudando.
5. **Plan del día:** la página `docs/daytrading.html` dice qué acciones vigilar, en qué
   precio entrar, dónde va el stop y si el agente la tomaría. Para el plan de hoy, córrelo
   después de las 9:36 de Nueva York.

- **¿Sirven las velas de los libros?:** la página también muestra qué pasó de verdad
  después de cada patrón clásico (martillo, envolventes, doji...).
- **Datos:** Yahoo da gratis solo los últimos 60 días de velas de 5 minutos; cada vez que
  lo entrenas, las nuevas se suman a `data/velas_5m.csv.gz`. Para años de historia hace
  falta otra fuente (por ejemplo, Alpaca).
- **Simulación honesta:** decide en orden de hora sin ver el futuro, cobra comisiones y
  medio centavo por acción, y si con velas de 5 minutos no se sabe si el stop se tocó
  antes o después de entrar, cuenta mitad y mitad.
- Reglas en la sección `day_trading` de `config.yaml`.

### Historial en vivo (público)
El examen con días pasados sirve para aprender, pero la prueba de verdad es el historial en vivo:
- Cada tarde (`python -m agent.historial`, en GitHub corre solo) se evalúa el día con el modelo
  que **ya estaba guardado antes de que abriera la bolsa**, y se anotan sus operaciones y resultados
  en `docs/data/historial_dt.json`. Después se re-entrena para el día siguiente.
- Los días solo se agregan, nunca se borran. En GitHub cada anotación queda con fecha y hora en el
  historial del repositorio: nadie puede cambiar los días malos. Eso es lo que da confianza.
- Lo ves en la sección "Historial en vivo" de `docs/daytrading.html`.

## Cómo hacerlo más "carbón"
- **Más datos:** datos fundamentales (ganancias, deuda), sector de cada empresa, noticias.
- **Ojo con el sesgo de supervivencia:** las 100 acciones son las grandes de *hoy*; en 2012
  no sabías cuáles iban a llegar. Por eso la comparación justa es contra las mismas 100
  en partes iguales, no solo contra SPY.
- **Ajustar** `top_n` o `rebalance_days` en `config.yaml` y revisar el backtest después
  de cada cambio.
- **Cuidado con el sobreajuste:** si pruebas 50 configuraciones y te quedas con la mejor,
  el backtest se ve increíble y en vivo falla. Cambia poco, valida mucho.
- **Analista con IA:** agregar un LLM que lea noticias y explique cada operación.

## Estructura
```
agent/        el agente (datos, features, entrenamiento, backtest, trading)
docs/         la página web + docs/data/*.json que genera el agente
models/       el modelo entrenado
github-actions/  los archivos que lo corren solo en GitHub (van en .github/workflows/)
config.yaml   acciones, riesgo y parámetros del modelo
```
