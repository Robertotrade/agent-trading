# Automatización en GitHub

Estos archivos hacen que los agentes corran solos en GitHub:

- `train.yml`: re-entrena al agente semanal cada sábado.
- `trade.yml`: el agente semanal opera de lunes a viernes en tu cuenta demo de Alpaca
  (si no hay llaves de Alpaca, solo publica las señales).
- `daytrading.yml`: el agente de day trading. Cada mañana publica el plan del día y cada
  tarde anota el día en el **historial público** (con el modelo guardado antes de que abriera
  la bolsa) y se re-entrena para el día siguiente.

Cuando subas el proyecto a GitHub, van dentro de la carpeta `.github/workflows/`
(la carpeta empieza con punto, por eso tu Mac la esconde).
Si le pides a Claude que suba el proyecto, él los pone en su lugar.
