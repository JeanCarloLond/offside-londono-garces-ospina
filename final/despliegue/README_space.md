---
title: Offside
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Noticias de fútbol para apostadores, con fuente
---

# Offside

Pregúntale por la disponibilidad de un jugador o sube la captura de una noticia. Offside busca en la
prensa deportiva recolectada (Marca y El Tiempo, hasta el 14 de octubre de 2026), te dice qué
significa para el partido y de qué noticia lo saca. Si no lo sabe, lo dice.

Proyecto integrador del curso SI4006 (Universidad EAFIT), equipo Offside. El código, la evaluación
y el reporte están en https://github.com/JeanCarloLond/offside-londono-garces-ospina.

**Health check:** `POST /gradio_api/call/salud` (o `Client("Jclo/offside").predict(api_name="/salud")`
con `gradio_client`). Devuelve si los modelos cargaron y si el índice responde, sin usar GPU.

El texto de las noticias es de cada medio; el corpus solo guarda el titular y la entradilla que
publican para sindicación, con su enlace.
