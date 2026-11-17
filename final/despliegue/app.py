"""Offside · demo pública (Hugging Face Spaces, ZeroGPU).

El sistema completo de los módulos 1 a 5, en una sola aplicación:
  pregunta (y, si se quiere, una captura) -> visión (M4) -> RAG avanzado con válvula y
  herramienta de fechas (M3) -> la señal que mide el harness (M2), con las mismas
  categorías del clasificador de M1.

Rutas de la API (además de la interfaz):
  /responder   la pregunta y la imagen opcional
  /salud       health check: ¿cargaron los modelos y responde el índice? No usa GPU.

Variables de entorno (en el Space, como secretos o variables):
  WANDB_API_KEY   si está, cada petición se registra en W&B (proyecto "offside")
  OFFSIDE_INT4    "1" para cargar el generador cuantizado a int4
  OFFSIDE_VISION  "0" para desactivar el componente visual
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

try:  # en el Space, ZeroGPU
    import spaces

    GPU = spaces.GPU
except ImportError:  # en local: decorador vacío

    def GPU(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


AQUI = Path(__file__).resolve().parent
for ruta in (AQUI, AQUI / "rag", AQUI.parent.parent / "proyecto1" / "rag", AQUI.parent / "vision"):
    if ruta.exists():
        sys.path.insert(0, str(ruta))

import generador as gen_mod  # noqa: E402
import gradio as gr  # noqa: E402
import sistemas  # noqa: E402
import torch  # noqa: E402
from indice import Recuperador  # noqa: E402
from observabilidad import Observador  # noqa: E402

EN_ZEROGPU = os.getenv("SPACES_ZERO_GPU", "").lower() in ("1", "true")
DEVICE = "cuda" if (EN_ZEROGPU or torch.cuda.is_available()) else "cpu"
CUANTIZAR = os.getenv("OFFSIDE_INT4", "0") == "1"
USAR_VISION = os.getenv("OFFSIDE_VISION", "1") == "1"
VERSION = f"generador {'int4' if CUANTIZAR else 'fp16'} · visión {'sí' if USAR_VISION else 'no'}"

INICIO = time.time()
recuperador = Recuperador()
generador = gen_mod.Generador(device=DEVICE, cuantizar=CUANTIZAR)
sistema = sistemas.SistemaRAG(sistemas.FINAL, recuperador, generador)
vision = None
if USAR_VISION:
    from vision import Vision

    vision = Vision(device=DEVICE)
observador = Observador(VERSION)

NOMBRES = {
    "baja_confirmada": "Baja confirmada",
    "sancion_suspension": "Sanción",
    "duda_fisica": "Duda física",
    "regreso_alta": "Regreso",
    "cambio_tactico": "Cambio táctico o alineación",
    "declaracion_contexto": "Declaración",
    "rumor_no_confirmado": "Rumor sin confirmar",
    "irrelevante": "Sin señal",
}
IMPACTOS = {
    "negativo_alto": "negativo alto",
    "negativo_bajo": "negativo bajo",
    "neutro": "neutro",
    "positivo_bajo": "positivo bajo",
    "positivo_alto": "positivo alto",
}
AVISO = (
    "_Offside informa, no recomienda apostar. Responde solo con noticias recolectadas hasta el "
    "14 de octubre de 2026; comprueba la fuente antes de decidir._"
)


# ZeroGPU cobra a la cuota del visitante (2 min/día anónimo, 5 min con sesión) y rechaza la
# petición si la reserva no cabe. La primera llamada tras arrancar carga los pesos en la GPU
# y necesita más margen; las siguientes caben en 45 s.
ESTADO = {"calentado": False}


def _duracion(pregunta, imagen) -> int:
    return 45 if ESTADO["calentado"] else 110


def _etapa(nombre: str, t0: float) -> None:
    print(
        json.dumps({"evento": "etapa", "etapa": nombre, "s": round(time.perf_counter() - t0, 2)}),
        flush=True,
    )


@GPU(duration=_duracion)
def _pipeline(pregunta: str, imagen):
    """Corre en la GPU. Devuelve también los tiempos por etapa: entrar a esta función es
    lo que tarda ZeroGPU en asignar la GPU y cargar los pesos."""
    tiempos = {"entrada_gpu": time.time()}
    t = time.perf_counter()
    _etapa("entrada_gpu", t)
    lectura = vision.leer(imagen) if (vision is not None and imagen is not None) else None
    tiempos["vision"] = round(time.perf_counter() - t, 3)
    _etapa("vision", t)
    if lectura is not None and not lectura.es_futbol:
        return None, lectura, tiempos
    salida = sistema(pregunta, captura=lectura.texto if lectura else None)
    _etapa("rag", t)
    tiempos.update(salida["traza"].get("tiempos_s", {}))
    return salida, lectura, tiempos


def _formatear(salida, lectura, imagen) -> str:
    partes = []
    if imagen is not None and lectura is None:
        partes.append("_El componente visual está desactivado: se ignoró la imagen._")
    if lectura is not None and not lectura.es_futbol:
        return (
            "### No encuentro información de fútbol en la imagen\n\n"
            "La captura no parece una noticia, un parte o una alineación, así que no hay "
            "nada que responder con ella.\n\n" + AVISO
        )
    valvula = salida["respuesta"].startswith("No encuentro")
    partes.append(f"### {'No lo sé' if valvula else 'Respuesta'}\n\n{salida['respuesta']}")
    if not valvula:
        partes.append(
            f"**Señal:** {NOMBRES.get(salida['category'], salida['category'])} · impacto "
            f"{IMPACTOS.get(salida['impact'], salida['impact'])} · {salida['team']}"
        )
    if lectura is not None:
        partes.append(f"**Lo que leí en la captura (sin verificar):** {lectura.texto}")
    for h in salida.get("herramientas", []):
        res = h["resultado"]
        partes.append(f"**Cálculo de fechas:** {res.get('lectura', res.get('error'))}")
    if salida.get("ids"):
        fuentes = []
        for i, doc_id in enumerate(salida["ids"], start=1):
            c = recuperador.por_id[doc_id]
            fuentes.append(f"{i}. [{c.medio}, {c.fecha}]({c.link}): {c.texto[:140]}…")
        partes.append("**Noticias usadas:**\n\n" + "\n".join(fuentes))
    partes.append(AVISO)
    return "\n\n".join(partes)


def responder(pregunta: str, imagen=None) -> str:
    pregunta = (pregunta or "").strip()
    if not pregunta and imagen is None:
        return "Escribe una pregunta o sube una captura."
    if not pregunta:
        pregunta = "¿Qué significa esta captura para el próximo partido?"
    t0 = time.perf_counter()
    try:
        t_llamada = time.time()
        salida, lectura, tiempos = _pipeline(pregunta, imagen)
        tiempos["espera_gpu"] = round(tiempos.pop("entrada_gpu") - t_llamada, 3)
        ESTADO["calentado"] = True
    except Exception as e:  # el error queda registrado, el usuario recibe un mensaje claro
        observador.registrar(
            time.perf_counter() - t0,
            True,
            False,
            imagen is not None,
            False,
            "",
            "",
            len(pregunta),
            error=f"{type(e).__name__}: {str(e)[:200]}",
        )
        return "Hubo un problema al responder. Inténtalo de nuevo en un momento."
    latencia = time.perf_counter() - t0
    if salida is None:  # válvula visual
        observador.registrar(
            latencia, False, False, True, True, "irrelevante", "neutro", len(pregunta)
        )
    else:
        observador.registrar(
            latencia,
            usa_rag=True,
            usa_herramienta=bool(salida.get("herramientas")),
            usa_vision=lectura is not None,
            valvula=salida["respuesta"].startswith("No encuentro"),
            categoria=salida["category"],
            impacto=salida["impact"],
            largo_pregunta=len(pregunta),
            tiempos=tiempos,
        )
    return _formatear(salida, lectura, imagen)


def salud() -> dict:
    """Health check. No hace una pregunta real ni usa GPU: comprueba que los modelos están
    cargados y que el índice recupera."""
    t0 = time.perf_counter()
    estado = {"version": VERSION, "device": DEVICE, "en_zerogpu": EN_ZEROGPU}
    try:
        ids = recuperador.dense("¿Vinicius puede jugar el Clásico?", 1)
        estado["indice"] = {
            "chunks": recuperador.coleccion.count(),
            "documentos": len(recuperador.chunks),
            "responde": bool(ids),
        }
        estado["generador_cargado"] = generador.model is not None
        estado["vision_cargada"] = vision is not None
        estado["observabilidad"] = observador.resumen()
        estado["uptime_s"] = round(time.time() - INICIO)
        estado["estado"] = "ok" if ids and generador.model is not None else "degradado"
    except Exception as e:
        estado["estado"] = "error"
        estado["error"] = f"{type(e).__name__}: {e}"
    estado["latencia_s"] = round(time.perf_counter() - t0, 3)
    return estado


EJEMPLOS = [
    ["¿Volverá Uche a jugar con el Getafe esta temporada?", None],
    ["¿Puede contar el Oviedo con Chaira para el 29 de noviembre?", None],
    ["¿Kylian Mbappé se pierde el Clásico?", None],
]
for nombre, pregunta in (
    ("V-01.png", "Me llegó esta captura. ¿Entonces Vinicius no juega el Clásico?"),
    ("V-07.png", "¿Mbappé se pierde el Clásico?"),
):
    for carpeta in (AQUI / "ejemplos", AQUI.parent / "vision" / "casos"):
        if (carpeta / nombre).exists():
            EJEMPLOS.append([pregunta, str(carpeta / nombre)])
            break

with gr.Blocks(title="Offside") as demo:
    gr.Markdown(
        "# Offside\n"
        "Pregúntale por la disponibilidad de un jugador o sube la captura de una noticia. "
        "Offside busca en la prensa deportiva recolectada (Marca y El Tiempo, hasta el 14 de "
        "octubre de 2026), te dice qué significa para el partido y de qué noticia lo saca. Si "
        "no lo sabe, lo dice."
    )
    with gr.Row():
        with gr.Column():
            pregunta = gr.Textbox(label="Pregunta", placeholder="¿Juega Vinicius el Clásico?")
            imagen = gr.Image(label="Captura (opcional)", type="pil")
            boton = gr.Button("Preguntar", variant="primary")
        with gr.Column():
            salida = gr.Markdown()
    gr.Examples(EJEMPLOS, inputs=[pregunta, imagen])
    boton.click(responder, [pregunta, imagen], salida, api_name="responder")
    pregunta.submit(responder, [pregunta, imagen], salida, api_name=False)
    with gr.Accordion("Estado del sistema (health check)", open=False):
        boton_salud = gr.Button("Comprobar")
        json_salud = gr.JSON()
        boton_salud.click(salud, None, json_salud, api_name="salud")

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
