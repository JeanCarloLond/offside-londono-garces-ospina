"""Construye los casos con imagen y sus variantes de invarianza.

    python final/vision/crear_casos.py   ->  casos/*.png  y  eval_set_vision.json

Las capturas se generan aquí, a partir de fragmentos reales del corpus (los mismos
que ya están en data/rag/corpus.jsonl), con un estilo neutro de "publicación en
redes": sin logos ni marcas de ningún medio, y con la fuente citada como texto. Así
no redistribuimos imágenes de terceros y sabemos exactamente qué dice cada captura.

Siete casos:
  V-01..V-05  capturas de noticias que el corpus respalda
  V-06        trampa: una imagen que no es de fútbol (debe activar la válvula)
  V-07        trampa: una captura FALSA (una lesión de Mbappé que ninguna noticia
              recoge). Lo correcto es tratarla como no confirmada, no como un hecho.

Invarianza: para V-01..V-05 hay tres variantes con el mismo contenido (pequeña y
comprimida, tema oscuro con otra fuente, rotada). El sistema debería dar la misma
señal en las cuatro versiones.
"""

from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

AQUI = Path(__file__).resolve().parent
CASOS = AQUI / "casos"
FUENTES = Path("C:/Windows/Fonts")

PUBLICACIONES = {
    "V-01": {
        "cuenta": "@futbol_al_dia",
        "texto": "Vinicius se pierde el Clásico: cuatro partidos de sanción. El Comité de "
        "Disciplina le impone una dura sanción por su roja ante el Villarreal.",
        "fuente": "Vía Marca · 14 oct 2026",
    },
    "V-02": {
        "cuenta": "@noticias_sevilla",
        "texto": "El Sevilla prepara la visita al Bernabéu sin Guridi. El centrocampista sufre "
        "una lesión en el aductor largo del muslo izquierdo.",
        "fuente": "Vía Marca · 14 oct 2026",
    },
    "V-03": {
        "cuenta": "@bundesliga_es",
        "texto": "El Bayern no correrá riesgos con Musiala. El alemán vuelve a ejercitarse tras "
        "el desgarro que sufrió en la Nations frente a Países Bajos.",
        "fuente": "Vía Marca · 8 oct 2026",
    },
    "V-04": {
        "cuenta": "@mercado_che",
        "texto": "El agente de Erik Lira se reúne con Braulio Vázquez en Paterna y alimenta los "
        "rumores sobre su futuro.",
        "fuente": "Vía Marca · 14 oct 2026",
    },
    "V-05": {
        "cuenta": "@malaga_hoy",
        "texto": "Funes ya tiene a su último fichaje listo. Jens Cajuste vuelve al grupo y podrá "
        "jugar tras un mes de lesión.",
        "fuente": "Vía Marca · 12 oct 2026",
    },
    "V-07": {
        "cuenta": "@rumores_futbol_24",
        "texto": "ÚLTIMA HORA: Mbappé, tres meses de baja por una rotura del ligamento cruzado. "
        "Se perderá el Clásico.",
        "fuente": "Sin fuente citada · hace 20 min",
    },
}

CASOS_DEF = [
    {
        "eval_id": "V-01",
        "pregunta": "Me llegó esta captura. ¿Entonces Vinicius no juega el Clásico?",
        "esperado": {
            "category": "sancion_suspension",
            "impact": "negativo_alto",
            "team": "Real Madrid",
        },
        "criterio": "La captura y el corpus coinciden: Vinicius tiene cuatro partidos de sanción y se pierde el Clásico. Sanción, impacto negativo alto para el Real Madrid.",
        "docs_relevantes": ["cba83938135d711f", "121aa9cb87ef0452", "a223284dd29ea88b"],
        "adversarial": False,
        "tipo": "captura respaldada",
    },
    {
        "eval_id": "V-02",
        "pregunta": "¿Este jugador juega en el Bernabéu?",
        "esperado": {"category": "baja_confirmada", "impact": "negativo_alto", "team": "Sevilla"},
        "criterio": "Guridi tiene una lesión en el aductor y el Sevilla viaja sin él: baja confirmada, impacto negativo alto para el Sevilla.",
        "docs_relevantes": ["3b31d5310b9bd6e4"],
        "adversarial": True,
        "tipo": "la pregunta solo dice «este jugador»: el nombre está en la imagen",
    },
    {
        "eval_id": "V-03",
        "pregunta": "¿Ya puede volver a jugar?",
        "esperado": {
            "category": "regreso_alta",
            "impact": "positivo_bajo",
            "team": "Bayern Munich",
        },
        "criterio": "Musiala vuelve a ejercitarse tras el desgarro, aunque el Bayern no correrá riesgos: regreso de impacto positivo bajo.",
        "docs_relevantes": ["8969f60685def770"],
        "adversarial": True,
        "tipo": "la pregunta no nombra a nadie; noticias de agosto que lo daban de baja",
    },
    {
        "eval_id": "V-04",
        "pregunta": "¿Ya es oficial el fichaje?",
        "esperado": {"category": "rumor_no_confirmado", "impact": "neutro", "team": "Valencia"},
        "criterio": "No hay fichaje: es una reunión que alimenta rumores. Rumor no confirmado; darlo por hecho es el error más caro.",
        "docs_relevantes": ["8615c6577f427a94"],
        "adversarial": True,
        "tipo": "la pregunta presupone el hecho",
    },
    {
        "eval_id": "V-05",
        "pregunta": "¿Ya puede jugar este jugador?",
        "esperado": {"category": "regreso_alta", "impact": "positivo_bajo", "team": "Malaga"},
        "criterio": "Cajuste vuelve al grupo y podrá jugar tras un mes de lesión: regreso, impacto positivo bajo para el Málaga.",
        "docs_relevantes": ["25a07578dc93f4c8"],
        "adversarial": False,
        "tipo": "captura respaldada",
    },
    {
        "eval_id": "V-06",
        "pregunta": "¿Qué jugador está lesionado según la imagen?",
        "esperado": {"category": "irrelevante", "impact": "neutro", "team": "ninguno"},
        "criterio": "La imagen es una receta de cocina, no tiene nada de fútbol: lo correcto es decir que no encuentra la información y no emitir señal.",
        "docs_relevantes": [],
        "adversarial": True,
        "en_corpus": False,
        "tipo": "imagen que no es de fútbol",
    },
    {
        "eval_id": "V-07",
        "pregunta": "¿Mbappé se pierde el Clásico?",
        "esperado": {
            "category": "rumor_no_confirmado",
            "impact": "negativo_bajo",
            "team": "Real Madrid",
        },
        "criterio": "La captura no tiene fuente y ninguna noticia del corpus habla de una lesión de Mbappé. Lo correcto es tratarla como no confirmada (rumor). Presentarla como baja confirmada es el peor error.",
        "docs_relevantes": [],
        "adversarial": True,
        "en_corpus": False,
        "tipo": "captura falsa sin respaldo en el corpus",
    },
]


def fuente(nombre: str, tam: int):
    for f in (nombre, "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(str(FUENTES / f), tam)
        except OSError:
            continue
    return ImageFont.load_default()


def publicacion(p: dict, oscura: bool = False, familia: str = "segoeui.ttf") -> Image.Image:
    fondo, tinta, gris = (
        ((21, 32, 43), (231, 233, 234), (139, 152, 165))
        if oscura
        else (
            (255, 255, 255),
            (15, 20, 25),
            (83, 100, 113),
        )
    )
    ancho = 900
    lineas = textwrap.wrap(p["texto"], width=46)
    alto = 170 + 46 * len(lineas) + 70
    img = Image.new("RGB", (ancho, alto), fondo)
    d = ImageDraw.Draw(img)
    d.ellipse((40, 40, 110, 110), fill=(29, 155, 240))
    d.text((130, 48), p["cuenta"], font=fuente(familia, 30), fill=tinta)
    d.text((130, 86), "Publicación compartida", font=fuente(familia, 22), fill=gris)
    y = 140
    for linea in lineas:
        d.text((40, y), linea, font=fuente(familia, 34), fill=tinta)
        y += 46
    d.text((40, y + 20), p["fuente"], font=fuente(familia, 24), fill=gris)
    return img


def receta() -> Image.Image:
    img = Image.new("RGB", (900, 520), (253, 246, 227))
    d = ImageDraw.Draw(img)
    d.text(
        (40, 40),
        "Arroz con pollo para 4 personas",
        font=fuente("georgia.ttf", 40),
        fill=(90, 50, 20),
    )
    pasos = [
        "1. Sofreír cebolla, ajo y pimentón.",
        "2. Dorar el pollo en presas.",
        "3. Agregar el arroz y el caldo caliente.",
        "4. Cocinar 20 minutos a fuego bajo.",
    ]
    for i, paso in enumerate(pasos):
        d.text((40, 130 + 70 * i), paso, font=fuente("georgia.ttf", 32), fill=(60, 40, 20))
    return img


def variantes(img: Image.Image, p: dict) -> dict[str, Image.Image]:
    pequeno = img.resize((img.width // 2, img.height // 2))
    buf = io.BytesIO()
    pequeno.save(buf, "JPEG", quality=35)
    return {
        "pequena_jpeg": Image.open(io.BytesIO(buf.getvalue())).convert("RGB"),
        "oscura": publicacion(p, oscura=True, familia="georgia.ttf"),
        "rotada": img.rotate(4, expand=True, fillcolor=(255, 255, 255)),
    }


def main() -> int:
    CASOS.mkdir(parents=True, exist_ok=True)
    salida = []
    for c in CASOS_DEF:
        eid = c["eval_id"]
        img = receta() if eid == "V-06" else publicacion(PUBLICACIONES[eid])
        ruta = CASOS / f"{eid}.png"
        img.save(ruta)
        vari = {}
        if c.get("en_corpus", True):
            for nombre, v in variantes(img, PUBLICACIONES[eid]).items():
                rv = CASOS / f"{eid}_{nombre}.png"
                v.save(rv)
                vari[nombre] = rv.relative_to(AQUI).as_posix()
        salida.append(
            {
                "eval_id": eid,
                "input": {"text": c["pregunta"], "imagen": ruta.relative_to(AQUI).as_posix()},
                "esperado": c["esperado"],
                "criterio": c["criterio"],
                "docs_relevantes": c["docs_relevantes"],
                "en_corpus": c.get("en_corpus", True),
                "adversarial": c["adversarial"],
                "tipo_dificultad": c["tipo"],
                "variantes": vari,
            }
        )
    (AQUI / "eval_set_vision.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(salida)} casos, {sum(len(s['variantes']) for s in salida)} variantes -> {CASOS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
