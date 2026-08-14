"""Correcciones manuales a la etiqueta débil (léxico) sobre el conjunto de
VALIDACIÓN real, hechas leyendo cada fragmento uno por uno.

Por qué existe: si evaluamos el baseline de léxico contra etiquetas que el
propio léxico generó, el resultado es circular (acierta 100% por
construcción, no porque sea bueno). El plan del equipo (Context.md, sección
4) es justamente evitar esto con un "conjunto dorado" anotado a mano,
distinto del conjunto de entrenamiento con supervisión débil.

Este archivo es esa primera pasada de verificación manual sobre el conjunto
de validación (49 fragmentos reales, recolectados 2026-08-09). Se hizo
leyendo cada titular — no hay recolección adicional de texto, solo la
corrección de la etiqueta cuando el léxico se equivocó.

Encontramos 2 correcciones sobre 49 (~96% de acuerdo léxico/lectura manual):
ambos casos son fragmentos "compuestos" (dos noticias distintas pegadas en
el mismo resumen RSS) donde el léxico se quedó con la mitad irrelevante del
texto y no vio la mención de lesión en la segunda mitad.

Las 49 filas resultantes (data/gold/gold_verified.jsonl) fueron revisadas
por los tres integrantes del equipo antes de la entrega. Ver docs/dataset.md,
sección "Anotación", para el procedimiento completo.
"""

from __future__ import annotations

# id (de weak_labeled.jsonl) -> (categoria_correcta, impacto_correcto, motivo)
CORRECTIONS: dict[str, tuple[str, str, str]] = {
    "200eadd0cede0fdb": (
        "regreso_alta",
        "positivo_bajo",
        "El léxico lo dejó en 'irrelevante'. El texto ('Joan García, Eric y "
        "Gordon adelantan su vuelta al trabajo. Se ejercitaron este domingo') "
        "es justamente un regreso a entrenamientos, solo que con vocabulario "
        "que el léxico actual no cubre ('adelantan su vuelta al trabajo').",
    ),
    "07ef5a01c92d183f": (
        "baja_confirmada",
        "negativo_alto",
        "Fragmento compuesto: el titular es sobre un fichaje (irrelevante para "
        "la tarea), pero el resumen trae una segunda noticia real -- 'El meta "
        "de Las Palmas tuvo que pasar por el quirófano tras un choque "
        "fortuito' -- que sí es una baja confirmada y que el léxico no "
        "detectó por no tener 'quirófano' en el patrón de baja_confirmada.",
    ),
}
