"""Construye el eval set de dominio de Offside (10 ejemplos gold).

Qué es esto
-----------
Un *eval set de dominio*: ejemplos nuestros, etiquetados a mano, que miden lo
que le importa a nuestro usuario. No sale de ningún benchmark público, así que
no puede estar contaminado: son noticias de agosto de 2026 y las etiquetas las
pusimos nosotros leyendo cada fragmento.

Cómo se eligieron los 10
------------------------
NO son 10 ejemplos al azar del corpus. Se eligieron a mano buscando **casos
difíciles**: sitios donde el baseline de léxico se rompe, donde el impacto es
matizado, o donde hay vocabulario disparador en un contexto que no aplica.
Un eval set de casos fáciles no informa nada.

Balance deliberado:
  - 6 casos donde el léxico se equivoca (en categoría o en impacto)
  - 4 distractores donde el léxico acierta pero podría no hacerlo

Los distractores importan: un eval set formado solo por fallos también está
sesgado, y no detectaría una regresión que rompa lo que hoy sí funciona.

Procedencia
-----------
El texto NO se transcribe a mano: se extrae del corpus por `id`, así que es
byte a byte el que recolectamos. Aquí solo viven las etiquetas gold y el
razonamiento de por qué esa es la respuesta correcta.

Uso:
    python build_eval_set.py [--corpus ../data/processed/weak_labeled.jsonl]
                             [--out eval_set.jsonl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# corpus_id -> anotación gold hecha a mano por el equipo.
#   category / impact / team : la salida esperada
#   difficulty               : qué tipo de caso difícil ejercita
#   why                      : por qué esa es la respuesta correcta
#   lexicon_fails            : si el baseline de léxico se equivoca aquí
CURATED: dict[str, dict] = {
    "024ca02de28ba07e": {
        "category": "baja_confirmada",
        "impact": "negativo_alto",
        "team": "Getafe",
        "difficulty": "declaracion_que_contiene_un_hecho",
        "lexicon_fails": True,
        "why": (
            "El club comunica una grave lesión y confirma que el jugador se pierde toda la "
            "temporada: es una baja confirmada, no una simple declaración. El léxico ve las "
            "comillas del titular, dispara 'declaracion_contexto' y se detiene antes de leer "
            "el hecho. Distinguir el continente (una declaración) del contenido (una baja) es "
            "justo lo que un modelo con contexto debería hacer mejor que un regex."
        ),
    },
    "f538603932f7ce91": {
        "category": "baja_confirmada",
        "impact": "negativo_alto",
        "team": "Real Oviedo",
        "difficulty": "falso_negativo_total",
        "lexicon_fails": True,
        "why": (
            "'Baja sensible' y 'se perderá entre dos y tres meses' es inequívocamente una baja "
            "confirmada de alto impacto. El léxico la clasifica como irrelevante porque 'baja "
            "sensible' solo está en sentiment_boost, no como patrón de categoría. Es el peor "
            "error posible para nuestro usuario: la noticia más accionable del corpus, "
            "invisible para el baseline."
        ),
    },
    "1c4483eeeea95edd": {
        "category": "irrelevante",
        "impact": "neutro",
        "team": "Valencia",
        "difficulty": "vocabulario_disparador_fuera_de_contexto",
        "lexicon_fails": True,
        "why": (
            "Hay una expulsión, pero en el Trofeo Naranja: un amistoso de pretemporada. Una roja "
            "en amistoso no arrastra sanción liguera, así que no cambia la disponibilidad para "
            "ningún partido apostable. El léxico ve 'expulsado' y dispara sancion_suspension con "
            "impacto negativo alto. Para nuestro usuario esa señal es ruido puro."
        ),
    },
    "c7e96a97e238a5e9": {
        "category": "regreso_alta",
        "impact": "positivo_alto",
        "team": "Chelsea",
        "difficulty": "tiempo_verbal_signo_invertido",
        "lexicon_fails": True,
        "why": (
            "La noticia es que el jugador VUELVE tras 615 días; la suspensión está en pasado "
            "('fue suspendido... en noviembre de 2024') y es el contexto, no el hecho. El léxico "
            "ve 'suspendido' y devuelve sancion_suspension / negativo_alto. La etiqueta correcta "
            "tiene el signo contrario: es un alta. Confundir el signo del impacto es peor que no "
            "detectar nada, porque induce una decisión en la dirección equivocada."
        ),
    },
    "f31bd83c4ff5d3f4": {
        "category": "regreso_alta",
        "impact": "positivo_bajo",
        "team": "desconocido",
        "difficulty": "regreso_sin_vocabulario_de_regreso",
        "lexicon_fails": True,
        "why": (
            "'Reaparece once meses después' es un regreso, pero no usa ninguna de las fórmulas "
            "que el léxico busca ('vuelve a la convocatoria', 'recibe el alta'), así que cae en "
            "irrelevante. Además el impacto es positivo BAJO, no alto: el propio texto dice que "
            "tiene una contrarreloj para recuperar la forma. Reaparecer no es estar disponible."
        ),
    },
    "200eadd0cede0fdb": {
        "category": "regreso_alta",
        "impact": "positivo_bajo",
        "team": "FC Barcelona",
        "difficulty": "impacto_sobrestimado",
        "lexicon_fails": True,
        "why": (
            "La categoría sí la acierta el léxico (regreso), pero le asigna positivo_alto por el "
            "impact_default de la categoría. Adelantar la vuelta a los entrenamientos no es estar "
            "disponible para jugar: el impacto real es positivo bajo. Este ejemplo existe para "
            "medir el matiz de intensidad, que es donde la heurística categoría→impacto se rompe."
        ),
    },
    "07ef5a01c92d183f": {
        "category": "baja_confirmada",
        "impact": "negativo_alto",
        "team": "Las Palmas",
        "difficulty": "fragmento_compuesto",
        "lexicon_fails": False,
        "why": (
            "Distractor. El resumen RSS pega dos noticias distintas: un fichaje (irrelevante) y "
            "un portero operado (baja confirmada). La señal está en la segunda mitad. El léxico "
            "hoy acierta solo porque le añadimos el patrón 'pasó por el quirófano' después de "
            "encontrar este error a mano; sirve como test de regresión de ese parche."
        ),
    },
    "118d03958a0c1f46": {
        "category": "irrelevante",
        "impact": "neutro",
        "team": "ninguno",
        "difficulty": "distractor_normativo",
        "lexicon_fails": False,
        "why": (
            "Distractor. Habla de amonestaciones y ciclos de amarillas, vocabulario cargado de "
            "sanción, pero es un cambio de normativa general de la UEFA: no afecta a ningún "
            "equipo concreto en ningún partido concreto. Sin equipo afectado no hay señal."
        ),
    },
    "d6bd98772e5975c9": {
        "category": "declaracion_contexto",
        "impact": "neutro",
        "team": "Sporting de Gijon",
        "difficulty": "homonimia_dudas",
        "lexicon_fails": False,
        "why": (
            "Distractor. Aparece la palabra 'dudas', disparador de duda_fisica, pero aquí "
            "significa dudas defensivas: un juicio táctico del entrenador, no una molestia de un "
            "jugador. El léxico acierta por el orden de evaluación, no porque entienda la "
            "diferencia; un modelo debería acertarla por significado."
        ),
    },
    "50271335c0a843ea": {
        "category": "declaracion_contexto",
        "impact": "neutro",
        "team": "Bayern Munich",
        "difficulty": "frontera_declaracion_rumor",
        "lexicon_fails": False,
        "why": (
            "Distractor y caso frontera. 'Insinúa' y 'podría' son lenguaje de rumor, pero la "
            "fuente es el propio jugador hablando de sí mismo, así que es una declaración. "
            "Además es sobre su retirada a fin de temporada: no afecta ningún partido próximo, "
            "por eso el impacto es neutro y no negativo."
        ),
    },
}


def main() -> int:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=here / ".." / "data" / "processed" / "weak_labeled.jsonl"
    )
    parser.add_argument("--out", type=Path, default=here / "eval_set.jsonl")
    args = parser.parse_args()

    corpus = {}
    with args.corpus.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                corpus[rec["id"]] = rec

    missing = [cid for cid in CURATED if cid not in corpus]
    if missing:
        raise SystemExit(f"Estos ids curados no están en el corpus: {missing}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i, (corpus_id, gold) in enumerate(CURATED.items(), start=1):
            src = corpus[corpus_id]
            record = {
                "eval_id": f"OFF-{i:02d}",
                # --- INPUT: lo que ve el sistema ---
                "input": {
                    "text": src["text"],
                    "source": src["source"],
                    "published_at": src.get("published_at"),
                },
                # --- SALIDA ESPERADA: la respuesta correcta según el equipo ---
                "expected": {
                    "category": gold["category"],
                    "impact": gold["impact"],
                    "team": gold["team"],
                },
                # --- Metadatos del caso difícil ---
                "difficulty": gold["difficulty"],
                "lexicon_fails": gold["lexicon_fails"],
                "why": gold["why"],
                "provenance": {"corpus_id": corpus_id, "link": src.get("link")},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_fail = sum(1 for g in CURATED.values() if g["lexicon_fails"])
    print(f"{len(CURATED)} ejemplos gold escritos en {args.out}")
    print(f"  casos donde el léxico se equivoca: {n_fail}")
    print(f"  distractores (léxico acierta):     {len(CURATED) - n_fail}")
    cats = sorted({g["category"] for g in CURATED.values()})
    print(f"  categorías ejercitadas: {len(cats)} -> {', '.join(cats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
