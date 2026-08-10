"""Construye un pequeño set semilla ANOTADO A MANO para arrancar el gold set.

Por qué existe este archivo (léanlo antes de usarlo en el notebook de M1):

El scraping real (collect_rss.py + weak_label.py) se corrió el 2026-08-09, en
plena pretemporada europea. El resultado honesto: de las 8 categorías del
proyecto, la prensa de esos días solo produce ejemplos reales de
"irrelevante", "declaracion_contexto", "sancion_suspension" y "regreso_alta"
en volumen aprovechable — "baja_confirmada", "duda_fisica", "cambio_tactico"
y "rumor_no_confirmado" casi no aparecen (ver docs/dataset.md, sección de
sesgos). Sin datos de esas categorías, ni el baseline de léxico ni el
fine-tuning tienen nada que aprender para ellas.

Para poder mostrar el pipeline completo (8 categorías) en el notebook de M1,
este script genera un seed PEQUEÑO Y SINTÉTICO:
  - Frases escritas por el equipo (no extraídas de ningún artículo real).
  - Con equipos y jugadores FICTICIOS a propósito — nunca se le atribuye una
    lesión/sanción/baja a una persona real que no la tuvo. Es la misma razón
    ética que el propio equipo documentó en Context.md (sección 7): no
    propagar información falsa sobre personas identificables.
  - Etiquetado como source="synthetic_seed" / label_method="manual_synthetic"
    para que nunca se confunda con datos reales en ningún análisis posterior.

Esto es un ANDAMIO, no el dataset final. El equipo debe reemplazarlo por
anotación manual real en cuanto la recolección (corrida repetidamente a lo
largo de las próximas semanas, ver README de data_collection) junte
suficientes ejemplos reales de estas categorías.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (texto, categoria, impacto) — equipos/jugadores ficticios a propósito.
SEED_EXAMPLES: list[tuple[str, str, str]] = [
    # --- baja_confirmada (negativo_alto por defecto) ---
    (
        "El Deportivo Sauce confirma la baja de su capitán para el resto de la semana: "
        "rotura fibrilar y no estará disponible ante el Unión Norte.",
        "baja_confirmada",
        "negativo_alto",
    ),
    (
        "Real Cauca causa baja a su lateral izquierdo, descartado para el derbi tras "
        "las pruebas médicas de esta mañana.",
        "baja_confirmada",
        "negativo_alto",
    ),
    (
        "El técnico del Atlético Peña confirmó que su delantero centro se pierde el "
        "partido del domingo por decisión médica.",
        "baja_confirmada",
        "negativo_alto",
    ),
    (
        "Club Andino se queda fuera de la convocatoria con dos titulares menos: "
        "ambos arrastran molestias que no llegaron a tiempo.",
        "baja_confirmada",
        "negativo_alto",
    ),
    (
        "Sporting Bahía anuncia que su portero titular no estará disponible por al "
        "menos tres semanas tras la lesión sufrida el sábado.",
        "baja_confirmada",
        "negativo_alto",
    ),
    (
        "El mediocampista de Unión Norte causa baja confirmada por fatiga muscular "
        "y no viajará con el resto del plantel.",
        "baja_confirmada",
        "negativo_alto",
    ),
    # --- sancion_suspension (real hay 2 casos ya; sumamos algunos más) ---
    (
        "El defensor de Real Cauca fue expulsado con roja directa y arrastra sanción "
        "para el próximo compromiso liguero.",
        "sancion_suspension",
        "negativo_alto",
    ),
    (
        "El comité de disciplina sancionó por dos partidos al volante de Atlético "
        "Peña tras la acusación de conducta antideportiva.",
        "sancion_suspension",
        "negativo_alto",
    ),
    (
        "Deportivo Sauce pierde a su capitán, apercibido y ahora suspendido tras "
        "acumular la quinta amarilla de la temporada.",
        "sancion_suspension",
        "negativo_alto",
    ),
    (
        "El delantero de Club Andino cumple sanción de un partido por la expulsión "
        "del fin de semana pasado.",
        "sancion_suspension",
        "negativo_alto",
    ),
    (
        "Sporting Bahía se queda sin su lateral derecho: fue sancionado tras la roja "
        "directa mostrada en la jornada anterior.",
        "sancion_suspension",
        "negativo_alto",
    ),
    # --- duda_fisica (negativo_bajo por defecto) ---
    (
        "El extremo de Unión Norte es duda para el fin de semana por unas molestias "
        "musculares que arrastra desde la semana pasada.",
        "duda_fisica",
        "negativo_bajo",
    ),
    (
        "Real Cauca reporta que su central está pendiente de pruebas médicas tras "
        "sentir sobrecarga en el entrenamiento de ayer.",
        "duda_fisica",
        "negativo_bajo",
    ),
    (
        "El técnico de Atlético Peña reconoció que dos jugadores están al límite "
        "físicamente y decidirá su participación el mismo día del partido.",
        "duda_fisica",
        "negativo_bajo",
    ),
    (
        "Deportivo Sauce mantiene en el dique seco a su mediapunta, con una dolencia "
        "que todavía no tiene diagnóstico definitivo.",
        "duda_fisica",
        "negativo_bajo",
    ),
    (
        "Club Andino no confirma si su portero titular llega al domingo: sigue con "
        "molestias en el hombro desde el último entrenamiento.",
        "duda_fisica",
        "negativo_bajo",
    ),
    # --- regreso_alta (positivo_alto por defecto; real hay 1 caso) ---
    (
        "Sporting Bahía recupera a su goleador, que vuelve a la convocatoria tras "
        "cumplir el plazo de recuperación de su lesión.",
        "regreso_alta",
        "positivo_alto",
    ),
    (
        "El capitán de Real Cauca recibe el alta médica y estará disponible de "
        "nuevo para el próximo partido de local.",
        "regreso_alta",
        "positivo_alto",
    ),
    (
        "Unión Norte celebra el regreso de su lateral a los entrenamientos con el "
        "grupo, tras dos meses de baja por lesión.",
        "regreso_alta",
        "positivo_alto",
    ),
    (
        "Atlético Peña confirma que su defensor está recuperado para el derbi "
        "después de superar la lesión que lo mantuvo fuera un mes.",
        "regreso_alta",
        "positivo_alto",
    ),
    # --- cambio_tactico (neutro por defecto) ---
    (
        "El entrenador de Club Andino prepara un cambio de sistema y pasaría a "
        "jugar con línea de cinco para el próximo partido.",
        "cambio_tactico",
        "neutro",
    ),
    (
        "Deportivo Sauce probó un nuevo dibujo táctico en el último entrenamiento "
        "de cara al partido del fin de semana.",
        "cambio_tactico",
        "neutro",
    ),
    (
        "La alineación probable de Real Cauca incluye tres cambios respecto al "
        "último partido, con rotación en el mediocampo.",
        "cambio_tactico",
        "neutro",
    ),
    (
        "Sporting Bahía manejaría un once inicial distinto, con su suplente "
        "habitual como novedad en el lateral derecho.",
        "cambio_tactico",
        "neutro",
    ),
    # --- rumor_no_confirmado (negativo_bajo por defecto) ---
    (
        "Según fuentes cercanas al club, el delantero de Atlético Peña podría "
        "perderse el próximo partido, aunque nada está confirmado todavía.",
        "rumor_no_confirmado",
        "negativo_bajo",
    ),
    (
        "Trascendió que el capitán de Unión Norte tendría una lesión de mayor "
        "gravedad de la reportada inicialmente, sin confirmación oficial.",
        "rumor_no_confirmado",
        "negativo_bajo",
    ),
    (
        "No está claro si el mediocampista de Club Andino llega al fin de semana: "
        "el club no ha confirmado la gravedad de la molestia.",
        "rumor_no_confirmado",
        "negativo_bajo",
    ),
    (
        "Se especula con una posible sanción para el defensor de Deportivo Sauce, "
        "a falta de confirmación por parte del comité de disciplina.",
        "rumor_no_confirmado",
        "negativo_bajo",
    ),
    # --- irrelevante (neutro; ya sobra en el corpus real, poquitos de refuerzo) ---
    (
        "Sporting Bahía presentó su nueva camiseta suplente de cara a la temporada "
        "que arranca el próximo mes.",
        "irrelevante",
        "neutro",
    ),
    (
        "Real Cauca anunció un amistoso de pretemporada contra un equipo de la "
        "categoría inferior para la próxima semana.",
        "irrelevante",
        "neutro",
    ),
]


def main() -> int:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=here / ".." / "data" / "gold" / "gold_seed_synthetic.jsonl"
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as f:
        for i, (text, category, impact) in enumerate(SEED_EXAMPLES):
            record = {
                "id": f"synthetic_{i:03d}",
                "source": "synthetic_seed",
                "region": "es",
                "link": None,
                "published_at": None,
                "text": text,
                "category": category,
                "impact": impact,
                "label_method": "manual_synthetic",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{len(SEED_EXAMPLES)} ejemplos sintéticos escritos en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
