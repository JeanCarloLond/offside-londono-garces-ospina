"""Detección y mitigación del sesgo de VERBOSIDAD del juez.

El sesgo
--------
Es el sesgo mejor documentado de los jueces LLM: puntúan más alto las respuestas
largas aunque no aporten información. Nos afecta de lleno, porque en M3 el RAG
va a redactar señales con más texto que el clasificador de M1, y si el juez
premia la longitud, el RAG "ganaría" sin ser mejor. La comparación entre
módulos —que es todo el punto de tener una vara fija— quedaría rota.

El experimento
--------------
Tomamos las MISMAS predicciones de un sistema y las renderizamos con tres
longitudes distintas y exactamente la misma información:

    escueta   "baja_confirmada/negativo_alto"                    (~30 chars)
    fija      la plantilla del harness, que es la que usamos      (~70 chars)
    verbosa   la misma información envuelta en relleno cortés     (~300 chars)

Si el juez sube la nota al crecer la longitud, el sesgo está y es medible: nada
más cambió.

La mitigación
-------------
`render_senal()` en harness.py emite TODAS las señales con la misma plantilla de
longitud fija, sea cual sea el sistema. Así la longitud deja de ser una variable
y no puede llevar información: el juez no tiene de dónde sacar la preferencia.
Es una mitigación estructural, no un parche sobre el puntaje.

No pretendemos eliminar el sesgo del modelo — eso no se puede desde fuera. Lo
que hacemos es quitarle la señal de la que se alimenta, y medir cuánta ventaja
habría tenido un sistema verboso si no lo hubiéramos hecho.

Uso:
    python bias_check.py               # sobre el sistema de léxico
    python bias_check.py --json bias_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness import SistemaLexico, render_senal


def render_escueta(respuesta: dict) -> str:
    return f"{respuesta['category']}/{respuesta['impact']}"


def render_verbosa(respuesta: dict) -> str:
    """Misma información, envuelta en relleno que no añade nada."""
    return (
        "Tras analizar cuidadosamente el fragmento de prensa proporcionado y considerar el "
        "contexto deportivo relevante, el sistema concluye lo siguiente: la categoría del hecho "
        f"identificado es {respuesta['category']}, con un impacto estimado de "
        f"{respuesta['impact']} sobre el equipo {respuesta.get('team') or 'ninguno'}. "
        "Esta valoración se apoya en los elementos informativos presentes en el texto analizado."
    )


RENDERS = {
    "escueta": render_escueta,
    "fija": render_senal,
    "verbosa": render_verbosa,
}


def main() -> int:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=here / "eval_set.json")
    parser.add_argument(
        "--lexicon", type=Path, default=here / ".." / "data_collection" / "lexicon.yaml"
    )
    parser.add_argument("--json", type=Path, default=here / "bias_report.json")
    args = parser.parse_args()

    from judge import Judge

    eval_set = json.loads(args.eval_set.read_text(encoding="utf-8"))
    sistema = SistemaLexico(args.lexicon)
    juez = Judge()

    print(f"Juez: {juez.modelo_id} | rúbrica v{juez.version_rubrica} | device={juez.device}")
    print(f"Sesgo evaluado: verbosidad | {len(eval_set)} ejemplos x {len(RENDERS)} longitudes\n")

    # Las predicciones se calculan UNA vez: lo único que cambia entre las tres
    # condiciones es cómo se escribe la misma respuesta.
    predicciones = [sistema(ej["input"]) for ej in eval_set]

    resultados: dict[str, list[int]] = {k: [] for k in RENDERS}
    longitudes: dict[str, list[int]] = {k: [] for k in RENDERS}
    for nombre, render in RENDERS.items():
        print(f"Puntuando condición '{nombre}'...", flush=True)
        for ej, pred in zip(eval_set, predicciones):
            senal = render(pred)
            longitudes[nombre].append(len(senal))
            resultados[nombre].append(juez.score(ej["input"]["text"], ej["criterio"], senal).nivel)

    medias = {k: sum(v) / len(v) for k, v in resultados.items()}
    largos = {k: sum(v) / len(v) for k, v in longitudes.items()}
    delta = medias["verbosa"] - medias["escueta"]
    cambian = sum(1 for a, b in zip(resultados["escueta"], resultados["verbosa"]) if a != b)

    print("\n" + "=" * 70)
    print("SESGO DE VERBOSIDAD DEL JUEZ")
    print("=" * 70)
    print(f"{'condición':<12}{'long. media':>13}{'nota media':>13}")
    for k in RENDERS:
        print(f"{k:<12}{largos[k]:>13.0f}{medias[k]:>13.2f}")
    print(f"\n  delta verbosa - escueta: {delta:+.2f} puntos de rúbrica")
    print(f"  ejemplos que cambian de nivel solo por la longitud: {cambian}/{len(eval_set)}")

    detectado = delta > 0.1
    print(
        f"\n  ¿Sesgo detectado? {'SÍ' if detectado else 'NO'} — "
        f"{'el juez premia la longitud sin más información' if detectado else 'la longitud no movió la nota de forma apreciable'}"
    )
    print("\n  MITIGACIÓN: el harness emite todas las señales con la plantilla FIJA")
    print("  (`render_senal`), así que ningún sistema puede ganar puntos escribiendo")
    print("  más largo. Es la condición del medio de esta tabla.")

    informe = {
        "sesgo": "verbosidad",
        "juez": juez.modelo_id,
        "rubrica_version": juez.version_rubrica,
        "n_ejemplos": len(eval_set),
        "condiciones": {
            k: {
                "longitud_media_chars": round(largos[k], 1),
                "nota_media": round(medias[k], 3),
                "niveles": resultados[k],
            }
            for k in RENDERS
        },
        "delta_verbosa_menos_escueta": round(delta, 3),
        "ejemplos_que_cambian_de_nivel": cambian,
        "detectado": detectado,
        "mitigacion": (
            "render_senal(): plantilla de longitud fija para todos los sistemas, así la "
            "longitud no lleva información y el juez no puede premiarla."
        ),
    }
    args.json.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nInforme -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
