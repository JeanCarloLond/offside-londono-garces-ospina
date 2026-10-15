"""Fase 2 de la evaluación: el harness de M2 y RAGAS sobre lo que respondió cada sistema.

    python evaluar_m3.py                 # harness (3 dimensiones) + RAGAS
    python evaluar_m3.py --sin-ragas
    python evaluar_m3.py --sin-juez      # dimensiones 1 y 3 solamente (sin GPU)

Lee resultados/respuestas/<sistema>.jsonl (fase 1, correr_sistemas.py) y escribe:
    resultados/scorecard_m3.csv / .json   la tabla evolutiva, con latencia mediana
    resultados/ragas_m3.csv               las cuatro métricas por sistema
    resultados/ragas_m3_por_caso.csv      y por pregunta

La vara no cambia
-----------------
Se llama a `harness(eval_set, sistema, nombre, juez)` de eval/harness.py tal
cual, con el juez de M2 (eval/judge.py, rúbrica v1) sin tocar. El "sistema" que
recibe es la reproducción de lo que el sistema real respondió en la fase 1: el
harness lo invoca pregunta por pregunta como a cualquier otro sistema.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys

import config

sys.path.insert(0, str(config.RAIZ / "eval"))
from harness import diagnostico_juez, harness  # noqa: E402

# El orden es el de la tabla evolutiva. Las filas que no tengan respuestas
# grabadas se omiten.
TABLA = [
    ("m2_lexico", "Línea base M2 · léxico (sin RAG)"),
    ("m2_lora", "Línea base M2 · LoRA de M1 (sin RAG)"),
    ("llm_sin_rag", "Mismo LLM, sin recuperación"),
    ("rag_ingenuo", "RAG ingenuo (sesión 7)"),
    ("rag_avanzado", "RAG avanzado"),
    ("sistema_final", "Sistema final (con herramienta)"),
]
ABLACION = [
    "rag_ingenuo_prompt_v1",
    "llm_sin_rag_prompt_v1",
    "rag_hibrido",
    "rag_rerank",
    "rag_hyde",
    "rag_verificado",
    "rag_avanzado_sin_hibrido",
    "rag_avanzado_sin_rerank",
    "rag_avanzado_sin_verificador",
]
RAGAS_SISTEMAS = ["rag_ingenuo", "rag_avanzado", "sistema_final"]


def cargar_respuestas(nombre: str) -> dict | None:
    ruta = config.RESULTADOS / "respuestas" / f"{nombre}.jsonl"
    if not ruta.exists():
        return None
    return {r["eval_id"]: r for r in map(json.loads, ruta.open(encoding="utf-8"))}


def reproductor(respuestas: dict, eval_set: list[dict]):
    """Un sistema que devuelve lo que el sistema real respondió a esa pregunta."""
    por_texto = {e["input"]["text"]: respuestas[e["eval_id"]] for e in eval_set}

    def _sistema(entrada: dict) -> dict:
        return por_texto[entrada["text"]]

    return _sistema


def extras(respuestas: dict, eval_set: list[dict]) -> dict:
    """Lo que el harness de M2 no mide y M3 pide: latencia, válvula y herramienta."""
    lat = [r["latencia_s"] for r in respuestas.values()]
    fuera = [e for e in eval_set if not e["en_corpus"]]
    dentro = [e for e in eval_set if e["en_corpus"]]
    calla = {k: r.get("respuesta", "").startswith("No encuentro") for k, r in respuestas.items()}
    usa = {k: bool(r.get("herramientas")) for k, r in respuestas.items()}
    necesita = [e for e in eval_set if e["necesita_herramienta"]]
    return {
        "latencia_mediana_s": round(statistics.median(lat), 3),
        "latencia_p90_s": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 3),
        "valvula_correcta": f"{sum(calla[e['eval_id']] for e in fuera)}/{len(fuera)}",
        "valvula_falsa": f"{sum(calla[e['eval_id']] for e in dentro)}/{len(dentro)}",
        "herramienta_cuando_hace_falta": f"{sum(usa[e['eval_id']] for e in necesita)}/{len(necesita)}",
        "herramienta_sin_hacer_falta": sum(
            usa[e["eval_id"]] for e in eval_set if not e["necesita_herramienta"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sin-juez", action="store_true")
    parser.add_argument("--sin-ragas", action="store_true")
    parser.add_argument("--solo-ragas", action="store_true")
    args = parser.parse_args()

    eval_set = json.loads(config.EVAL_SET.read_text(encoding="utf-8"))
    nombres = [n for n, _ in TABLA] + ABLACION
    grabados = {n: r for n in nombres if (r := cargar_respuestas(n)) is not None}
    etiquetas = dict(TABLA)

    juez, sanity = None, None
    if not args.sin_juez and not args.solo_ragas:
        from judge import Judge

        juez = Judge()
        print(f"Juez D2: {juez.modelo_id} | rúbrica v{juez.version_rubrica} | {juez.device}")
        sanity = juez.sanity_check()

    if not args.solo_ragas:
        tarjetas, filas_csv = [], []
        for nombre, respuestas in grabados.items():
            print(f"harness · {nombre}", flush=True)
            t = harness(eval_set, reproductor(respuestas, eval_set), nombre, juez)
            tarjetas.append(t)
            filas_csv.append(
                {
                    "sistema": nombre,
                    "fila": etiquetas.get(nombre, "ablación"),
                    **{k: v for k, v in t.como_fila_csv().items() if k != "sistema"},
                    **extras(respuestas, eval_set),
                }
            )
        config.RESULTADOS.mkdir(parents=True, exist_ok=True)
        with (config.RESULTADOS / "scorecard_m3.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(filas_csv[0]))
            w.writeheader()
            w.writerows(filas_csv)
        payload = {
            "eval_set": config.EVAL_SET.name,
            "n_ejemplos": len(eval_set),
            "juez": None
            if juez is None
            else {
                "modelo": juez.modelo_id,
                "rubrica_version": juez.version_rubrica,
                "sanity_check": sanity,
                "poder_discriminante": diagnostico_juez(tarjetas),
            },
            "sistemas": {
                t.sistema: {"fila_csv": fila, "filas": t.filas}
                for t, fila in zip(tarjetas, filas_csv)
            },
        }
        (config.RESULTADOS / "scorecard_m3.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        cols = [
            "d1_f1_macro",
            "d2_juez_media",
            "d3_tasa_accionable",
            "latencia_mediana_s",
            "valvula_correcta",
            "valvula_falsa",
            "herramienta_cuando_hace_falta",
            "herramienta_sin_hacer_falta",
        ]
        print(f"\n{'sistema':<30}" + "".join(f"{c[:14]:>16}" for c in cols))
        for f in filas_csv:
            print(f"{f['sistema']:<30}" + "".join(f"{str(f[c]):>16}" for c in cols))

    if args.sin_ragas:
        return 0
    # ------------------------------------------------------------------ RAGAS
    from ragas_local import JuezRagas

    if juez is not None:
        jr = JuezRagas(juez.model, juez.tokenizer, juez.device)
    else:
        jr = JuezRagas()
    print(f"\nJuez RAGAS: {jr.modelo_id} | {jr.device}")
    s = jr.sanity_check()
    print(f"sanity check del juez RAGAS: {s['aciertos']}/{s['total']}")
    for c in s["casos"]:
        print(
            f"   {'OK ' if c['acierta'] else 'XX '}{c['tipo']:<20} p(sí)={c['p_si']:.2f}  {c['afirmacion']}"
        )

    casos = [e for e in eval_set if e["en_corpus"]]
    por_caso, resumen = [], []
    for nombre in RAGAS_SISTEMAS:
        respuestas = grabados.get(nombre)
        if respuestas is None:
            continue
        print(f"RAGAS · {nombre}", flush=True)
        filas = [{"sistema": nombre, **jr.evaluar(e, respuestas[e["eval_id"]])} for e in casos]
        por_caso += filas

        def media(clave, filas=filas):
            vals = [f[clave] for f in filas if f.get(clave) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        resumen.append(
            {
                "sistema": nombre,
                "n_casos": len(filas),
                "faithfulness": media("faithfulness"),
                "n_faithfulness": sum(f["faithfulness"] is not None for f in filas),
                "answer_relevancy": media("answer_relevancy"),
                "context_precision": media("context_precision"),
                "context_recall": media("context_recall"),
                "context_precision_ids": media("context_precision_ids"),
                "juez": jr.modelo_id,
            }
        )
        print("   ", resumen[-1])

    with (config.RESULTADOS / "ragas_m3.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(resumen[0]))
        w.writeheader()
        w.writerows(resumen)
    claves = [
        "sistema",
        "eval_id",
        "valvula",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "context_precision_ids",
    ]
    with (config.RESULTADOS / "ragas_m3_por_caso.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=claves, extrasaction="ignore")
        w.writeheader()
        w.writerows(por_caso)
    (config.RESULTADOS / "ragas_m3_detalle.json").write_text(
        json.dumps({"sanity_check": s, "casos": por_caso}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
