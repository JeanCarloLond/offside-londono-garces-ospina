"""Fase 1 de la evaluación: correr cada sistema sobre el eval set y registrar todo.

    python correr_sistemas.py                         # todos los de sistemas.SISTEMAS
    python correr_sistemas.py rag_ingenuo rag_rerank  # solo esos

Por cada sistema escribe resultados/respuestas/<sistema>.jsonl con, por pregunta:
la señal, la respuesta en prosa, los chunks que vio el generador, las llamadas a
herramientas y la latencia. Ese registro es la entrada de la fase 2
(evaluar_m3.py): el harness de M2 y RAGAS.

Por qué dos fases: el generador y el juez son modelos de 1.5B y en una GPU de
4 GB no caben a la vez. Separar "responder" de "evaluar" además asegura que el
juez puntúa exactamente lo que el sistema respondió (y que la latencia medida no
incluye al juez).

La latencia se mide por consulta, de punta a punta (recuperar + transformar +
generar + herramientas), después de una consulta de calentamiento que no cuenta.
Los baselines de M2 (léxico y LoRA) también se cronometran.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import config
import sistemas

sys.path.insert(0, str(config.RAIZ / "eval"))
from harness import SistemaLexico, SistemaLoRA  # noqa: E402

CALENTAMIENTO = "¿Qué jugadores están lesionados?"


def fijar_semillas() -> None:
    import numpy as np
    import torch

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)


def baseline_m2(nombre: str):
    """Los sistemas del scorecard de M2, que reciben la PREGUNTA como si fuera un fragmento."""
    lexicon = config.RAIZ / "data_collection" / "lexicon.yaml"
    if nombre == "m2_lexico":
        s = SistemaLexico(lexicon)
    else:
        s = SistemaLoRA(
            config.RAIZ / "m1_lora_adapter_holdout",
            lexicon,
            "dccuchile/bert-base-spanish-wwm-cased",
        )

    def _sistema(pregunta: str) -> dict:
        t0 = time.perf_counter()
        r = s({"text": pregunta})
        return {
            **r,
            "respuesta": "",
            "contextos": [],
            "ids": [],
            "herramientas": [],
            "latencia_s": time.perf_counter() - t0,
        }

    return _sistema


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sistemas", nargs="*")
    args = parser.parse_args()
    fijar_semillas()

    nombres = args.sistemas or ["m2_lexico", "m2_lora", *sistemas.SISTEMAS]
    eval_set = json.loads(config.EVAL_SET.read_text(encoding="utf-8"))
    carpeta = config.RESULTADOS / "respuestas"
    carpeta.mkdir(parents=True, exist_ok=True)

    for nombre in nombres:
        if nombre.startswith("m2_"):
            sistema = baseline_m2(nombre)
        else:
            cfg = sistemas.SISTEMAS.get(nombre) or sistemas.CANDIDATAS[nombre]
            sistema = sistemas.construir(cfg)
        sistema(CALENTAMIENTO)

        t0 = time.perf_counter()
        with (carpeta / f"{nombre}.jsonl").open("w", encoding="utf-8") as f:
            for ej in eval_set:
                salida = sistema(ej["input"]["text"])
                f.write(
                    json.dumps(
                        {"eval_id": ej["eval_id"], "pregunta": ej["input"]["text"], **salida},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(
            f"{nombre:<26} {len(eval_set)} preguntas en {time.perf_counter() - t0:6.1f} s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
