"""Ajuste del prompt base sobre el DEV set, nunca sobre el eval set.

    python ajuste_prompt.py

El prompt base es el mismo para todas las filas de la tabla, así que hay que
fijarlo antes de comparar nada. Ajustarlo mirando el eval set sería sobreajustar
la vara. Por eso hay un dev set aparte (eval/dev_set_m3.json): 14 preguntas sobre
otras noticias del corpus, 3 de ellas sin respuesta.

Compara las versiones de generador.PROMPTS con el RAG ingenuo y cuenta:
    aciertos          categoría correcta
    valvula_falsa     calló teniendo la respuesta en el corpus
    valvula_correcta  calló cuando la respuesta no estaba
-> resultados/ajuste_prompt_m3.csv
"""

from __future__ import annotations

import argparse
import csv
import json

import config
import generador as gen
from indice import Recuperador
from sistemas import Config, SistemaRAG


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("versiones", nargs="*", help="por defecto, todas las de generador.PROMPTS")
    args = parser.parse_args()
    dev = json.loads((config.RAIZ / "eval" / "dev_set_m3.json").read_text(encoding="utf-8"))
    rec, g = Recuperador(con_reranker=False), gen.Generador()
    filas = []
    for version in args.versiones or list(gen.PROMPTS):
        s = SistemaRAG(Config(f"ingenuo_{version}", prompt=version), rec, g)
        aciertos = falsa = correcta = 0
        for d in dev:
            r = s(d["pregunta"])
            calla = r["respuesta"] == gen.VALVULA
            aciertos += r["category"] == d["categoria"]
            falsa += calla and d["en_corpus"]
            correcta += calla and not d["en_corpus"]
            print(f"  {version} {d['dev_id']} {r['category']:<20} {r['respuesta'][:90]}")
        n_in = sum(d["en_corpus"] for d in dev)
        filas.append(
            {
                "prompt": version,
                "aciertos_categoria": f"{aciertos}/{len(dev)}",
                "valvula_falsa": f"{falsa}/{n_in}",
                "valvula_correcta": f"{correcta}/{len(dev) - n_in}",
            }
        )
        print(filas[-1])
    config.RESULTADOS.mkdir(parents=True, exist_ok=True)
    ruta = config.RESULTADOS / "ajuste_prompt_m3.csv"
    previas = []
    if ruta.exists():
        with ruta.open(encoding="utf-8") as f:
            previas = [
                r for r in csv.DictReader(f) if r["prompt"] not in {x["prompt"] for x in filas}
            ]
    with ruta.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(sorted(previas + filas, key=lambda r: r["prompt"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
