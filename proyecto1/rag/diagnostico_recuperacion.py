"""¿Llega el chunk correcto? Diagnóstico de la recuperación, sin generar nada.

    python diagnostico_recuperacion.py             # recuperadores sin LLM
    python diagnostico_recuperacion.py --con-llm   # + HyDE y multi-query (necesitan el generador)

Para cada pregunta del eval set con respuesta en el corpus mira en qué posición
aparece el primer documento relevante (`docs_relevantes`), y resume con
recall@k (¿está en el top-k?) y MRR. Es la mitad del consejo del enunciado:
"¿no llegó el chunk?, ¿llegó y el modelo lo ignoró?". Esto responde la primera
pregunta; la segunda la responde el harness.

Produce:
    resultados/embeddings_m3.csv    tres modelos de embeddings, mismo recuperador dense
    resultados/recuperacion_m3.csv  rango del primer relevante por pregunta y recuperador
"""

from __future__ import annotations

import argparse
import csv
import json

import config
from indice import Recuperador

EMBEDDINGS_CANDIDATOS = [
    "intfloat/multilingual-e5-small",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-base",
]
KS = [1, 2, 4, 10]
PROFUNDIDAD = 50


def rango(ids: list[str], relevantes: set[str]) -> int | None:
    for i, d in enumerate(ids, start=1):
        if d in relevantes:
            return i
    return None


def resumir(rangos: list[int | None]) -> dict:
    n = len(rangos)
    fila = {f"recall@{k}": sum(1 for r in rangos if r and r <= k) / n for k in KS}
    fila["mrr"] = sum(1 / r for r in rangos if r) / n
    fila["no_llega_en_50"] = sum(1 for r in rangos if r is None)
    return fila


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--con-llm", action="store_true")
    parser.add_argument("--sin-comparar-embeddings", action="store_true")
    args = parser.parse_args()

    eval_set = json.loads(config.EVAL_SET.read_text(encoding="utf-8"))
    casos = [e for e in eval_set if e["en_corpus"]]
    config.RESULTADOS.mkdir(parents=True, exist_ok=True)

    # 1 · Embeddings: mismo recuperador dense, tres modelos.
    if not args.sin_comparar_embeddings:
        filas = []
        for modelo in EMBEDDINGS_CANDIDATOS:
            rec = Recuperador(modelo, con_reranker=False)
            rangos = [
                rango(rec.dense(e["input"]["text"], PROFUNDIDAD), set(e["docs_relevantes"]))
                for e in casos
            ]
            filas.append({"modelo": modelo, **resumir(rangos)})
            print(
                f"{modelo:<62} "
                + "  ".join(f"{k}={v:.2f}" for k, v in filas[-1].items() if k != "modelo")
            )
        with (config.RESULTADOS / "embeddings_m3.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(
                {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in filas
            )

    # 2 · Recuperadores sobre el modelo elegido.
    rec = Recuperador()
    modos = {
        "dense": lambda q: rec.dense(q, PROFUNDIDAD),
        "bm25": lambda q: rec.lexico(q, PROFUNDIDAD),
        "hibrido": lambda q: Recuperador.rrf(
            [rec.dense(q, config.CANDIDATOS), rec.lexico(q, config.CANDIDATOS)]
        ),
        "dense+rerank": lambda q: [
            i for i, _ in rec.reordenar(q, rec.dense(q, config.CANDIDATOS), config.CANDIDATOS)
        ],
        "hibrido+rerank": lambda q: [
            i
            for i, _ in rec.reordenar(
                q,
                Recuperador.rrf(
                    [rec.dense(q, config.CANDIDATOS), rec.lexico(q, config.CANDIDATOS)]
                )[: config.CANDIDATOS],
                config.CANDIDATOS,
            )
        ],
    }
    if args.con_llm:
        from generador import Generador
        from sistemas import Config, SistemaRAG

        g = Generador()
        for nombre, cfg in {
            "hyde": Config("hyde", hyde=True),
            "multiquery": Config("mq", multiquery=True),
            "hibrido+hyde+rerank": Config("x", hibrido=True, hyde=True, rerank=True),
        }.items():
            s = SistemaRAG(cfg, rec, g)
            # k = CANDIDATOS para ver el rango dentro de todo el pool, no solo el top-4
            modos[nombre] = (lambda s_: lambda q: s_.recuperar(q, k=config.CANDIDATOS)[0])(s)

    por_pregunta, resumen = [], []
    for nombre, fn in modos.items():
        rangos = []
        for e in casos:
            r = rango(fn(e["input"]["text"]), set(e["docs_relevantes"]))
            rangos.append(r)
            por_pregunta.append(
                {"eval_id": e["eval_id"], "recuperador": nombre, "rango_primer_relevante": r or ""}
            )
        resumen.append({"recuperador": nombre, **resumir(rangos)})

    print()
    print(f"{'recuperador':<22}" + "".join(f"{k:>11}" for k in resumen[0] if k != "recuperador"))
    for r in resumen:
        print(
            f"{r['recuperador']:<22}"
            + "".join(
                f"{v:>11.2f}" if isinstance(v, float) else f"{v:>11}"
                for k, v in r.items()
                if k != "recuperador"
            )
        )

    print("\nRango del primer documento relevante (– = no está en el top-50):")
    ids = [e["eval_id"] for e in casos]
    print(f"{'':<8}" + "".join(f"{n[:14]:>16}" for n in modos))
    for eid in ids:
        celdas = [
            next(
                p["rango_primer_relevante"]
                for p in por_pregunta
                if p["eval_id"] == eid and p["recuperador"] == n
            )
            for n in modos
        ]
        print(f"{eid:<8}" + "".join(f"{(c if c != '' else '–'):>16}" for c in celdas))

    with (config.RESULTADOS / "recuperacion_m3.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(resumen[0]))
        w.writeheader()
        w.writerows(
            {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in resumen
        )
    with (config.RESULTADOS / "recuperacion_por_pregunta_m3.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(por_pregunta[0]))
        w.writeheader()
        w.writerows(por_pregunta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
