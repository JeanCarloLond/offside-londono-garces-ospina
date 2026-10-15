"""¿Qué puede hacer de válvula? Dos candidatos, medidos sobre el DEV set.

    python calibrar_umbral.py

El ajuste del prompt (ajuste_prompt.py) mostró que el 1.5B, redactando, no sabe
callar a tiempo. Así que la decisión "¿hay en las noticias algo que responda?"
tiene que tomarla otra pieza. Dos candidatos, sobre los mismos 4 chunks que ve el
avanzado (híbrida + reranker):

    cross_encoder  el puntaje del mejor chunk según el reranker, con un umbral
    verificador    el mismo Qwen como clasificador sí/no por logits (veredicto.py):
                   P(sí) máxima entre los chunks, con el umbral natural de 0,5

Para cada uno: ¿separa las preguntas con respuesta de las que no la tienen?
Se mide con el dev set, nunca con el eval set.
-> resultados/valvula_dev_m3.csv
"""

from __future__ import annotations

import csv
import json

import config
from generador import Generador
from indice import Recuperador
from veredicto import SISTEMA_RELEVANCIA, Veredicto, usuario_relevancia


def main() -> int:
    dev = json.loads((config.RAIZ / "eval" / "dev_set_m3.json").read_text(encoding="utf-8"))
    rec, g = Recuperador(), Generador()
    v = Veredicto(g.model, g.tokenizer, g.device)
    filas = []
    for d in dev:
        q = d["pregunta"]
        pool = Recuperador.rrf([rec.dense(q, config.CANDIDATOS), rec.lexico(q, config.CANDIDATOS)])
        top = rec.reordenar(q, pool[: config.CANDIDATOS], config.TOP_K)
        p_si = [
            v.p_si(SISTEMA_RELEVANCIA, usuario_relevancia(rec.por_id[i].para_contexto, q))
            for i, _ in top
        ]
        filas.append(
            {
                "dev_id": d["dev_id"],
                "en_corpus": d["en_corpus"],
                "cross_encoder_max": round(top[0][1], 3),
                "verificador_max": round(max(p_si), 3),
                "verificador_abre": max(p_si) >= config.UMBRAL_VERIFICADOR,
                "mejor_doc": rec.por_id[top[0][0]].texto[:70],
            }
        )
        f = filas[-1]
        print(
            f"{f['dev_id']} en_corpus={f['en_corpus']!s:<5} cross={f['cross_encoder_max']:7.3f} "
            f"verif={f['verificador_max']:.2f}  {f['mejor_doc']}"
        )

    print()
    for col in ("cross_encoder_max", "verificador_max"):
        con = [f[col] for f in filas if f["en_corpus"]]
        sin = [f[col] for f in filas if not f["en_corpus"]]
        sep = "separables" if min(con) > max(sin) else "NO separables (se solapan)"
        print(
            f"{col:<18} con respuesta: min {min(con):7.3f} | sin respuesta: max {max(sin):7.3f} -> {sep}"
        )
    aciertos = sum(f["verificador_abre"] == f["en_corpus"] for f in filas)
    print(f"verificador con umbral {config.UMBRAL_VERIFICADOR}: {aciertos}/{len(filas)} aciertos")

    config.RESULTADOS.mkdir(parents=True, exist_ok=True)
    with (config.RESULTADOS / "valvula_dev_m3.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
