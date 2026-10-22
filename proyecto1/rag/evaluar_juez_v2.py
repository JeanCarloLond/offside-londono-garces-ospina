"""El juez v2 sobre el eval set de M3, y su sesgo de verbosidad con texto.

    python evaluar_juez_v2.py --modelo Qwen/Qwen2.5-3B-Instruct --4bit
    python evaluar_juez_v2.py --solo-sesgo

Sale del feedback del M2 (ver eval/judge_razonado.py y eval/judge_rubric_v2.yaml).
No reemplaza al juez v1: la columna D2 de la tabla evolutiva sigue siendo la de
M2, para comparar con esa entrega. Esto añade una columna D2·v2, calculada igual
para todas las filas con la misma función harness(), cambiando solo el juez y lo
que ve (render_senal_y_texto: la señal más la respuesta en prosa).

Además mide el sesgo que el feedback anticipó: con texto, ¿el juez premia la
longitud? Las mismas respuestas del sistema final, puntuadas tres veces:
    original          la respuesta tal cual
    relleno           + una frase larga sin información, con el recorte de 240 caracteres
    relleno_sin_tope  + la misma frase, sin recorte
-> resultados/scorecard_m3_juez_v2.csv, resultados/juez_v2_m3.json,
   resultados/sesgo_verbosidad_texto_m3.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys

import config

sys.path.insert(0, str(config.RAIZ / "eval"))
from evaluar_m3 import cargar_respuestas, reproductor  # noqa: E402
from harness import (  # noqa: E402
    diagnostico_juez,
    harness,
    render_senal,
    render_senal_y_texto,
)
from judge_razonado import JudgeRazonado  # noqa: E402

SISTEMAS = ["m2_lexico", "m2_lora", "llm_sin_rag", "rag_ingenuo", "rag_avanzado", "sistema_final"]
RELLENO = (
    " Es una noticia relevante para el equipo y conviene seguir su evolución en los próximos "
    "días, porque cualquier cambio en la disponibilidad de la plantilla puede influir en el "
    "rendimiento del equipo y en la forma en que se plantee el próximo partido."
)


def sesgo_verbosidad(juez, eval_set: list[dict]) -> list[dict]:
    respuestas = cargar_respuestas("sistema_final")
    filas = []
    for e in eval_set:
        r = respuestas[e["eval_id"]]
        if not r.get("respuesta") or r["respuesta"].startswith("No encuentro"):
            continue
        largo = {**r, "respuesta": r["respuesta"] + RELLENO}
        condiciones = {
            "original": render_senal_y_texto(r),
            "relleno": render_senal_y_texto(largo),
            "relleno_sin_tope": f"{render_senal(largo)}\nRESPUESTA: {largo['respuesta']}",
        }
        fila = {"eval_id": e["eval_id"]}
        for nombre, senal in condiciones.items():
            fila[nombre] = juez.score(e["input"]["text"], e["criterio"], senal).nivel
            fila[f"chars_{nombre}"] = len(senal)
        filas.append(fila)
        print(f"  sesgo {e['eval_id']}: {[fila[k] for k in condiciones]}", flush=True)
    return filas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--4bit", dest="cuantizar", action="store_true")
    parser.add_argument("--sin-razonar", action="store_true")
    parser.add_argument("--solo-sesgo", action="store_true")
    args = parser.parse_args()

    eval_set = json.loads(config.EVAL_SET.read_text(encoding="utf-8"))
    juez = JudgeRazonado(
        modelo=args.modelo, razonar=not args.sin_razonar, cuantizar_4bit=args.cuantizar
    )
    print(f"Juez v2: {juez.nombre}", flush=True)
    salida = {"juez": juez.nombre, "rubrica_version": juez.version_rubrica}

    if not args.solo_sesgo:
        salida["sanity_check"] = juez.sanity_check()
        tarjetas, filas_csv = [], []
        for nombre in SISTEMAS:
            print(f"harness v2 · {nombre}", flush=True)
            t = harness(
                eval_set,
                reproductor(cargar_respuestas(nombre), eval_set),
                nombre,
                juez,
                render=render_senal_y_texto,
            )
            tarjetas.append(t)
            filas_csv.append(
                {
                    "sistema": nombre,
                    "d2_juez_v2_media": round(t.d2_juez_media, 3),
                    "d2_juez_v2_distribucion": json.dumps(t.d2_juez_distribucion),
                    "juez": juez.nombre,
                }
            )
            print(f"   media {t.d2_juez_media:.2f}", flush=True)
        salida["poder_discriminante"] = diagnostico_juez(tarjetas)
        salida["filas"] = {t.sistema: t.filas for t in tarjetas}
        with (config.RESULTADOS / "scorecard_m3_juez_v2.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(filas_csv[0]))
            w.writeheader()
            w.writerows(filas_csv)

    sesgo = sesgo_verbosidad(juez, eval_set)
    resumen = {
        k: round(statistics.mean(f[k] for f in sesgo), 3)
        for k in ("original", "relleno", "relleno_sin_tope")
    }
    resumen["cambian_con_relleno"] = sum(f["relleno"] != f["original"] for f in sesgo)
    resumen["n"] = len(sesgo)
    salida["sesgo_verbosidad"] = resumen
    print("sesgo de verbosidad:", resumen)
    with (config.RESULTADOS / "sesgo_verbosidad_texto_m3.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(sesgo[0]))
        w.writeheader()
        w.writerows(sesgo)
    (config.RESULTADOS / "juez_v2_m3.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
