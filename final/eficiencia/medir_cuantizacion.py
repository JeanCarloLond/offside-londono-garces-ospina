"""La técnica de eficiencia, medida antes y después en la misma máquina.

    python final/eficiencia/medir_cuantizacion.py            # responder (fase 1)
    python final/eficiencia/medir_cuantizacion.py --evaluar  # harness de M2 (fase 2)

Antes: el sistema final del M3 con el generador en fp16.
Después: el mismo sistema, con el generador en int4 (bitsandbytes NF4).
El criterio de aceptación se escribió antes de correr esto: criterio.md.

Se corren las dos versiones en la misma sesión y la misma GPU, con el mismo eval set y el
mismo prompt base, y se guarda por versión:
    respuestas/<version>.jsonl   las respuestas, con su latencia (como en el M3)
    memoria.json                 memoria del generador: pesos y pico de GPU al responder
Con --evaluar se pasa el harness de M2 (con su juez v1) y se escribe tabla_eficiencia.csv con
calidad, latencia p50/p95, memoria y las pruebas que no miden exactitud (válvula y adversariales).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ / "proyecto1" / "rag"))
sys.path.insert(0, str(RAIZ / "proyecto1" / "eval"))

import config  # noqa: E402

RES = AQUI / "resultados"
VERSIONES = {"antes_fp16": False, "despues_int4": True}


def percentil(valores: list[float], p: float) -> float:
    v = sorted(valores)
    k = (len(v) - 1) * p
    i = int(k)
    return v[i] if i + 1 >= len(v) else v[i] + (v[i + 1] - v[i]) * (k - i)


def responder(versiones: list[str]) -> None:
    import random

    import numpy as np
    import torch
    from generador import Generador
    from indice import Recuperador
    from sistemas import FINAL, SistemaRAG

    eval_set = json.loads(config.EVAL_SET.read_text(encoding="utf-8"))
    rec = Recuperador()
    (RES / "respuestas").mkdir(parents=True, exist_ok=True)
    ruta_mem = RES / "memoria.json"
    memoria = json.loads(ruta_mem.read_text(encoding="utf-8")) if ruta_mem.exists() else {}
    for nombre in versiones:
        cuantizar = VERSIONES[nombre]
        random.seed(config.SEED)
        np.random.seed(config.SEED)
        torch.manual_seed(config.SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        g = Generador(cuantizar=cuantizar)
        pesos_mb = g.model.get_memory_footprint() / 2**20
        sistema = SistemaRAG(FINAL, rec, g)
        sistema("¿Qué jugadores están lesionados?")  # calentamiento, no cuenta
        t0 = time.perf_counter()
        with (RES / "respuestas" / f"{nombre}.jsonl").open("w", encoding="utf-8") as f:
            for ej in eval_set:
                salida = sistema(ej["input"]["text"])
                f.write(
                    json.dumps(
                        {"eval_id": ej["eval_id"], "pregunta": ej["input"]["text"], **salida},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        memoria[nombre] = {
            "pesos_generador_mb": round(pesos_mb, 1),
            "pico_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "gpu": torch.cuda.get_device_name(0),
        }
        print(f"{nombre}: {time.perf_counter() - t0:.0f} s · {memoria[nombre]}", flush=True)
        del sistema, g
        ruta_mem.write_text(json.dumps(memoria, indent=2), encoding="utf-8")


def evaluar() -> None:
    from evaluar_m3 import reproductor
    from harness import harness
    from judge import Judge

    eval_set = json.loads(config.EVAL_SET.read_text(encoding="utf-8"))
    memoria = json.loads((RES / "memoria.json").read_text(encoding="utf-8"))
    juez = Judge()
    filas = []
    fuera = [e for e in eval_set if not e["en_corpus"]]
    dentro = [e for e in eval_set if e["en_corpus"]]
    adv = [e for e in eval_set if e["adversarial"]]
    for nombre in VERSIONES:
        respuestas = {
            r["eval_id"]: r
            for r in map(
                json.loads, (RES / "respuestas" / f"{nombre}.jsonl").open(encoding="utf-8")
            )
        }
        t = harness(eval_set, reproductor(respuestas, eval_set), nombre, juez)
        lat = [r["latencia_s"] for r in respuestas.values()]
        calla = {k: r["respuesta"].startswith("No encuentro") for k, r in respuestas.items()}
        acierta = {f["eval_id"]: f["accionable"] for f in t.filas}
        filas.append(
            {
                "version": nombre,
                "d1_f1_macro": round(t.d1_f1_macro, 4),
                "d2_juez_v1": round(t.d2_juez_media, 3),
                "d3_tasa_accionable": round(t.d3_tasa_accionable, 4),
                "latencia_p50_s": round(statistics.median(lat), 2),
                "latencia_p95_s": round(percentil(lat, 0.95), 2),
                "pesos_generador_mb": memoria[nombre]["pesos_generador_mb"],
                "pico_gpu_mb": memoria[nombre]["pico_gpu_mb"],
                "valvula_correcta": f"{sum(calla[e['eval_id']] for e in fuera)}/{len(fuera)}",
                "valvula_falsa": f"{sum(calla[e['eval_id']] for e in dentro)}/{len(dentro)}",
                "accionable_adversariales": f"{sum(acierta[e['eval_id']] for e in adv)}/{len(adv)}",
                "senal_falsa_alto_impacto": t.d3_detalle["senal_falsa_alto_impacto"],
                "signo_invertido": t.d3_detalle["signo_invertido"],
                "rumor_como_hecho": t.d3_detalle["rumor_como_hecho"],
            }
        )
        print(filas[-1], flush=True)
    with (RES / "tabla_eficiencia.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluar", action="store_true")
    parser.add_argument(
        "--version",
        choices=list(VERSIONES),
        action="append",
        help="una por proceso: en una GPU de 4 GB, cargar int4 tras fp16 en el mismo proceso falla",
    )
    args = parser.parse_args()
    evaluar() if args.evaluar else responder(args.version or list(VERSIONES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
