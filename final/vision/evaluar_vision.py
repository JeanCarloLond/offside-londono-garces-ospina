"""Evaluación del componente visual con el harness de M2.

    python final/vision/evaluar_vision.py --leer          # fase 1: el modelo de visión lee las capturas
    python final/vision/evaluar_vision.py --responder     # fase 2: los sistemas responden
    python final/vision/evaluar_vision.py --evaluar       # fase 3: harness, invarianza y válvula

En una GPU de 4 GB no caben a la vez el modelo de visión y el generador, así que va en
fases (como en el M3): las lecturas de la fase 1 quedan guardadas y la fase 2 las usa.

Sistemas que responden los casos con imagen (eval_set_vision.json):
  m2_lexico, m2_lora   las líneas base de M2: solo ven la pregunta
  m3_sin_vision        el sistema final del M3: solo ve la pregunta, no la imagen
  final_con_vision     el mismo sistema con el texto que leyó el componente visual

Pruebas que no miden exactitud (las que pide el enunciado sobre la versión final):
  invarianza   la misma captura en tres variantes (pequeña y comprimida, tema oscuro,
               rotada): ¿da la misma señal que con la original?
  válvula      V-06 (imagen sin fútbol): ¿calla?  V-07 (captura falsa): ¿la da por hecho?

-> resultados/lecturas.json, resultados/respuestas_<sistema>.jsonl, resultados/vision_m4.csv,
   resultados/invarianza.csv
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
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(RAIZ / "proyecto1" / "rag"))
sys.path.insert(0, str(RAIZ / "proyecto1" / "eval"))

RES = AQUI / "resultados"
EVAL = json.loads((AQUI / "eval_set_vision.json").read_text(encoding="utf-8"))


def imagenes() -> dict[str, str]:
    """Todas las imágenes a leer: la original de cada caso y sus variantes."""
    out = {}
    for c in EVAL:
        out[c["eval_id"]] = c["input"]["imagen"]
        for nombre, ruta in c["variantes"].items():
            out[f"{c['eval_id']}_{nombre}"] = ruta
    return out


def leer(cuantizar: bool) -> None:
    from PIL import Image
    from vision import Vision

    v = Vision(cuantizar=cuantizar)
    lecturas = {}
    for clave, ruta in imagenes().items():
        t = time.perf_counter()
        lec = v.leer(Image.open(AQUI / ruta))
        lecturas[clave] = {
            "texto": lec.texto,
            "es_futbol": lec.es_futbol,
            "latencia_s": round(time.perf_counter() - t, 2),
        }
        print(clave, lecturas[clave], flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "lecturas.json").write_text(
        json.dumps(
            {"modelo": v.modelo_id, "cuantizado": cuantizar, "lecturas": lecturas},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def responder(cuantizar_generador: bool) -> None:
    import config
    from correr_sistemas import baseline_m2
    from generador import Generador
    from indice import Recuperador
    from sistemas import FINAL, SistemaRAG

    lect = json.loads((RES / "lecturas.json").read_text(encoding="utf-8"))["lecturas"]
    rec = Recuperador()
    g = Generador(cuantizar=cuantizar_generador)
    final = SistemaRAG(FINAL, rec, g)
    final("¿Qué jugadores están lesionados?")  # calentamiento

    def con_vision(clave: str, pregunta: str) -> dict:
        lec = lect[clave]
        t = time.perf_counter()
        if not lec["es_futbol"]:  # válvula visual, como en la app
            from generador import SIN_SENAL, VALVULA

            salida = {**SIN_SENAL, "respuesta": VALVULA, "ids": [], "herramientas": []}
        else:
            salida = final(pregunta, captura=lec["texto"])
        salida["latencia_s"] = time.perf_counter() - t + lec["latencia_s"]
        return salida

    sistemas = {
        "m2_lexico": baseline_m2("m2_lexico"),
        "m2_lora": baseline_m2("m2_lora"),
        "m3_sin_vision": final,
    }
    for nombre, fn in sistemas.items():
        _escribir(nombre, {c["eval_id"]: fn(c["input"]["text"]) for c in EVAL})
    # el sistema final con visión: casos originales y variantes
    filas = {}
    for c in EVAL:
        filas[c["eval_id"]] = con_vision(c["eval_id"], c["input"]["text"])
        for nombre in c["variantes"]:
            clave = f"{c['eval_id']}_{nombre}"
            filas[clave] = con_vision(clave, c["input"]["text"])
    _escribir("final_con_vision", filas)
    print("config:", config.GENERADOR, "int4" if cuantizar_generador else "fp16")


def _escribir(nombre: str, filas: dict) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    with (RES / f"respuestas_{nombre}.jsonl").open("w", encoding="utf-8") as f:
        for clave, salida in filas.items():
            limpio = {k: v for k, v in salida.items() if k != "texto_modelo"}
            f.write(json.dumps({"clave": clave, **limpio}, ensure_ascii=False, default=str) + "\n")
    print("ok", nombre, flush=True)


def _cargar(nombre: str) -> dict:
    with (RES / f"respuestas_{nombre}.jsonl").open(encoding="utf-8") as f:
        return {r["clave"]: r for r in map(json.loads, f)}


def evaluar(con_juez: bool) -> None:
    from evaluar_m3 import reproductor
    from harness import harness

    juez = None
    if con_juez:
        from judge import Judge

        juez = Judge()
    filas = []
    for nombre in ("m2_lexico", "m2_lora", "m3_sin_vision", "final_con_vision"):
        resp = _cargar(nombre)
        t = harness(
            EVAL, reproductor({c["eval_id"]: resp[c["eval_id"]] for c in EVAL}, EVAL), nombre, juez
        )
        lat = [resp[c["eval_id"]]["latencia_s"] for c in EVAL]
        filas.append(
            {
                "sistema": nombre,
                "n_casos": len(EVAL),
                "d1_f1_macro": round(t.d1_f1_macro, 4),
                "d2_juez_v1": "" if t.d2_juez_media is None else round(t.d2_juez_media, 3),
                "d3_tasa_accionable": round(t.d3_tasa_accionable, 4),
                "aciertos_categoria": sum(
                    f["pred_category"] == f["gold_category"] for f in t.filas
                ),
                "rumor_como_hecho": t.d3_detalle["rumor_como_hecho"],
                "latencia_mediana_s": round(statistics.median(lat), 2),
            }
        )
        print(filas[-1], flush=True)
    with (RES / "vision_m4.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    final = _cargar("final_con_vision")
    inv = []
    for c in EVAL:
        base = final[c["eval_id"]]
        for nombre in c["variantes"]:
            v = final[f"{c['eval_id']}_{nombre}"]
            inv.append(
                {
                    "eval_id": c["eval_id"],
                    "variante": nombre,
                    "categoria_original": base["category"],
                    "categoria_variante": v["category"],
                    "misma_senal": (base["category"], base["impact"])
                    == (v["category"], v["impact"]),
                    "misma_categoria": base["category"] == v["category"],
                }
            )
    with (RES / "invarianza.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inv[0]))
        w.writeheader()
        w.writerows(inv)
    print(
        f"invarianza: misma categoría en {sum(i['misma_categoria'] for i in inv)}/{len(inv)} "
        f"variantes; misma señal completa en {sum(i['misma_senal'] for i in inv)}/{len(inv)}"
    )
    print("V-06 (sin fútbol):", final["V-06"]["category"], "|", final["V-06"]["respuesta"][:100])
    print("V-07 (captura falsa):", final["V-07"]["category"], "|", final["V-07"]["respuesta"][:160])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leer", action="store_true")
    parser.add_argument("--responder", action="store_true")
    parser.add_argument("--evaluar", action="store_true")
    parser.add_argument("--vision-4bit", action="store_true", help="en GPUs de 4 GB")
    parser.add_argument("--generador-int4", action="store_true")
    parser.add_argument("--sin-juez", action="store_true")
    args = parser.parse_args()
    if args.leer:
        leer(args.vision_4bit)
    if args.responder:
        responder(args.generador_int4)
    if args.evaluar:
        evaluar(not args.sin_juez)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
