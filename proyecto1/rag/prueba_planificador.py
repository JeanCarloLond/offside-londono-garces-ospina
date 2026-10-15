"""Prueba aislada del planificador del agente, con noticias FICTICIAS.

    python prueba_planificador.py

En el corpus real solo hay dos bajas con duración numérica, y las dos están en el
eval set. Para ajustar y comprobar el planificador sin mirar el eval set, se le
pasan noticias inventadas (clubes ficticios) directamente, sin recuperación:
cuatro preguntas que necesitan la herramienta y cuatro que no.
-> resultados/prueba_planificador_m3.csv
"""

from __future__ import annotations

import csv

import config
import generador as gen
import herramientas as herr

CASOS = [
    # (noticia, pregunta, necesita_herramienta)
    (
        "(2026-09-02 · Marca · deportivo sauce) Lunar, baja para el derbi. El extremo del Deportivo "
        "Sauce estará seis semanas de baja por una rotura de fibras",
        "¿Podrá jugar Lunar el 20 de octubre?",
        True,
    ),
    (
        "(2026-08-15 · Marca · atletico ribera) Ferro se perderá entre uno y dos meses por una "
        "lesión de rodilla",
        "¿Cuenta el Atlético Ribera con Ferro para el 1 de septiembre?",
        True,
    ),
    (
        "(2026-10-01 · Marca · union costa) Quintana, cuatro meses fuera tras romperse el tobillo",
        "¿Estará Quintana disponible el 15 de marzo?",
        True,
    ),
    (
        "(2026-09-20 · Marca · real valle) Medrano estará diez días de baja por una sobrecarga",
        "¿Puede jugar Medrano el 25 de septiembre?",
        True,
    ),
    (
        "(2026-09-02 · Marca · deportivo sauce) Lunar, baja para el derbi. El extremo del Deportivo "
        "Sauce estará seis semanas de baja por una rotura de fibras",
        "¿Qué lesión tiene Lunar?",
        False,
    ),
    (
        "(2026-09-03 · Marca · atletico ribera) Ferro, expulsado ante el Sauce: se perderá el "
        "próximo partido",
        "¿Ferro está sancionado?",
        False,
    ),
    (
        "(2026-09-05 · Marca · union costa) Quintana se perderá toda la temporada por una rotura "
        "del cruzado",
        "¿Volverá Quintana esta temporada?",
        False,
    ),
    (
        "(2026-09-06 · Marca · real valle) El Real Valle presenta su nueva equipación",
        "¿Quién es el portero titular del Real Valle?",
        False,
    ),
]


MESES = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre"


def marcas_de_fecha(iso: str) -> list[str]:
    """Formas en que la respuesta puede mencionar lo que calculó la herramienta: la
    fecha de fin del plazo (2026-11-27 o "27 de noviembre") o los días de diferencia
    ("44 días antes"). No vale "diez días de baja": eso lo dice la noticia."""
    _, mes, dia = iso.split("-")
    return [iso, f"{int(dia)} de {MESES.split()[int(mes) - 1]}", "días antes", "días después"]


def main() -> int:
    g = gen.Generador()
    filas = []
    for modo, noticia, pregunta, necesita in [(m, *c) for m in ("llm", "regla") for c in CASOS]:
        llamadas, pasos = herr.planificar(g, pregunta, [noticia], modo=modo)
        validas = [h for h in llamadas if "error" not in h["resultado"]]
        filas.append(
            {
                "modo": modo,
                "pregunta": pregunta,
                "necesita": necesita,
                "llama": bool(llamadas),
                "llamada_valida": bool(validas),
                "pasos": pasos,
                "argumentos": [h["arguments"] for h in llamadas],
                "lectura": [
                    h["resultado"].get("lectura", h["resultado"].get("error")) for h in llamadas
                ],
            }
        )
        print(
            f"{modo:<6}{'NECESITA' if necesita else 'no      '} llama={bool(llamadas)!s:<5} {pregunta}"
        )
        for h in llamadas:
            print("     ", h["arguments"], "->", h["resultado"].get("lectura", h["resultado"]))
    for modo in ("llm", "regla"):
        fs = [f for f in filas if f["modo"] == modo]
        n_si = sum(f["necesita"] for f in fs)
        print(
            f"\nmodo {modo}: llama cuando hace falta "
            f"{sum(f['llama'] for f in fs if f['necesita'])}/{n_si} "
            f"(válida: {sum(f['llamada_valida'] for f in fs if f['necesita'])}/{n_si}) | "
            f"llama sin hacer falta: {sum(f['llama'] for f in fs if not f['necesita'])}/"
            f"{len(fs) - n_si}"
        )
    # ¿El generador USA el resultado? Respuesta final con el resultado sin fecha y
    # con la fecha de la consulta (la versión del sistema final).
    print("\nUso del resultado en la respuesta final:")
    for noticia, pregunta, necesita in CASOS:
        if not necesita:
            continue
        llamadas, _ = herr.planificar(g, pregunta, [noticia])
        fila = next(f for f in filas if f["modo"] == "regla" and f["pregunta"] == pregunta)
        for fechado in (False, True):
            ctx = [noticia, *herr.contexto_herramientas(llamadas, fechado=fechado)]
            r = gen.parsear_respuesta(g.generar(gen.mensajes_base(pregunta, ctx)))
            fin = [
                h["resultado"]["fecha_fin_plazo"] for h in llamadas if "lectura" in h["resultado"]
            ]
            usa = bool(fin) and any(m in r["respuesta"] for m in marcas_de_fecha(fin[0]))
            clave = "con_fecha" if fechado else "sin_fecha"
            fila[f"respuesta_{clave}"] = r["respuesta"]
            fila[f"usa_resultado_{clave}"] = usa
            print(f"  {clave:<9} usa={usa!s:<5} {pregunta} -> {r['respuesta'][:110]}")
    for f in filas:
        for clave in ("sin_fecha", "con_fecha"):
            f.setdefault(f"respuesta_{clave}", "")
            f.setdefault(f"usa_resultado_{clave}", "")

    config.RESULTADOS.mkdir(parents=True, exist_ok=True)
    with (config.RESULTADOS / "prueba_planificador_m3.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
