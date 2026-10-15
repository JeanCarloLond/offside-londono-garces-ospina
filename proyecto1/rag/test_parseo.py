"""Pruebas del parseo tolerante y de la herramienta. Sin LLM: corren en el CI.

    python test_parseo.py

Los casos malformados son del tipo que el 1.5B produjo de verdad durante el
desarrollo (solo "0", JSON con comillas simples, ```json, cantidad "semanas",
fecha en español...).
"""

from __future__ import annotations

import generador as gen
import herramientas as herr


def test_respuesta_final():
    r = gen.parsear_respuesta(
        '{"noticia": 1, "respuesta": "No juega [1].", "categoria": '
        '"sancion_suspension", "impacto": "negativo_alto", "equipo": "Madrid"}'
    )
    assert r["category"] == "sancion_suspension" and r["parse_ok"]
    # envuelto en ```json, comillas simples y coma colgando
    r = gen.parsear_respuesta(
        "```json\n{'respuesta': 'Baja.', 'categoria': 'baja confirmada', "
        "'impacto': 'negativo-alto', 'equipo': 'Getafe',}\n```"
    )
    assert (r["category"], r["impact"]) == ("baja_confirmada", "negativo_alto")
    # categoría mal escrita: se normaliza por similitud...
    assert (
        gen.parsear_respuesta('{"respuesta": "x", "categoria": "sancion suspencion"}')["category"]
        == "sancion_suspension"
    )
    # ...pero si es ambigua (demasiado corta), falla cerrado: sin señal
    assert gen.parsear_respuesta('{"respuesta": "x", "categoria": "sancion"}')["category"] == (
        "irrelevante"
    )
    # "noticia": 0 es la válvula aunque el modelo haya rellenado una categoría
    r = gen.parsear_respuesta(
        '{"noticia": 0, "respuesta": "Juega.", "categoria": "baja_confirmada"}'
    )
    assert r["respuesta"] == gen.VALVULA and r["category"] == "irrelevante"
    # el modelo devuelve solo "0"
    assert gen.parsear_respuesta("0")["respuesta"] == gen.VALVULA
    # texto sin JSON: se conserva y la señal cae a "sin señal"
    r = gen.parsear_respuesta("Vinicius está sancionado.")
    assert not r["parse_ok"] and r["category"] == "irrelevante"


def test_llamada_herramienta():
    nativa = (
        '<tool_call>{"name": "calcular_fecha", "arguments": {"fecha_inicio": "2026-08-27", '
        '"cantidad": 3, "unidad": "meses", "fecha_objetivo": "2026-11-01"}}</tool_call>'
    )
    assert herr.extraer_llamada(nativa)["name"] == "calcular_fecha"
    propia = (
        "{'herramienta': 'calcular_fechas', 'argumentos': '{\"fecha_inicio\": \"2026-08-08\"}'}"
    )
    llamada = herr.extraer_llamada(propia)
    assert llamada["name"] == "calcular_fecha"  # nombre aproximado -> corregido
    assert llamada["arguments"]["fecha_inicio"] == "2026-08-08"  # argumentos como string JSON
    assert herr.extraer_llamada('{"herramienta": null}') is None
    assert herr.extraer_llamada('{"respuesta": "x", "categoria": "irrelevante"}') is None


def test_ejecutar_y_errores():
    ok = herr.ejecutar(
        {
            "name": "calcular_fecha",
            "arguments": {
                "fecha_inicio": "27 de agosto de 2026",
                "cantidad": "tres",
                "unidad": "Meses",
                "fecha_objetivo": "1 de noviembre",
            },
        }
    )
    assert ok["fecha_fin_plazo"] == "2026-11-27" and ok["dias_objetivo_menos_fin"] == -26
    rango = herr.ejecutar(
        {
            "name": "calcular_fecha",
            "arguments": {
                "fecha_inicio": "2026-08-08",
                "cantidad": "2-3",
                "unidad": "meses",
                "fecha_objetivo": "2026-11-29",
            },
        }
    )
    assert rango["fecha_fin_plazo"] == "2026-11-08"  # en un rango, el máximo
    assert "error" in herr.ejecutar(
        {
            "name": "calcular_fecha",
            "arguments": {
                "fecha_inicio": "2026-08-27",
                "cantidad": "semanas",
                "unidad": "semanas",
                "fecha_objetivo": "2026-10-14",
            },
        }
    )
    assert "error" in herr.ejecutar(
        {
            "name": "calcular_fecha",
            "arguments": {
                "fecha_inicio": "2026-10-01",
                "cantidad": 4,
                "unidad": "meses",
                "fecha_objetivo": "2026-03-15",
            },
        }
    )  # objetivo anterior a la noticia
    assert "faltan" in herr.ejecutar({"name": "calcular_fecha", "arguments": {}})["error"]
    assert "no existe" in herr.ejecutar({"name": "buscar_web", "arguments": {}})["error"]
    # 31 de enero + 1 mes = 28 de febrero (no se desborda)
    assert herr.calcular_fecha("2027-01-31", 1, "meses")["fecha_fin_plazo"] == "2027-02-28"


def test_regla_cuando():
    ctx = ["(2026-08-27 · Marca · malaga) Aaron Ochoa estará de baja alrededor de tres meses."]
    assert herr.necesita_calculo("¿Podrá jugar Aaron Ochoa el 15 de diciembre?", ctx)
    assert not herr.necesita_calculo("¿Qué le pasa a Aaron Ochoa?", ctx)
    assert not herr.necesita_calculo(
        "¿Juega Uche el 3 de mayo?", ["Uche se perderá toda la temporada"]
    )


if __name__ == "__main__":
    for nombre, fn in list(globals().items()):
        if nombre.startswith("test_"):
            fn()
            print("ok ", nombre)
