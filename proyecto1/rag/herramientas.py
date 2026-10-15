"""La capa agéntica: una herramienta de fechas y el bucle que la usa.

Por qué ESTA herramienta
------------------------
Las bajas en nuestro corpus llegan como duraciones relativas a la fecha de la
noticia: "estará de baja alrededor de tres meses" (27 de agosto), "se perderá
entre dos y tres meses" (8 de agosto). La pregunta del apostador, en cambio, es
por una fecha concreta: "¿llega al partido del 1 de noviembre?". Pasar de una a
otra es aritmética de calendario, y un modelo de 1.5B la hace mal: suma meses de
memoria, se equivoca de mes o compara fechas al revés. Una función de diez
líneas no se equivoca nunca. Es exactamente el caso de "dale una calculadora".

Cuándo NO se usa
----------------
En todo lo demás. Una pregunta como "¿volverá Uche esta temporada?" suena a
fechas, pero la noticia ya lo responde ("toda la temporada"): no hay nada que
calcular. El esquema lo dice explícitamente y el harness cuenta las llamadas
innecesarias (sobre-uso) como hallazgo, no las esconde.

Robustez
--------
- Esquema JSON con nombre, descripción y parámetros (formato de function calling),
  incluido en el prompt.
- Parseo tolerante de lo que propone el modelo: <tool_call>{...}</tool_call>,
  JSON desnudo, comillas simples, comas colgando, argumentos como string JSON,
  fechas en español ("27 de agosto de 2026"), cantidades en palabras ("tres") o
  rangos ("2-3": se usa el máximo, que es la lectura prudente).
- Si los argumentos no son válidos, el error vuelve al modelo como resultado de
  la herramienta para que corrija; eso consume un paso.
- Tope de pasos (config.MAX_PASOS): al alcanzarlo se le pide la respuesta final
  sin herramientas. El agente no puede entrar en bucle.
"""

from __future__ import annotations

import calendar
import difflib
import json
import re
from datetime import date

import config
from generador import _sin_tildes, extraer_objeto, reparar_json

ESQUEMA_CALCULAR_FECHA = {
    "type": "function",
    "function": {
        "name": "calcular_fecha",
        "description": (
            "Suma un plazo (dias, semanas o meses) a una fecha y lo compara con una fecha "
            "objetivo. Usala SOLO si la pregunta pide saber si un jugador estara disponible en "
            "una fecha concreta Y la noticia da la duracion de su baja. Si la noticia ya "
            "responde la pregunta sin calcular nada, NO la uses."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fecha_inicio": {
                    "type": "string",
                    "description": "Fecha de publicacion de la noticia de la baja, AAAA-MM-DD.",
                },
                "cantidad": {
                    "type": "number",
                    "description": "Duracion de la baja. Si es un rango (2-3), el valor maximo.",
                },
                "unidad": {"type": "string", "enum": ["dias", "semanas", "meses"]},
                "fecha_objetivo": {
                    "type": "string",
                    "description": "Fecha por la que pregunta el usuario, AAAA-MM-DD.",
                },
            },
            "required": ["fecha_inicio", "cantidad", "unidad", "fecha_objetivo"],
        },
    },
}
HERRAMIENTAS = {"calcular_fecha": ESQUEMA_CALCULAR_FECHA}

# Cómo se reparte el trabajo, y por qué (prueba_planificador.py lo mide):
#  1. Con la plantilla nativa de tools de Qwen (apply_chat_template(tools=...)) el
#     sistema recibe un bloque en inglés y el 1.5B respondía "0" (ninguna noticia)
#     incluso a preguntas que sin herramienta contestaba bien.
#  2. Con la herramienta descrita dentro del prompt base, nunca la llamaba.
#  3. Con un paso de planificación aparte en el que el LLM decide (modo "llm"),
#     tampoco: 0 de 4 llamadas cuando hacía falta. Leyendo su decisión por logits,
#     daba 0,93 a "¿Qué lesión tiene Lunar?", que no necesita ningún cálculo.
# Conclusión: este modelo NO sabe decidir CUÁNDO usar la herramienta. Así que el
# cuándo lo decide una regla verificable (necesita_calculo) y el QUÉ lo propone el
# LLM: fecha de la noticia, cantidad, unidad y fecha objetivo, en un JSON que se
# parsea con tolerancia y, si es inválido, vuelve al modelo con el error. El paso
# de respuesta usa el MISMO prompt base del RAG avanzado, con el resultado de la
# herramienta añadido al bloque de noticias (delta limpio).
PROMPT_PLANIFICADOR = (
    "Eres el planificador de Offside. Hoy es {fecha}. Recibes NOTICIAS (cada una con su fecha) "
    "y una PREGUNTA. Tienes UNA herramienta:\n"
    + json.dumps(ESQUEMA_CALCULAR_FECHA["function"], ensure_ascii=False)
    + "\n\nDecide si hace falta llamarla. Hace falta SOLO si se cumplen las dos cosas: la PREGUNTA "
    "nombra una fecha concreta (por ejemplo 'el 30 de octubre') y una noticia da una duración "
    "con número para la baja ('tres meses', 'entre dos y tres meses', 'seis semanas'). En ese "
    "caso devuelve SOLO:\n"
    '{"herramienta": "calcular_fecha", "argumentos": {"fecha_inicio": "AAAA-MM-DD", '
    '"cantidad": N, "unidad": "dias|semanas|meses", "fecha_objetivo": "AAAA-MM-DD"}}\n'
    "donde fecha_inicio es la fecha de la noticia de la baja y fecha_objetivo la de la pregunta.\n"
    'En cualquier otro caso devuelve SOLO: {"herramienta": null}'
)

PROMPT_ARGUMENTOS = (
    "Preparas una llamada a una herramienta. Hoy es {fecha}. Recibes NOTICIAS (cada una con su "
    "fecha entre paréntesis) y una PREGUNTA sobre si un jugador estará disponible en una fecha. "
    "La herramienta es:\n"
    + json.dumps(ESQUEMA_CALCULAR_FECHA["function"], ensure_ascii=False)
    + "\n\nDevuelve SOLO este JSON, rellenado:\n"
    '{"herramienta": "calcular_fecha", "argumentos": {"fecha_inicio": "AAAA-MM-DD", '
    '"cantidad": N, "unidad": "dias|semanas|meses", "fecha_objetivo": "AAAA-MM-DD"}}\n'
    "fecha_inicio: la fecha de la noticia que da la duración de la baja del jugador.\n"
    "cantidad y unidad: esa duración; si es un rango ('entre dos y tres meses'), el máximo.\n"
    "fecha_objetivo: la fecha por la que pregunta la PREGUNTA."
)

FECHA_EN_PREGUNTA_RE = re.compile(
    r"\b\d{1,2} de (enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre"
    r"|octubre|noviembre|diciembre)\b|\b\d{4}-\d{2}-\d{2}\b",
    re.I,
)
DURACION_RE = re.compile(
    r"\b(un|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|\d+)\s+"
    r"(d[ií]as?|semanas?|mes|meses)\b",
    re.I,
)


def necesita_calculo(pregunta: str, contextos: list[str]) -> bool:
    """La regla del CUÁNDO: la pregunta nombra una fecha concreta y algún chunk trae
    una duración con número. Es deliberadamente estrecha: preferimos no llamar la
    herramienta (y que el generador responda como el avanzado) a llamarla de más."""
    return bool(FECHA_EN_PREGUNTA_RE.search(pregunta)) and any(
        DURACION_RE.search(c) for c in contextos
    )


def prompt_planificador() -> str:
    return PROMPT_PLANIFICADOR.replace("{fecha}", config.FECHA_CONSULTA)


def prompt_argumentos() -> str:
    return PROMPT_ARGUMENTOS.replace("{fecha}", config.FECHA_CONSULTA)


MESES = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}
NUMEROS = {
    "un": 1,
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
}
UNIDADES = {"dia": "dias", "dias": "dias", "semana": "semanas", "semanas": "semanas"}
UNIDADES |= {"mes": "meses", "meses": "meses", "day": "dias", "days": "dias"}
UNIDADES |= {"week": "semanas", "weeks": "semanas", "month": "meses", "months": "meses"}


class ErrorArgumento(ValueError):
    pass


# ---------------------------------------------------------------------------
# Normalización de argumentos
# ---------------------------------------------------------------------------
def parsear_fecha(valor) -> date:
    s = _sin_tildes(str(valor)).strip().lower()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:t.*)?", s)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", s)
    if m:
        return date(int(m[3]), int(m[2]), int(m[1]))
    m = re.fullmatch(r"(\d{1,2}) de ([a-z]+)(?: de(?:l)? (\d{4}))?", s)
    if m and m[2] in MESES:
        anio = int(m[3]) if m[3] else int(config.FECHA_CONSULTA[:4])
        return date(anio, MESES[m[2]], int(m[1]))
    raise ErrorArgumento(f"no entiendo la fecha '{valor}'; usa AAAA-MM-DD")


def parsear_cantidad(valor) -> float:
    if isinstance(valor, int | float):
        return float(valor)
    s = _sin_tildes(str(valor)).lower()
    numeros = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    numeros += [float(NUMEROS[p]) for p in re.findall(r"[a-z]+", s) if p in NUMEROS]
    if not numeros:
        raise ErrorArgumento(f"no entiendo la cantidad '{valor}'; usa un número")
    return max(numeros)  # en un rango, el extremo prudente


def sumar_plazo(inicio: date, cantidad: float, unidad: str) -> date:
    if unidad == "dias":
        return date.fromordinal(inicio.toordinal() + round(cantidad))
    if unidad == "semanas":
        return date.fromordinal(inicio.toordinal() + round(7 * cantidad))
    meses_enteros = int(cantidad)
    total = inicio.month - 1 + meses_enteros
    anio, mes = inicio.year + total // 12, total % 12 + 1
    dia = min(inicio.day, calendar.monthrange(anio, mes)[1])
    resultado = date(anio, mes, dia)
    if cantidad != meses_enteros:  # medio mes = 15 días
        resultado = date.fromordinal(resultado.toordinal() + round(30 * (cantidad - meses_enteros)))
    return resultado


def calcular_fecha(fecha_inicio, cantidad, unidad, fecha_objetivo=None) -> dict:
    """La herramienta. Pura y determinista."""
    inicio = parsear_fecha(fecha_inicio)
    n = parsear_cantidad(cantidad)
    u = UNIDADES.get(_sin_tildes(str(unidad)).strip().lower())
    if u is None:
        raise ErrorArgumento(f"unidad '{unidad}' no válida; usa dias, semanas o meses")
    fin = sumar_plazo(inicio, n, u)
    salida = {"fecha_inicio": inicio.isoformat(), "fecha_fin_plazo": fin.isoformat()}
    if fecha_objetivo:
        objetivo = parsear_fecha(fecha_objetivo)
        if objetivo < inicio:
            # Un error típico del modelo: "15 de marzo" con el año de la noticia, no el
            # siguiente. Se le devuelve para que lo corrija en el siguiente paso.
            raise ErrorArgumento(
                f"fecha_objetivo {objetivo.isoformat()} es anterior a la noticia "
                f"({inicio.isoformat()}); revisa el año"
            )
        dif = (objetivo - fin).days
        salida["fecha_objetivo"] = objetivo.isoformat()
        salida["dias_objetivo_menos_fin"] = dif
        if dif >= 0:
            salida["lectura"] = (
                f"El {objetivo.isoformat()} es {dif} dias DESPUES del fin del plazo "
                f"({fin.isoformat()}): para esa fecha el plazo de baja YA se habra cumplido, "
                "aunque no hay alta confirmada."
            )
        else:
            salida["lectura"] = (
                f"El {objetivo.isoformat()} es {-dif} dias ANTES del fin del plazo "
                f"({fin.isoformat()}): para esa fecha el jugador TODAVIA estara de baja."
            )
    return salida


# ---------------------------------------------------------------------------
# Parseo tolerante de la llamada que propone el modelo
# ---------------------------------------------------------------------------
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)


def extraer_llamada(texto: str) -> dict | None:
    """Devuelve {"name", "arguments"} si el texto propone una llamada, o None.

    Acepta el formato nativo de Qwen (<tool_call>{...}</tool_call>) y, como
    respaldo, un JSON desnudo con "name" + "arguments" (o "herramienta" +
    "argumentos"): los modelos pequeños olvidan las etiquetas a menudo.
    """
    bloques = [m.group(1) for m in TOOL_CALL_RE.finditer(texto)]
    candidatos = [extraer_objeto(b) or reparar_json(b.strip()) for b in bloques]
    if not any(candidatos):
        candidatos = [extraer_objeto(texto)]
    for obj in candidatos:
        if not obj:
            continue
        obj = {_sin_tildes(str(k)).lower(): v for k, v in obj.items()}
        if "function" in obj and isinstance(obj["function"], dict):
            obj = {**obj, **obj["function"]}
        nombre = obj.get("name") or obj.get("herramienta") or obj.get("tool")
        if not nombre:
            continue  # es la respuesta final, no una llamada
        args = obj.get("arguments") or obj.get("argumentos") or obj.get("parameters") or {}
        if isinstance(args, str):
            args = reparar_json(args) or {}
        nombre_ok = difflib.get_close_matches(str(nombre), list(HERRAMIENTAS), n=1, cutoff=0.6)
        return {"name": nombre_ok[0] if nombre_ok else str(nombre), "arguments": args}
    return None


def ejecutar(llamada: dict) -> dict:
    """Ejecuta una llamada ya parseada. Nunca lanza: los errores vuelven al modelo."""
    if llamada["name"] not in HERRAMIENTAS:
        return {"error": f"no existe la herramienta '{llamada['name']}'"}
    args = {_sin_tildes(str(k)).lower(): v for k, v in dict(llamada["arguments"]).items()}
    faltan = [
        p
        for p in ESQUEMA_CALCULAR_FECHA["function"]["parameters"]["required"]
        if p not in args or args[p] in (None, "")
    ]
    if faltan:
        return {"error": f"faltan argumentos: {', '.join(faltan)}"}
    try:
        return calcular_fecha(
            args["fecha_inicio"], args["cantidad"], args["unidad"], args.get("fecha_objetivo")
        )
    except (ErrorArgumento, ValueError, OverflowError) as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
def planificar(
    generador, pregunta: str, contextos: list[str], modo: str = "regla"
) -> tuple[list[dict], int]:
    """Bucle de la herramienta con tope de pasos.

    modo="regla": necesita_calculo() decide si se llama; el LLM propone los
        argumentos. Es el modo del sistema final.
    modo="llm": el LLM decide también si llama (prompt_planificador). Se conserva
        para medir por qué no se usa (prueba_planificador.py).

    En cada paso el modelo propone un JSON; si los argumentos no son válidos, el
    error vuelve al modelo y lo intenta de nuevo, hasta MAX_PASOS. Se corta en la
    primera llamada válida o si el modelo no propone ninguna.

    Devuelve (llamadas hechas, pasos usados).
    """
    if modo == "regla" and not necesita_calculo(pregunta, contextos):
        return [], 0
    sistema = prompt_argumentos() if modo == "regla" else prompt_planificador()
    mensajes = [
        {"role": "system", "content": sistema},
        {"role": "user", "content": _prompt_usuario(pregunta, contextos)},
    ]
    llamadas = []
    for paso in range(1, config.MAX_PASOS + 1):
        salida = generador.generar(mensajes, max_new_tokens=120)
        llamada = extraer_llamada(salida)
        if llamada is None:
            return llamadas, paso
        resultado = ejecutar(llamada)
        llamadas.append({**llamada, "resultado": resultado, "texto_modelo": salida})
        if "error" not in resultado:
            return llamadas, paso
        mensajes = [
            *mensajes,
            {"role": "assistant", "content": salida},
            {
                "role": "user",
                "content": "ERROR de calcular_fecha: "
                + resultado["error"]
                + ". Corrige los argumentos y devuelve SOLO el JSON de la llamada.",
            },
        ]
    return llamadas, config.MAX_PASOS


def _prompt_usuario(pregunta: str, contextos: list[str]) -> str:
    noticias = "\n".join(f"[{i}] {c}" for i, c in enumerate(contextos, start=1)) or "(ninguna)"
    return f"NOTICIAS:\n{noticias}\n\nPREGUNTA: {pregunta}"


def contexto_herramientas(llamadas: list[dict], fechado: bool = True) -> list[str]:
    """Los resultados válidos, como entradas extra del bloque NOTICIAS.

    Van con la fecha de la consulta (fechado=True). La primera versión los añadía
    sin fecha y el generador los ignoraba: en el eval set, en Chaira, la
    herramienta calculó bien y la respuesta repitió "entre dos y tres meses". Con
    fecha, la regla del prompt base "si dos noticias se contradicen, manda la más
    reciente" juega a su favor. Medido en prueba_planificador.py (casos ficticios):
    el generador usa el resultado en 0 de 4 casos sin fecha y en 2 de 4 con fecha.
    """
    cabecera = (
        f"({config.FECHA_CONSULTA} · cálculo de Offside)"
        if fechado
        else "(resultado de la herramienta calcular_fecha)"
    )
    return [
        f"{cabecera} {h['resultado']['lectura']}" for h in llamadas if "lectura" in h["resultado"]
    ]
