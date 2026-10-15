"""El generador y el prompt base, compartidos por TODAS las versiones del sistema.

Delta limpio: el modelo, los parámetros de decodificación y este prompt son los
mismos para el LLM sin recuperación, el RAG ingenuo, el avanzado y el agente. Lo
único que cambia entre filas de la tabla es qué noticias llegan al bloque
NOTICIAS (y, en el agente, que se le ofrece una herramienta).

El modelo devuelve un JSON con la respuesta en prosa y la señal estructurada que
mide el harness de M2. El parseo es tolerante: un modelo de 1.5B a veces envuelve
el JSON en ```json, usa comillas simples o deja una coma colgando, y eso no debe
convertirse en una respuesta perdida.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
import unicodedata

import config

CATEGORIAS = [
    "baja_confirmada",
    "sancion_suspension",
    "duda_fisica",
    "regreso_alta",
    "cambio_tactico",
    "declaracion_contexto",
    "rumor_no_confirmado",
    "irrelevante",
]
IMPACTOS = ["negativo_alto", "negativo_bajo", "neutro", "positivo_bajo", "positivo_alto"]
VALVULA = "No encuentro esa información en las noticias recolectadas."
SIN_SENAL = {"category": "irrelevante", "impact": "neutro", "team": "ninguno"}

CATEGORIAS_TXT = """categoria (elige una):
- baja_confirmada: el jugador no estará (lesión o ausencia confirmada)
- sancion_suspension: expulsión o sanción que le impide jugar
- duda_fisica: molestias, pruebas pendientes o vuelta sin confirmar
- regreso_alta: vuelve a estar disponible
- cambio_tactico: alineación, once probable o cambio de sistema
- declaracion_contexto: declaraciones u opiniones sin un hecho de disponibilidad
- rumor_no_confirmado: fichajes o hechos sin confirmar
- irrelevante: no hay ninguna señal para un partido
impacto, para el equipo afectado: negativo_alto, negativo_bajo, neutro, positivo_bajo o positivo_alto."""

# v1: la primera versión. La conservamos porque su fallo es un hallazgo: con el
# documento correcto en la posición 1, el modelo copiaba literalmente la frase de
# la válvula (ver docs/rag-m3.md, sección 5, y la fila rag_ingenuo_prompt_v1).
PROMPT_V1 = (
    """Eres Offside, un asistente que responde preguntas de apostadores recreativos sobre noticias de fútbol. Hoy es {fecha}.

Reglas:
1. Responde SOLO con la información de las NOTICIAS que recibes. No uses lo que sepas por tu cuenta.
2. Si las noticias no contienen la respuesta, responde exactamente "{valvula}" con categoria "irrelevante", impacto "neutro" y equipo "ninguno".
3. Si dos noticias se contradicen, manda la más reciente.
4. Cita entre corchetes el número de la noticia que usas, por ejemplo [2].

Devuelve SOLO un objeto JSON con estas claves:
{{"respuesta": "...", "categoria": "...", "impacto": "...", "equipo": "..."}}

"""
    + CATEGORIAS_TXT
)

# v2: ajustada SOLO sobre el dev set (eval/dev_set_m3.json), nunca sobre el eval
# set. Cambios: el procedimiento empieza por buscar la noticia que responde, la
# válvula es la excepción explícita ("solo si ninguna noticia...") y hay dos
# ejemplos resueltos con clubes ficticios, uno que responde y uno que calla.
PROMPT_V2 = (
    """Eres Offside, un asistente que responde preguntas de apostadores recreativos sobre noticias de fútbol. Hoy es {fecha}.

Recibes NOTICIAS numeradas, cada una con su fecha, y una PREGUNTA. Procedimiento:
1. Busca la noticia que habla del jugador o del club por el que se pregunta y responde con lo que ella dice, citando su número entre corchetes, por ejemplo [1].
2. Usa SOLO las NOTICIAS, nunca lo que sepas por tu cuenta. Si dos noticias se contradicen, manda la más reciente.
3. Solo si NINGUNA noticia trata lo que se pregunta, la respuesta es "{valvula}", con categoria "irrelevante", impacto "neutro" y equipo "ninguno".

Devuelve SOLO un objeto JSON con las claves "respuesta", "categoria", "impacto" y "equipo".

"""
    + CATEGORIAS_TXT
)

_NOTICIAS_DEMO = [
    "(2026-09-02 · Marca · deportivo sauce) Lunar, baja para el derbi. El extremo del "
    "Deportivo Sauce sufre una rotura de fibras en el isquiotibial",
    "(2026-09-01 · Marca · atletico ribera) El Atlético Ribera presenta su nueva equipación "
    "para la temporada",
]
FEW_SHOT_V2 = [
    (
        "¿Juega Lunar el derbi?",
        {
            "respuesta": "No. Lunar es baja para el derbi por una rotura de fibras [1].",
            "categoria": "baja_confirmada",
            "impacto": "negativo_alto",
            "equipo": "Deportivo Sauce",
        },
    ),
    (
        "¿Está sancionado el portero del Atlético Ribera?",
        {
            "respuesta": VALVULA,
            "categoria": "irrelevante",
            "impacto": "neutro",
            "equipo": "ninguno",
        },
    ),
]
# v3: sin ejemplos (con ellos el 1.5B imitaba la demo en vez de leer las noticias)
# y con un primer campo "noticia" que obliga a ELEGIR la noticia que responde
# antes de redactar. noticia = 0 es la válvula, y se aplica de forma determinista
# en parsear_respuesta: la frase de la válvula deja de ser un atractor.
PROMPT_V3 = (
    """Eres Offside, un asistente que responde preguntas de apostadores recreativos sobre noticias de fútbol. Hoy es {fecha}.

Recibes NOTICIAS numeradas, cada una con su fecha, y una PREGUNTA. Devuelve SOLO un objeto JSON con estas claves, en este orden:
- "noticia": el número de la noticia que responde la pregunta, o 0 si ninguna habla de lo que se pregunta.
- "respuesta": la respuesta en una o dos frases, usando SOLO lo que dicen las noticias y citando su número, por ejemplo [1]. Si "noticia" es 0, escribe exactamente "{valvula}".
- "categoria", "impacto" y "equipo": la señal para el apostador.

Las preguntas como "¿juega X?" o "¿cuenta el equipo con X?" se responden con la noticia sobre la baja, la sanción, la duda o el regreso de X.
Usa solo las NOTICIAS, nunca lo que sepas por tu cuenta. Si dos noticias se contradicen, manda la más reciente. Si "noticia" es 0: categoria "irrelevante", impacto "neutro" y equipo "ninguno".

"""
    + CATEGORIAS_TXT
)

# v4: v3 contestaba todo, también lo que no está (le atribuía a Pedri la lesión de
# Guridi). Se añade la comprobación explícita antes de elegir la noticia.
PROMPT_V4 = PROMPT_V3.replace(
    '- "noticia": el número de la noticia que responde la pregunta, o 0 si ninguna habla de lo que se pregunta.',
    '- "noticia": el número de la noticia que responde la pregunta. Antes de elegirla, comprueba '
    "que esa noticia NOMBRA al jugador o al club de la pregunta Y trata de lo que se pregunta. "
    "Si ninguna cumple las dos cosas, pon 0.",
)

PROMPTS = {
    "v1": (PROMPT_V1, []),
    "v2": (PROMPT_V2, FEW_SHOT_V2),
    "v3": (PROMPT_V3, []),
    "v4": (PROMPT_V4, []),
}
# v3 es la elegida con el dev set (9/14 aciertos, 0/11 válvulas falsas). Una
# primera corrida completa salió por error con v4 (se quedó como vigente tras
# probarla); se detectó en la auditoría previa a la entrega y se repitió todo.
PROMPT_VIGENTE = "v3"


def prompt_usuario(pregunta: str, contextos: list[str]) -> str:
    if contextos:
        noticias = "\n".join(f"[{i}] {c}" for i, c in enumerate(contextos, start=1))
    else:
        noticias = "(ninguna)"
    return f"NOTICIAS:\n{noticias}\n\nPREGUNTA: {pregunta}"


def mensajes_base(
    pregunta: str, contextos: list[str], version: str = PROMPT_VIGENTE, extra_sistema: str = ""
) -> list[dict]:
    """Los mensajes que recibe el generador. Idénticos para todas las versiones del
    sistema salvo por `contextos` (y `extra_sistema` en el agente)."""
    plantilla, demos = PROMPTS[version]
    sistema = plantilla.format(fecha=config.FECHA_CONSULTA, valvula=VALVULA) + extra_sistema
    mensajes = [{"role": "system", "content": sistema}]
    for pregunta_demo, respuesta_demo in demos:
        mensajes.append({"role": "user", "content": prompt_usuario(pregunta_demo, _NOTICIAS_DEMO)})
        mensajes.append(
            {"role": "assistant", "content": json.dumps(respuesta_demo, ensure_ascii=False)}
        )
    mensajes.append({"role": "user", "content": prompt_usuario(pregunta, contextos)})
    return mensajes


# ---------------------------------------------------------------------------
# Parseo tolerante
# ---------------------------------------------------------------------------
def _sin_tildes(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def reparar_json(bruto: str) -> dict | None:
    """Intenta, en orden, cada vez más permisivo, convertir `bruto` en un dict."""
    candidatos = [bruto]
    t = bruto.replace("“", '"').replace("”", '"').replace("’", "'")
    t = re.sub(r",\s*([}\]])", r"\1", t)  # comas colgando
    candidatos.append(t)
    candidatos.append(re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", t)))
    for c in candidatos:
        try:
            obj = json.loads(c)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass
    try:  # comillas simples, None/True de Python
        obj = ast.literal_eval(t)
        return obj if isinstance(obj, dict) else None
    except (ValueError, SyntaxError):
        return None


def extraer_objeto(texto: str) -> dict | None:
    """El primer objeto {...} balanceado del texto que se pueda reparar."""
    texto = re.sub(r"```(?:json)?", "", texto)
    for inicio in [m.start() for m in re.finditer(r"\{", texto)]:
        prof = 0
        for j in range(inicio, len(texto)):
            if texto[j] == "{":
                prof += 1
            elif texto[j] == "}":
                prof -= 1
                if prof == 0:
                    obj = reparar_json(texto[inicio : j + 1])
                    if obj is not None:
                        return obj
                    break
    return None


def normalizar_etiqueta(valor: str | None, opciones: list[str], defecto: str) -> str:
    if not valor:
        return defecto
    v = _sin_tildes(str(valor)).strip().lower().replace(" ", "_").replace("-", "_")
    if v in opciones:
        return v
    cerca = difflib.get_close_matches(v, opciones, n=1, cutoff=0.6)
    return cerca[0] if cerca else defecto


def es_valvula(respuesta: str) -> bool:
    r = _sin_tildes(respuesta).lower()
    return "no encuentro esa informacion" in r or "no encuentro informacion" in r


def parsear_respuesta(texto: str) -> dict:
    """Texto del modelo -> {respuesta, category, impact, team, parse_ok}.

    Si no hay JSON recuperable se conserva el texto como respuesta y la señal
    cae a "sin señal": ante la duda, callar (lo que premia la rúbrica).
    Si la respuesta en prosa es la válvula, la señal también lo es, aunque el
    modelo haya rellenado otra categoría: no puede decir "no sé" y a la vez
    emitir una baja.
    """
    obj = extraer_objeto(texto)
    if obj is None:
        # A veces el modelo devuelve solo "0": el valor de "noticia" sin el JSON.
        bruto = VALVULA if texto.strip() in ("0", '"0"') else texto.strip()
        salida = {"respuesta": bruto, **SIN_SENAL, "parse_ok": False}
    else:
        claves = {_sin_tildes(str(k)).lower(): v for k, v in obj.items()}
        salida = {
            "respuesta": str(claves.get("respuesta") or "").strip(),
            "category": normalizar_etiqueta(claves.get("categoria"), CATEGORIAS, "irrelevante"),
            "impact": normalizar_etiqueta(claves.get("impacto"), IMPACTOS, "neutro"),
            "team": str(claves.get("equipo") or "ninguno").strip() or "ninguno",
            "parse_ok": True,
        }
        if str(claves.get("noticia", "")).strip() in ("0", "0.0"):
            salida["respuesta"] = VALVULA  # el modelo eligió "ninguna noticia"
    if es_valvula(salida["respuesta"]):
        salida.update(SIN_SENAL)
        salida["respuesta"] = VALVULA
    return salida


# ---------------------------------------------------------------------------
class Generador:
    """Qwen2.5-1.5B-Instruct con decodificación greedy (determinista)."""

    def __init__(self, modelo: str = config.GENERADOR, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.modelo_id = modelo
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        rev = config.revision(modelo)
        self.tokenizer = AutoTokenizer.from_pretrained(modelo, revision=rev)
        self.model = AutoModelForCausalLM.from_pretrained(modelo, dtype=dtype, revision=rev).to(
            self.device
        )
        self.model.eval()

    def generar(
        self,
        mensajes: list[dict],
        tools: list[dict] | None = None,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
    ) -> str:
        texto = self.tokenizer.apply_chat_template(
            mensajes, tools=tools, tokenize=False, add_generation_prompt=True
        )
        entradas = self.tokenizer(texto, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            salida = self.model.generate(
                **entradas,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        nuevos = salida[0, entradas["input_ids"].shape[1] :]
        return self.tokenizer.decode(nuevos, skip_special_tokens=True)

    def liberar(self) -> None:
        del self.model
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
