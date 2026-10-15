"""Las versiones del sistema, todas con la misma firma: sistema(pregunta) -> respuesta.

    sistema_sin_rag(pregunta)        mismo LLM y mismo prompt, sin noticias
    sistema_rag(pregunta)            RAG ingenuo (sesión 7): dense top-k -> prompt -> genera
    sistema_rag_avanzado(pregunta)   + las técnicas elegidas por diagnóstico
    sistema_final(pregunta)          + la herramienta de fechas (agente)

Todas se construyen con `SistemaRAG` y una configuración de flags. Así la
ablación (quitar UNA técnica y medir) no requiere código nuevo, y es imposible
que dos filas de la tabla difieran en algo que no esté en sus flags.

La respuesta es un dict:
    category, impact, team    la señal que mide el harness de M2 (contrato intacto)
    respuesta                 la respuesta en prosa (la que evalúa RAGAS)
    contextos, ids            los chunks que llegaron al generador (RAGAS los necesita)
    herramientas, pasos       qué hizo el agente
    latencia_s                tiempo de pared de la consulta completa

`para_harness(sistema)` lo adapta al contrato de harness.py, que recibe un dict
de entrada: el harness de M2 no se toca.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import config
import generador as gen
import herramientas as herr
from indice import Recuperador
from veredicto import SISTEMA_RELEVANCIA, Veredicto, usuario_relevancia


@dataclass(frozen=True)
class Config:
    nombre: str
    contexto: bool = True  # False = el LLM responde sin noticias
    hibrido: bool = False  # BM25 + dense fusionados con RRF
    rerank: bool = False  # cross-encoder sobre el pool de candidatos
    hyde: bool = False  # transformación de la consulta: documento hipotético
    multiquery: bool = False  # transformación de la consulta: reformulaciones
    verificar: bool = (
        False  # válvula verificada: filtra chunks que no responden; si no queda ninguno, calla
    )
    herramienta: bool = False  # agente con calcular_fecha
    prompt: str = gen.PROMPT_VIGENTE  # solo cambia en el registro del ajuste del prompt


SIN_RAG = Config("llm_sin_rag", contexto=False)
INGENUO = Config("rag_ingenuo")

# El avanzado, elegido con los fallos del ingenuo (docs/rag-m3.md, "Diagnóstico"):
#   rerank     el chunk llegaba pero debajo de un distractor (Musiala, Celta, Ochoa)
#   hibrido    nombres propios raros: BM25 los pone primero más a menudo que dense
#   verificar  el ingenuo inventa en las preguntas sin respuesta (válvula 0/4)
# HyDE y multi-query se probaron y se descartaron: empeoran la recuperación
# (resultados/recuperacion_m3.csv).
AVANZADO = Config("rag_avanzado", hibrido=True, rerank=True, verificar=True)
FINAL = Config("sistema_final", hibrido=True, rerank=True, verificar=True, herramienta=True)

# Lo que se corre: las filas de la tabla y la ablación (una técnica a la vez).
SISTEMAS = {
    c.nombre: c
    for c in [
        SIN_RAG,
        INGENUO,
        AVANZADO,
        FINAL,
        # cada técnica sola, sobre el ingenuo
        Config("rag_hibrido", hibrido=True),
        Config("rag_rerank", rerank=True),
        Config("rag_verificado", verificar=True),
        Config("rag_hyde", hyde=True),
        # el avanzado quitando una técnica
        Config("rag_avanzado_sin_hibrido", rerank=True, verificar=True),
        Config("rag_avanzado_sin_rerank", hibrido=True, verificar=True),
        Config("rag_avanzado_sin_verificador", hibrido=True, rerank=True),
        # registro del ajuste del prompt (prompt v1, descartado)
        Config("llm_sin_rag_prompt_v1", contexto=False, prompt="v1"),
        Config("rag_ingenuo_prompt_v1", prompt="v1"),
    ]
}
CANDIDATAS = SISTEMAS


PROMPT_HYDE = (
    "Escribe un titular y una entradilla breves (dos frases), al estilo de la prensa "
    "deportiva española, de la noticia que respondería a esta pregunta. No expliques nada, "
    "solo la noticia.\n\nPregunta: {pregunta}"
)
PROMPT_MULTIQUERY = (
    "Reescribe la pregunta de tres formas distintas, como las buscaría alguien en titulares de "
    "prensa deportiva: usa sinónimos y nombra al club si lo sabes por la pregunta. Una por "
    "línea, sin numerar.\n\nPregunta: {pregunta}"
)


class SistemaRAG:
    def __init__(self, cfg: Config, recuperador: Recuperador | None, generador: gen.Generador):
        self.cfg = cfg
        self.rec = recuperador
        self.gen = generador
        self.verificador = (
            Veredicto(generador.model, generador.tokenizer, generador.device)
            if cfg.verificar
            else None
        )

    # ------------------------------------------------------------------ retrieve
    def recuperar(self, pregunta: str, k: int = config.TOP_K) -> tuple[list[str], dict]:
        """Devuelve los ids de los k chunks y un registro de lo que se hizo."""
        cfg, rec, traza = self.cfg, self.rec, {}
        avanzado = cfg.hibrido or cfg.rerank or cfg.hyde or cfg.multiquery
        n = config.CANDIDATOS if avanzado else config.TOP_K

        listas = [rec.dense(pregunta, n)]
        if cfg.hibrido:
            listas.append(rec.lexico(pregunta, n))
        if cfg.hyde:
            hipotetico = self.gen.generar(
                [{"role": "user", "content": PROMPT_HYDE.format(pregunta=pregunta)}],
                max_new_tokens=80,
            ).strip()
            traza["hyde"] = hipotetico
            listas.append(rec.dense_por_texto(hipotetico, n))
        if cfg.multiquery:
            salida = self.gen.generar(
                [{"role": "user", "content": PROMPT_MULTIQUERY.format(pregunta=pregunta)}],
                max_new_tokens=90,
            )
            variantes = [v.strip(" -•\t") for v in salida.splitlines() if v.strip()][:3]
            traza["multiquery"] = variantes
            for v in variantes:
                listas.append(rec.dense(v, n))
                if cfg.hibrido:
                    listas.append(rec.lexico(v, n))

        candidatos = Recuperador.rrf(listas) if len(listas) > 1 else listas[0]
        if cfg.rerank:
            reordenados = rec.reordenar(pregunta, candidatos[: config.CANDIDATOS], k)
            traza["rerank_puntajes"] = [round(p, 3) for _, p in reordenados]
            return [i for i, _ in reordenados], traza
        return candidatos[:k], traza

    def verificar(self, pregunta: str, ids: list[str], traza: dict) -> list[str]:
        """Válvula verificada: el mismo LLM, como clasificador sí/no por logits,
        decide chunk a chunk si la noticia permite responder la pregunta.

        Existe porque el 1.5B, redactando, no sabe callar a tiempo (ajuste_prompt.py):
        con el prompt que responde bien, inventa en las preguntas sin respuesta. Como
        clasificador de una sola decisión acierta 13 de 14 en el dev set con el umbral
        natural de 0,5 (sin ajustar; ver calibrar_umbral.py). Si ningún
        chunk pasa, el sistema calla sin llamar al generador.
        """
        probs = [
            self.verificador.p_si(
                SISTEMA_RELEVANCIA, usuario_relevancia(self.rec.por_id[i].para_contexto, pregunta)
            )
            for i in ids
        ]
        traza["verificador"] = [round(p, 3) for p in probs]
        return [i for i, p in zip(ids, probs) if p >= config.UMBRAL_VERIFICADOR]

    # ------------------------------------------------------------------ answer
    def __call__(self, pregunta: str) -> dict:
        t0 = time.perf_counter()
        ids, traza = self.recuperar(pregunta) if self.cfg.contexto else ([], {})
        if self.cfg.verificar:
            ids = self.verificar(pregunta, ids, traza)
            if not ids:
                return {
                    **gen.SIN_SENAL,
                    "respuesta": gen.VALVULA,
                    "parse_ok": True,
                    "contextos": [],
                    "ids": [],
                    "herramientas": [],
                    "pasos": 0,
                    "texto_modelo": "",
                    "traza": {**traza, "valvula": "verificador"},
                    "latencia_s": time.perf_counter() - t0,
                }
        contextos = [self.rec.por_id[i].para_contexto for i in ids]

        llamadas, pasos = [], 0
        contextos_gen = contextos
        if self.cfg.herramienta:
            llamadas, pasos = herr.planificar(self.gen, pregunta, contextos)
            contextos_gen = contextos + herr.contexto_herramientas(llamadas)
        mensajes = gen.mensajes_base(pregunta, contextos_gen, version=self.cfg.prompt)
        texto = self.gen.generar(mensajes)
        pasos += 1

        salida = gen.parsear_respuesta(texto)
        salida.update(
            {
                "contextos": contextos,
                "ids": ids,
                "herramientas": llamadas,
                "pasos": pasos,
                "texto_modelo": texto,
                "traza": traza,
                "latencia_s": time.perf_counter() - t0,
            }
        )
        return salida


def para_harness(sistema):
    """Adapta sistema(pregunta: str) al contrato de harness.py: sistema(entrada: dict)."""

    def _sistema(entrada: dict) -> dict:
        return sistema(entrada["text"])

    return _sistema


# ---------------------------------------------------------------------------
# Las funciones con nombre que pide el enunciado. Se cargan una vez y se
# reutilizan; construir el recuperador y el generador es lo caro.
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _componentes():
    if "gen" not in _CACHE:
        _CACHE["rec"] = Recuperador()
        _CACHE["gen"] = gen.Generador()
    return _CACHE["rec"], _CACHE["gen"]


def construir(cfg: Config) -> SistemaRAG:
    rec, g = _componentes()
    return SistemaRAG(cfg, rec, g)


def sistema_sin_rag(pregunta: str) -> dict:
    return construir(SIN_RAG)(pregunta)


def sistema_rag(pregunta: str) -> dict:
    return construir(INGENUO)(pregunta)


def sistema_rag_avanzado(pregunta: str) -> dict:
    return construir(AVANZADO)(pregunta)


def sistema_final(pregunta: str) -> dict:
    return construir(FINAL)(pregunta)
