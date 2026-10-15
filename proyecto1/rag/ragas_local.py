"""Las cuatro métricas de RAGAS, implementadas con un juez local y determinista.

Por qué no la librería `ragas`
------------------------------
`ragas` pide al juez salidas JSON estructuradas en cada paso. Nuestro juez es un
LLM local de 1.5B (lo que cabe en Colab gratis junto a todo lo demás), y ese mismo
modelo, como generador, devolvió JSON incompletos con frecuencia (ver el parseo
tolerante de generador.py). No probamos la librería: preferimos que ningún
veredicto dependa de parsear texto. Implementamos las mismas
definiciones (Es et al., 2023, y el laboratorio de la sesión 10) y resolvemos
cada veredicto leyendo los logits de "sí"/"no" (veredicto.py), igual que el juez
de M2: un forward pass, sin parseo, sin azar.

Juez: Qwen/Qwen2.5-1.5B-Instruct (config.GENERADOR). Es el mismo modelo base que
genera las respuestas: el juez no sabe qué sistema produjo cada respuesta, pero
el riesgo de autopreferencia existe y lo declaramos. `JuezRagas(modelo=...)`
acepta otro modelo (p. ej. Qwen2.5-7B-Instruct en una GPU grande).

Las definiciones
----------------
faithfulness       afirmaciones de la respuesta que se deducen del contexto / afirmaciones.
                   Las afirmaciones las extrae el LLM; cada una se verifica sí/no.
answer_relevancy   el LLM escribe 3 preguntas que la respuesta contesta; media del
                   coseno (embeddings e5) con la pregunta original. Una respuesta
                   evasiva (la válvula) vale 0, como en ragas.
context_precision  precisión media ponderada por rango: ¿los chunks útiles para la
                   referencia están arriba? Cada chunk se juzga sí/no.
context_recall     afirmaciones de la REFERENCIA atribuibles al contexto / afirmaciones.
                   Las de la referencia están escritas a mano en el eval set
                   (`referencia_afirmaciones`), no extraídas por el LLM: un paso
                   ruidoso menos.

Qué es "contexto"
-----------------
Los chunks que vio el generador. En el sistema con herramienta se añaden los
resultados de la herramienta: la respuesta "regresa hacia el 27 de noviembre" se
apoya en ese cálculo, y RAGAS debe poder verlo (si no, castigaría como infiel una
respuesta correcta). context_precision usa solo los chunks recuperados.

Las preguntas sin respuesta en el corpus no entran en las medias: ahí no existe
contexto relevante ni referencia que recuperar. Se reportan aparte (válvula).
"""

from __future__ import annotations

import re

import config
from veredicto import Veredicto

SIS_FIEL = (
    "Eres un verificador estricto. Decides si una afirmación se deduce de un contexto. "
    "Si el contexto no la dice, o dice otra cosa, la respuesta es no."
)
SIS_UTIL = "Decides si un fragmento de noticia aporta información a una respuesta de referencia."

UMBRAL = 0.5

# Sanity check del juez: afirmaciones de verdad conocida sobre clubes FICTICIOS
# (no salen del eval set). Cubren los errores que nos importan: inversión de
# signo, categoría cambiada, número cambiado, negación y otro jugador.
CTX_A = (
    "(2026-09-02 · Marca · deportivo sauce) Lunar, baja para el derbi. El extremo del Deportivo "
    "Sauce sufre una rotura de fibras y estará tres semanas fuera"
)
CTX_B = (
    "(2026-09-03 · Marca · atletico ribera) Ferro sufre un esguince de tobillo, pero el club "
    "descarta una lesión grave"
)
SANITY = [
    (CTX_A, "Lunar no jugará el derbi.", True, "directa"),
    (CTX_A, "Lunar estará tres semanas de baja.", True, "directa"),
    (CTX_A, "Lunar jugará el derbi.", False, "signo invertido"),
    (CTX_A, "Lunar está sancionado.", False, "categoría cambiada"),
    (CTX_A, "Lunar estará dos meses de baja.", False, "número cambiado"),
    (CTX_B, "La lesión de Ferro no es grave.", True, "negación bien leída"),
    (CTX_B, "Ferro sufre una lesión grave.", False, "negación mal leída"),
    (CTX_B, "Quintana está lesionado.", False, "otro jugador"),
]


def limpiar_lineas(texto: str, maximo: int = 6) -> list[str]:
    lineas = []
    for linea in texto.splitlines():
        linea = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", linea).strip()
        if len(linea) >= 8:
            lineas.append(linea)
    return lineas[:maximo]


class JuezRagas:
    def __init__(self, model=None, tokenizer=None, device=None, modelo: str = config.GENERADOR):
        import torch
        from sentence_transformers import SentenceTransformer

        self.torch = torch
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if device == "cuda" else torch.float32
            rev = config.revision(modelo)
            tokenizer = AutoTokenizer.from_pretrained(modelo, revision=rev)
            model = (
                AutoModelForCausalLM.from_pretrained(modelo, dtype=dtype, revision=rev)
                .to(device)
                .eval()
            )
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.modelo_id = getattr(model.config, "_name_or_path", modelo)
        self.v = Veredicto(model, tokenizer, device)
        self.embedder = SentenceTransformer(
            config.EMBEDDINGS,
            device=config.DEVICE_RECUPERACION,
            revision=config.revision(config.EMBEDDINGS),
        )

    # ------------------------------------------------------------------ utilidades
    def _generar(self, usuario: str, max_new_tokens: int = 120) -> str:
        mensajes = [{"role": "user", "content": usuario}]
        texto = self.tokenizer.apply_chat_template(
            mensajes, tokenize=False, add_generation_prompt=True
        )
        entradas = self.tokenizer(texto, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **entradas,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            out[0, entradas["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def _se_deduce(self, contexto: str, afirmacion: str) -> float:
        return self.v.p_si(
            SIS_FIEL,
            f"CONTEXTO:\n{contexto}\n\nAFIRMACIÓN: {afirmacion}\n\n"
            "¿La afirmación se deduce del CONTEXTO? Contesta solo sí o no.",
        )

    # ------------------------------------------------------------------ métricas
    def afirmaciones(self, pregunta: str, respuesta: str) -> list[str]:
        salida = self._generar(
            "Divide la RESPUESTA en afirmaciones simples, cada una comprensible por sí sola "
            "(cambia los pronombres por los nombres). Una por línea, sin numerar y sin añadir "
            "nada que la respuesta no diga.\n\n"
            f"PREGUNTA: {pregunta}\nRESPUESTA: {respuesta}"
        )
        return limpiar_lineas(salida) or [respuesta]

    def faithfulness(self, pregunta: str, respuesta: str, contexto: str) -> tuple[float, list]:
        afirmaciones = self.afirmaciones(pregunta, re.sub(r"\[\d+\]", "", respuesta))
        detalle = [(a, round(self._se_deduce(contexto, a), 3)) for a in afirmaciones]
        return sum(p >= UMBRAL for _, p in detalle) / len(detalle), detalle

    def answer_relevancy(self, pregunta: str, respuesta: str, evasiva: bool) -> tuple[float, list]:
        if evasiva:
            return 0.0, []
        generadas = limpiar_lineas(
            self._generar(
                "Escribe tres preguntas distintas que la siguiente RESPUESTA contesta. Una por "
                f"línea, sin numerar.\n\nRESPUESTA: {respuesta}",
                max_new_tokens=90,
            ),
            maximo=3,
        )
        if not generadas:
            return 0.0, []
        vecs = self.embedder.encode(
            ["query: " + pregunta] + ["query: " + g for g in generadas],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sims = (vecs[1:] @ vecs[0]).tolist()
        return float(sum(sims) / len(sims)), generadas

    def context_precision(
        self, pregunta: str, referencia: str, chunks: list[str]
    ) -> tuple[float, list[int]]:
        utiles = [
            int(
                self.v.p_si(
                    SIS_UTIL,
                    f"PREGUNTA: {pregunta}\nRESPUESTA DE REFERENCIA: {referencia}\n"
                    f"FRAGMENTO: {c}\n\n¿El FRAGMENTO aporta información que se usa en la "
                    "RESPUESTA DE REFERENCIA? Contesta solo sí o no.",
                )
                >= UMBRAL
            )
            for c in chunks
        ]
        return precision_media(utiles), utiles

    def context_recall(self, afirmaciones_ref: list[str], contexto: str) -> tuple[float, list]:
        detalle = [(a, round(self._se_deduce(contexto, a), 3)) for a in afirmaciones_ref]
        return sum(p >= UMBRAL for _, p in detalle) / len(detalle), detalle

    # ------------------------------------------------------------------
    def evaluar(self, caso: dict, salida: dict) -> dict:
        """Las cuatro métricas para un caso con respuesta en el corpus."""
        chunks = salida["contextos"]
        contexto = "\n".join(f"[{i}] {c}" for i, c in enumerate(chunks, start=1))
        for h in salida.get("herramientas", []):
            contexto += f"\n[herramienta {h['name']}] {h['resultado']}"
        evasiva = salida["respuesta"].startswith("No encuentro")
        fila = {"eval_id": caso["eval_id"], "valvula": evasiva}

        if evasiva or not chunks:
            # Sin afirmaciones no hay nada que verificar: ragas devuelve NaN.
            fila["faithfulness"] = None
        else:
            fila["faithfulness"], fila["faithfulness_detalle"] = self.faithfulness(
                caso["input"]["text"], salida["respuesta"], contexto
            )
        fila["answer_relevancy"], fila["preguntas_generadas"] = self.answer_relevancy(
            caso["input"]["text"], salida["respuesta"], evasiva
        )
        if chunks:
            fila["context_precision"], fila["chunks_utiles"] = self.context_precision(
                caso["input"]["text"], caso["referencia"], chunks
            )
            fila["context_recall"], fila["recall_detalle"] = self.context_recall(
                caso["referencia_afirmaciones"], contexto
            )
        else:  # la válvula verificada calló: no llegó ningún chunk
            fila["context_precision"], fila["context_recall"] = 0.0, 0.0
        relevantes = set(caso["docs_relevantes"])
        fila["context_precision_ids"] = precision_media(
            [int(i in relevantes) for i in salida["ids"]]
        )
        return fila

    def sanity_check(self) -> dict:
        filas = []
        for ctx, afirmacion, verdad, tipo in SANITY:
            p = self._se_deduce(ctx, afirmacion)
            filas.append(
                {
                    "tipo": tipo,
                    "afirmacion": afirmacion,
                    "esperado": verdad,
                    "p_si": round(p, 3),
                    "acierta": (p >= UMBRAL) == verdad,
                }
            )
        return {
            "aciertos": sum(f["acierta"] for f in filas),
            "total": len(filas),
            "casos": filas,
        }


def precision_media(utiles: list[int]) -> float:
    """Average precision@k de RAGAS: sum_k (precision@k * v_k) / sum_k v_k."""
    if not any(utiles):
        return 0.0
    acumulado, total = 0, 0.0
    for k, v in enumerate(utiles, start=1):
        acumulado += v
        total += (acumulado / k) * v
    return total / sum(utiles)
