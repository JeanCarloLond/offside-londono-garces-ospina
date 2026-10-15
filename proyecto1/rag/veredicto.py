"""Veredictos sí/no leídos de los logits de un LLM instruct.

Es el mismo truco que el juez de M2 (eval/judge.py): en vez de generar texto y
buscarle un "sí" con una regex, se comparan los logits de los tokens "sí" y "no"
en la primera posición de la respuesta. Un solo forward pass, determinista, sin
fallos de parseo, y con una probabilidad que dice cuánto dudó el modelo.

Lo usan dos piezas:
- la válvula verificada del RAG avanzado (sistemas.py): ¿esta noticia permite
  responder la pregunta?
- las métricas tipo RAGAS (ragas_local.py): ¿esta afirmación se deduce del
  contexto?, ¿este chunk sirve para llegar a la referencia?
"""

from __future__ import annotations


class Veredicto:
    def __init__(self, model, tokenizer, device: str):
        import torch

        self.torch = torch
        self.model, self.tokenizer, self.device = model, tokenizer, device

        def primeros(palabras):
            return sorted({tokenizer.encode(p, add_special_tokens=False)[0] for p in palabras})

        self.ids_si = primeros(["sí", "Sí", "si", "Si", " sí", " Sí", "SI", "SÍ"])
        self.ids_no = primeros(["no", "No", " no", " No", "NO"])

    def p_si(self, sistema: str, usuario: str) -> float:
        """P(sí) renormalizada entre el bloque "sí" y el bloque "no"."""
        mensajes = [{"role": "system", "content": sistema}, {"role": "user", "content": usuario}]
        texto = self.tokenizer.apply_chat_template(
            mensajes, tokenize=False, add_generation_prompt=True
        )
        entradas = self.tokenizer(texto, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.model(**entradas).logits[0, -1].float()
        a = self.torch.logsumexp(logits[self.ids_si], 0)
        b = self.torch.logsumexp(logits[self.ids_no], 0)
        return float(self.torch.sigmoid(a - b))


# v2 del verificador, ajustada sobre el dev set: la v1 ("¿la noticia permite
# responder?") rechazaba las preguntas de fechas, porque la noticia no menciona la
# fecha por la que se pregunta. La v2 pregunta si es LA FUENTE, aunque haya que
# deducir la respuesta.
SISTEMA_RELEVANCIA = (
    "Eres un verificador. Decides si una noticia de fútbol es la fuente para responder una "
    "pregunta. Lo es si habla del MISMO jugador o club por el que se pregunta y de su situación "
    "(lesión, baja, sanción, regreso, alineación, fichaje, declaraciones), aunque la respuesta "
    "haya que deducirla, por ejemplo calculando fechas. No lo es si habla de otro jugador u "
    "otro asunto."
)


def usuario_relevancia(noticia: str, pregunta: str) -> str:
    return (
        f"NOTICIA: {noticia}\n\nPREGUNTA: {pregunta}\n\n"
        "¿Es esta noticia la fuente para responder la pregunta? Contesta solo sí o no."
    )
