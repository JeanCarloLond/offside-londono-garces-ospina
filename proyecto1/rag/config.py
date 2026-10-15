"""Todas las decisiones del sistema RAG de Offside, en un solo lugar.

Si un número de la tabla cambia, la causa tiene que estar en este archivo o en
el eval set: nada se configura dentro de las funciones. Las justificaciones
largas (con los números que las respaldan) están en docs/rag-m3.md.
"""

from __future__ import annotations

from pathlib import Path

SEED = 42

RAIZ = Path(__file__).resolve().parent.parent  # proyecto1/
CORPUS = RAIZ / "data" / "rag" / "corpus.jsonl"
CHROMA_DIR = RAIZ / "rag" / "_chroma"  # se regenera; no se versiona
EVAL_SET = RAIZ / "eval" / "eval_set_m3.json"
RESULTADOS = RAIZ / "rag" / "resultados"

# Fecha en la que se "hacen" las preguntas: el día del corte del corpus. Fija,
# no la del reloj, para que la tabla salga igual otro día.
FECHA_CONSULTA = "2026-10-14"

# --- Chunking -----------------------------------------------------------------
# Un fragmento RSS = un chunk, sin solapamiento. Los fragmentos son titular +
# entradilla (mediana 162 caracteres, máximo 564): ya son la unidad mínima con
# sentido, y en ellos el sujeto y el hecho suelen quedar en frases distintas
# ("Chaira, baja sensible para el Oviedo. El marroquí se perderá entre dos y
# tres meses"). Partirlos separaría el nombre del jugador de su lesión.
CHUNK_MAX_CHARS = None  # sin partir
CHUNK_OVERLAP = 0

# --- Embeddings -----------------------------------------------------------------
# Comparado contra otros dos modelos sobre nuestras preguntas: ver
# resultados/embeddings_m3.csv y docs/rag-m3.md.
EMBEDDINGS = "intfloat/multilingual-e5-small"

# --- Recuperación -----------------------------------------------------------------
TOP_K = 4  # chunks que llegan al generador
CANDIDATOS = 20  # pool que ven la fusión híbrida y el reranker antes de cortar a TOP_K
RRF_K = 60  # constante de Reciprocal Rank Fusion (Cormack et al., 2009)
RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
UMBRAL_VERIFICADOR = 0.5  # P(sí) mínima para que un chunk pase la válvula verificada
DEVICE_RECUPERACION = "cpu"  # modelos pequeños: CPU en local y en Colab, misma latencia relativa

# --- Generación -------------------------------------------------------------------
# El MISMO generador y el MISMO prompt base para todas las filas de la tabla.
GENERADOR = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 200

# --- Versiones exactas de los modelos (commit del repo en Hugging Face) ----------------
# Fijadas para que "reconstruir el índice y obtener la misma tabla" no dependa de
# que alguien suba una versión nueva a main. El juez de M2 (eval/judge.py) carga
# Qwen2.5-1.5B-Instruct sin fijar revisión: no lo tocamos para no cambiar la vara;
# a fecha de esta entrega main apunta al mismo commit de abajo.
REVISIONES = {
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "intfloat/multilingual-e5-small": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": "1427fd652930e4ba29e8149678df786c240d8825",
}


def revision(modelo: str) -> str | None:
    return REVISIONES.get(modelo)


# --- Agente -------------------------------------------------------------------------
MAX_PASOS = 3  # tope de llamadas al modelo dentro del bucle del agente
