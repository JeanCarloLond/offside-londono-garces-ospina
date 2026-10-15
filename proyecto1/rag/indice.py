"""Pasos 2-4 del pipeline de indexación (chunk -> embed -> store) y la recuperación.

    python indice.py            # construye el índice Chroma desde data/rag/corpus.jsonl
    python indice.py --forzar   # lo reconstruye aunque exista

El índice no se versiona (rag/_chroma/ está en .gitignore): se reconstruye en
menos de un minuto en CPU, y así quien clona el repo comprueba que el pipeline
corre de principio a fin.

Qué se indexa y qué ve el generador
-----------------------------------
- Se indexa (BM25 y embeddings) la sección del enlace + el texto: en muchos
  fragmentos el club solo está en la URL (ver ingest.py).
- El generador recibe además la fecha y el medio en una cabecera. Sin la fecha,
  "estará de baja tres meses" no se puede convertir en un día de regreso.

Los cuatro recuperadores (ver docs/rag-m3.md para el porqué de cada uno):
    dense      embeddings + Chroma (similitud coseno)
    bm25       léxico, con normalización de tildes ("Fenerbahçe" == "fenerbahce")
    hibrido    dense + bm25 fusionados con Reciprocal Rank Fusion
    rerank     un cross-encoder reordena el pool de candidatos
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import config

MEDIOS = {
    "marca": "Marca",
    "marca_laliga": "Marca",
    "marca_segunda": "Marca",
    "marca_champions": "Marca",
    "marca_premier": "Marca",
    "marca_bundesliga": "Marca",
    "eltiempo_deportes": "El Tiempo",
    "eltiempo_futbol_colombiano": "El Tiempo",
}

# Stopwords mínimas: las que más ruido meten en BM25 sobre titulares cortos.
STOPWORDS = set(
    """a al algo como con contra cual cuales de del el ella en es esta este esto
    ha han hay la las le les lo los mas me mi muy no nos o para pero por que se
    si sin sobre su sus te tu un una uno unos y ya yo""".split()
)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto)
    sin_tildes = "".join(c for c in sin_tildes if not unicodedata.combining(c))
    return sin_tildes.lower()


def tokenizar(texto: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(normalizar(texto)) if t not in STOPWORDS]


@dataclass
class Chunk:
    id: str
    fecha: str
    medio: str
    seccion: str
    link: str
    texto: str

    @property
    def indexado(self) -> str:
        """Lo que ven BM25 y el modelo de embeddings."""
        return f"{self.seccion}. {self.texto}" if self.seccion else self.texto

    @property
    def para_contexto(self) -> str:
        """Lo que ve el generador: con fecha y medio delante."""
        cab = f"{self.fecha} · {self.medio}" + (f" · {self.seccion}" if self.seccion else "")
        return f"({cab}) {self.texto}"


def cargar_chunks(ruta: Path = config.CORPUS) -> list[Chunk]:
    """Ingest -> chunk. Un documento = un chunk (ver config.CHUNK_MAX_CHARS)."""
    chunks = []
    with ruta.open(encoding="utf-8") as f:
        for linea in f:
            d = json.loads(linea)
            chunks.append(
                Chunk(
                    id=d["id"],
                    fecha=d["fecha"],
                    medio=MEDIOS.get(d["fuente"], d["fuente"]),
                    seccion=d.get("seccion", ""),
                    link=d["link"],
                    texto=d["texto"],
                )
            )
    return chunks


def _prefijos(modelo: str) -> tuple[str, str]:
    """Los modelos E5 se entrenaron con prefijos; sin ellos rinden peor."""
    if "e5" in modelo.lower():
        return "query: ", "passage: "
    return "", ""


def _nombre_coleccion(modelo: str) -> str:
    return "offside_" + re.sub(r"[^a-zA-Z0-9]+", "_", modelo.split("/")[-1]).strip("_")[:50]


class Recuperador:
    """Carga el índice (o lo construye) y expone las cuatro formas de recuperar."""

    def __init__(
        self,
        modelo_embeddings: str = config.EMBEDDINGS,
        chroma_dir: Path = config.CHROMA_DIR,
        forzar: bool = False,
        con_reranker: bool = True,
    ):
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.chunks = cargar_chunks()
        self.por_id = {c.id: c for c in self.chunks}
        self.modelo_embeddings = modelo_embeddings
        self.pref_q, self.pref_p = _prefijos(modelo_embeddings)
        self.embedder = SentenceTransformer(
            modelo_embeddings,
            device=config.DEVICE_RECUPERACION,
            revision=config.revision(modelo_embeddings),
        )

        cliente = chromadb.PersistentClient(path=str(chroma_dir))
        nombre = _nombre_coleccion(modelo_embeddings)
        existentes = {c.name for c in cliente.list_collections()}
        if forzar and nombre in existentes:
            cliente.delete_collection(nombre)
            existentes.discard(nombre)
        self.coleccion = cliente.get_or_create_collection(nombre, metadata={"hnsw:space": "cosine"})
        if self.coleccion.count() != len(self.chunks):
            if self.coleccion.count():
                cliente.delete_collection(nombre)
                self.coleccion = cliente.get_or_create_collection(
                    nombre, metadata={"hnsw:space": "cosine"}
                )
            self._indexar()

        self.bm25 = BM25Okapi([tokenizar(c.indexado) for c in self.chunks])
        self.reranker = None
        if con_reranker:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(
                config.RERANKER,
                device=config.DEVICE_RECUPERACION,
                revision=config.revision(config.RERANKER),
            )

    # ------------------------------------------------------------------ store
    def _indexar(self) -> None:
        """Embed -> store. Los vectores se calculan aquí, no con la función de Chroma,
        para que el modelo de embeddings sea exactamente el de config.py."""
        textos = [self.pref_p + c.indexado for c in self.chunks]
        vectores = self.embedder.encode(
            textos, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        lote = 500
        for i in range(0, len(self.chunks), lote):
            parte = self.chunks[i : i + lote]
            self.coleccion.add(
                ids=[c.id for c in parte],
                documents=[c.indexado for c in parte],
                embeddings=vectores[i : i + lote].tolist(),
                metadatas=[
                    {"fecha": c.fecha, "medio": c.medio, "seccion": c.seccion, "link": c.link}
                    for c in parte
                ],
            )

    # ------------------------------------------------------------------ recuperar
    def dense(self, consulta: str, n: int) -> list[str]:
        vec = self.embedder.encode(
            [self.pref_q + consulta], normalize_embeddings=True, show_progress_bar=False
        )
        res = self.coleccion.query(query_embeddings=vec.tolist(), n_results=n)
        return res["ids"][0]

    def dense_por_texto(self, texto_pasaje: str, n: int) -> list[str]:
        """Busca con un texto que se parece a un PASAJE, no a una pregunta (HyDE)."""
        vec = self.embedder.encode(
            [self.pref_p + texto_pasaje], normalize_embeddings=True, show_progress_bar=False
        )
        res = self.coleccion.query(query_embeddings=vec.tolist(), n_results=n)
        return res["ids"][0]

    def lexico(self, consulta: str, n: int) -> list[str]:
        puntajes = self.bm25.get_scores(tokenizar(consulta))
        orden = sorted(range(len(self.chunks)), key=lambda i: (-puntajes[i], i))
        return [self.chunks[i].id for i in orden[:n] if puntajes[i] > 0]

    @staticmethod
    def rrf(listas: list[list[str]], k: int = config.RRF_K) -> list[str]:
        """Reciprocal Rank Fusion: suma 1/(k + rango) de cada lista.

        Usa solo el RANGO, no el puntaje: BM25 y coseno viven en escalas que no
        se pueden sumar directamente.
        """
        puntaje: dict[str, float] = {}
        for lista in listas:
            for rango, doc_id in enumerate(lista, start=1):
                puntaje[doc_id] = puntaje.get(doc_id, 0.0) + 1.0 / (k + rango)
        return sorted(puntaje, key=lambda d: (-puntaje[d], d))

    def reordenar(self, consulta: str, ids: list[str], k: int) -> list[tuple[str, float]]:
        pares = [(consulta, self.por_id[i].indexado) for i in ids]
        puntajes = self.reranker.predict(pares, show_progress_bar=False)
        orden = sorted(zip(ids, puntajes), key=lambda x: (-float(x[1]), x[0]))
        return [(i, float(p)) for i, p in orden[:k]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forzar", action="store_true")
    parser.add_argument("--modelo", default=config.EMBEDDINGS)
    args = parser.parse_args()
    import time

    t0 = time.perf_counter()
    rec = Recuperador(args.modelo, forzar=args.forzar, con_reranker=False)
    print(
        f"índice '{rec.coleccion.name}': {rec.coleccion.count()} chunks de "
        f"{len(rec.chunks)} documentos ({time.perf_counter() - t0:.1f} s)"
    )
    print("prueba:", rec.dense("¿Vinicius puede jugar el Clásico?", 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
