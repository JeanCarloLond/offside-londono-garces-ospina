"""Paso 1 del pipeline de indexación: ingest.

    news_raw.jsonl (crudo, no versionado)  ->  data/rag/corpus.jsonl (versionado)

El corpus del RAG es el mismo que recolecta `data_collection/collect_rss.py`
(titular + resumen de sindicación de Marca y El Tiempo), limpio y deduplicado.
NO es `data/processed/weak_labeled.jsonl`: ese archivo es el train de M1/M2 y
no lo tocamos, para que los números de esos módulos sigan siendo reproducibles.

Cada documento queda con su fecha y su fuente. No es decoración: nuestras
preguntas son sobre disponibilidad de jugadores, y "estará de baja tres meses"
no significa nada sin saber CUÁNDO se publicó. La herramienta de fechas del
agente (herramientas.py) vive de ese campo.

Uso:
    python ingest.py                     # reconstruye data/rag/corpus.jsonl desde data/raw/
    python ingest.py --raw otro.jsonl

Quien clona el repo no necesita correr esto: corpus.jsonl ya está versionado
(son fragmentos cortos de sindicación con su enlace, no artículos; ver
docs/dataset.md#idioma-y-licencia). Se deja para poder ampliar el corpus.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "data_collection"))
from weak_label import clean_text  # noqa: E402  la misma limpieza que M1

# Restos de la plantilla RSS de Marca que sobreviven a quitar el HTML: el
# enlace "Leer" al final del resumen. No aportan nada y ensucian BM25.
COLA_RE = re.compile(r"\s*Leer\s*$")
MIN_CHARS = 40

# Marca organiza sus URLs por club: /futbol/<seccion>/AAAA/MM/DD/... En muchos
# fragmentos el club SOLO aparece ahí ("Quique abre el casting para la banda
# derecha" no dice que es el Alavés; la URL sí). Lo guardamos como metadato.
SECCION_RE = re.compile(r"marca\.com/futbol/([a-z0-9-]+)/\d{4}/")
SECCIONES_GENERICAS = {"primera-division", "segunda-division", "champions-league", "album"}


def seccion(link: str) -> str:
    m = SECCION_RE.search(link)
    if not m or m.group(1) in SECCIONES_GENERICAS:
        return ""
    return m.group(1).replace("-", " ")


def limpiar(registro: dict) -> str:
    titulo = clean_text(registro.get("title") or "")
    resumen = COLA_RE.sub("", clean_text(registro.get("summary") or ""))
    titulo = titulo.rstrip(".")
    return f"{titulo}. {resumen}".strip() if resumen else titulo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=HERE.parent / "data" / "raw" / "news_raw.jsonl")
    parser.add_argument("--out", type=Path, default=HERE.parent / "data" / "rag" / "corpus.jsonl")
    args = parser.parse_args()

    vistos_id, vistos_texto, docs = set(), set(), []
    descartes = Counter()
    with args.raw.open(encoding="utf-8") as f:
        for linea in f:
            r = json.loads(linea)
            texto = limpiar(r)
            clave = texto.lower()
            if r["id"] in vistos_id:
                descartes["id_repetido"] += 1
                continue
            if clave in vistos_texto:
                # El mismo titular publicado en dos feeds de Marca (portada y
                # sección). Duplicarlo solo le roba un puesto del top-k a otra
                # noticia.
                descartes["texto_repetido"] += 1
                continue
            if len(texto) < MIN_CHARS:
                descartes["demasiado_corto"] += 1
                continue
            vistos_id.add(r["id"])
            vistos_texto.add(clave)
            docs.append(
                {
                    "id": r["id"],
                    "fecha": (r.get("published_at") or "")[:10],
                    "fuente": r["source"],
                    "region": r.get("region", ""),
                    "seccion": seccion(r["link"]),
                    "link": r["link"],
                    "texto": texto,
                }
            )

    docs.sort(key=lambda d: (d["fecha"], d["id"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    largos = sorted(len(d["texto"]) for d in docs)
    print(f"documentos: {len(docs)}  ({dict(descartes)} descartados)")
    print(f"rango de fechas: {docs[0]['fecha']} .. {docs[-1]['fecha']}")
    print(
        f"caracteres por documento: mediana {largos[len(largos) // 2]}, "
        f"p90 {largos[int(0.9 * len(largos))]}, max {largos[-1]}"
    )
    print(f"fuentes: {dict(Counter(d['fuente'] for d in docs))}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
