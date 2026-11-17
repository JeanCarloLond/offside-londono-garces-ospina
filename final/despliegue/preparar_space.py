"""Arma la carpeta del Space a partir del repo y la sube a Hugging Face.

    python final/despliegue/preparar_space.py            # solo arma la carpeta
    python final/despliegue/preparar_space.py --subir    # la arma y la sube a Jclo/offside

El código no se duplica en el repo: el Space se construye cada vez desde
proyecto1/rag (el RAG del M3), final/vision (el componente visual) y final/despliegue
(la app, la observabilidad y el health check). Hace falta haber iniciado sesión con
`hf auth login` y un token de escritura.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
DESTINO = AQUI / "_space"  # ignorado en git
SPACE = "Jclo/offside"

ARCHIVOS = {
    AQUI / "app.py": "app.py",
    AQUI / "observabilidad.py": "observabilidad.py",
    AQUI / "requirements.txt": "requirements.txt",
    AQUI / "README_space.md": "README.md",
    RAIZ / "final" / "vision" / "vision.py": "vision.py",
    RAIZ / "proyecto1" / "data" / "rag" / "corpus.jsonl": "data/rag/corpus.jsonl",
    RAIZ / "final" / "vision" / "casos" / "V-01.png": "ejemplos/V-01.png",
    RAIZ / "final" / "vision" / "casos" / "V-07.png": "ejemplos/V-07.png",
}
MODULOS_RAG = ["config", "indice", "generador", "herramientas", "veredicto", "sistemas"]


def armar() -> Path:
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    for origen, rel in ARCHIVOS.items():
        (DESTINO / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origen, DESTINO / rel)
    for m in MODULOS_RAG:
        (DESTINO / "rag").mkdir(parents=True, exist_ok=True)
        shutil.copy2(RAIZ / "proyecto1" / "rag" / f"{m}.py", DESTINO / "rag" / f"{m}.py")
    return DESTINO


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subir", action="store_true")
    args = parser.parse_args()
    destino = armar()
    print("carpeta del Space:", destino)
    if args.subir:
        from huggingface_hub import HfApi

        api = HfApi()
        # En una cuenta gratuita, Gradio solo puede correr sobre ZeroGPU.
        api.create_repo(
            SPACE,
            repo_type="space",
            space_sdk="gradio",
            space_hardware="zero-a10g",
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(destino),
            repo_id=SPACE,
            repo_type="space",
            commit_message="Despliegue desde el repo de Offside",
        )
        print(f"subido: https://huggingface.co/spaces/{SPACE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
