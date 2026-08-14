"""Bloquea credenciales antes de que entren al repo.

Existe porque ya nos pasó una vez: un token de GitLab (`glpat-...`) quedó
embebido en una URL de `git clone` dentro del notebook y se pusheó. En un
repo público eso es una credencial regalada a cualquiera que pase por ahí.

Corre en el pipeline de lint y en pre-commit. Sin dependencias externas.

Uso:
    python scripts/check_secrets.py [rutas...]   # por defecto, todo el repo
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Patrón -> descripción del riesgo. Deliberadamente cortos y específicos:
# preferimos pocos falsos positivos a cobertura exhaustiva, para que nadie
# aprenda a ignorar este check.
PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"glpat-[A-Za-z0-9_\-.]{20,}"), "GitLab personal access token"),
    (re.compile(r"ghp_[A-Za-z0-9]{36}"), "GitHub personal access token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{50,}"), "GitHub fine-grained token"),
    (re.compile(r"hf_[A-Za-z0-9]{34,}"), "Hugging Face token"),
    (re.compile(r"sk-[A-Za-z0-9]{32,}"), "OpenAI-style API key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "clave privada"),
    # Credenciales embebidas en URLs (https://usuario:secreto@host)
    (re.compile(r"https://[^/\s:@]+:[^/\s@]{8,}@"), "credencial dentro de una URL"),
]

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", ".cache", "node_modules"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".safetensors", ".bin", ".ckpt"}

# Este archivo contiene los patrones en sí; escanearlo se auto-denuncia.
SELF = Path(__file__).resolve()


def iter_files(roots: list[Path]):
    for root in roots:
        if root.is_file():
            yield root
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.suffix.lower() in SKIP_SUFFIXES:
                continue
            yield path


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv[1:]] or [Path(".")]
    hits = []

    for path in iter_files(roots):
        if path.resolve() == SELF:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern, label in PATTERNS:
                if pattern.search(line):
                    hits.append((path, lineno, label))

    if hits:
        print("ERROR: se encontraron posibles credenciales:\n", file=sys.stderr)
        for path, lineno, label in hits:
            print(f"  {path}:{lineno}  -> {label}", file=sys.stderr)
        print(
            "\nNo commitees esto. Si ya se pusheó, el token está quemado:\n"
            "revócalo en el proveedor (borrarlo del código NO basta, queda en\n"
            "el historial de git) y usa uno nuevo fuera del repo.",
            file=sys.stderr,
        )
        return 1

    print("check_secrets: sin credenciales detectadas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
