"""Limpieza mínima + supervisión débil (baseline de nivel 2: léxico/regex).

Toma proyecto1/data/raw/news_raw.jsonl (crudo, no versionado) y produce
proyecto1/data/processed/weak_labeled.jsonl (SÍ versionado: son fragmentos
cortos + etiqueta, no el artículo).

También imprime la distribución de clases resultante — es el primer chequeo
de sesgo del dataset (ver docs/dataset.md, "clase mayoritaria: irrelevante").

Uso:
    python weak_label.py [--raw ../data/raw/news_raw.jsonl]
                          [--lexicon lexicon.yaml]
                          [--out ../data/processed/weak_labeled.jsonl]
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path

import yaml

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    """Quita HTML y normaliza espacios — la limpieza mínima que pide M1."""
    no_tags = TAG_RE.sub(" ", raw)
    unescaped = html.unescape(no_tags)
    return WS_RE.sub(" ", unescaped).strip()


def compile_lexicon(cfg: dict) -> list[tuple[str, str, list[re.Pattern]]]:
    compiled = []
    for cat in cfg["categories"]:
        patterns = [re.compile(p, re.IGNORECASE) for p in cat["patterns"]]
        compiled.append((cat["name"], cat["impact_default"], patterns))
    return compiled


def compile_sentiment(cfg: dict) -> dict[str, list[re.Pattern]]:
    boost = cfg.get("sentiment_boost", {})
    return {label: [re.compile(p, re.IGNORECASE) for p in pats] for label, pats in boost.items()}


def label_one(text: str, categories, sentiment) -> tuple[str, str]:
    for name, impact_default, patterns in categories:
        if any(p.search(text) for p in patterns):
            impact = impact_default
            for boosted_label, pats in sentiment.items():
                if any(p.search(text) for p in pats):
                    impact = boosted_label
                    break
            return name, impact
    return "irrelevante", "neutro"


def main() -> int:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=here / ".." / "data" / "raw" / "news_raw.jsonl")
    parser.add_argument("--lexicon", type=Path, default=here / "lexicon.yaml")
    parser.add_argument(
        "--out", type=Path, default=here / ".." / "data" / "processed" / "weak_labeled.jsonl"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.lexicon.read_text(encoding="utf-8"))
    categories = compile_lexicon(cfg)
    sentiment = compile_sentiment(cfg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cat_counts: Counter[str] = Counter()
    impact_counts: Counter[str] = Counter()
    n_in = n_out = 0

    with args.raw.open(encoding="utf-8") as fin, args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            item = json.loads(line)
            title = clean_text(item["title"])
            summary = clean_text(item.get("summary", ""))
            text = f"{title}. {summary}".strip()
            if len(text) < 15:  # limpieza mínima: descarta fragmentos vacíos/rotos
                continue
            category, impact = label_one(text, categories, sentiment)
            record = {
                "id": item["id"],
                "source": item["source"],
                "region": item.get("region"),
                "link": item["link"],
                "published_at": item.get("published_at"),
                "text": text,
                "category": category,
                "impact": impact,
                "label_method": "weak_lexicon",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            cat_counts[category] += 1
            impact_counts[impact] += 1
            n_out += 1

    print(f"Fragmentos leídos: {n_in} | etiquetados y escritos: {n_out} -> {args.out}")
    print("\nDistribución de categoría (supervisión débil):")
    for cat, n in cat_counts.most_common():
        print(f"  {cat:<22} {n:>5}  ({100 * n / max(n_out, 1):.1f}%)")
    print("\nDistribución de impacto:")
    for imp, n in impact_counts.most_common():
        print(f"  {imp:<22} {n:>5}  ({100 * n / max(n_out, 1):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
