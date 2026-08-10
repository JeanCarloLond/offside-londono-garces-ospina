"""Recolecta titulares + resumen corto desde feeds RSS de prensa deportiva.

Guardamos SOLO lo que el propio medio publica para sindicación (título +
descripción corta), nunca el cuerpo del artículo — ver docs/dataset.md para
el porqué. Cada registro queda con su fuente, fecha y un hash del link para
poder deduplicar sin tener que re-descargar nada.

Uso:
    python collect_rss.py [--sources rss_sources.yaml] [--out ../data/raw/news_raw.jsonl]

Solo usa dependencias que ya trae el repo (pyyaml, requests) o la stdlib.
No requiere GPU ni instala nada de ML: esto corre en la laptop de cualquiera
del equipo, no en Colab.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
import yaml

USER_AGENT = "OffsideSTAI-bot/0.1 (uso academico EAFIT SI4006; contacto jclondonoo@eafit.edu.co)"
TIMEOUT_S = 15


def fragment_hash(link: str) -> str:
    """Hash estable del link — permite deduplicar entre corridas sin guardar el artículo."""
    return hashlib.sha256(link.encode("utf-8")).hexdigest()[:16]


def parse_pubdate(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).astimezone(UTC).isoformat()
    except (TypeError, ValueError):
        return None


def fetch_feed(url: str) -> ET.Element:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
    resp.raise_for_status()
    return ET.fromstring(resp.content)


def parse_items(root: ET.Element, outlet: str, region: str) -> list[dict]:
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        description = (item.findtext("description") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = parse_pubdate(item.findtext("pubDate"))
        if not title or not link:
            continue
        items.append(
            {
                "id": fragment_hash(link),
                "source": outlet,
                "region": region,
                "link": link,
                "published_at": pub_date,
                "collected_at": datetime.now(UTC).isoformat(),
                "title": title,
                "summary": description,
            }
        )
    return items


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["id"])
    return ids


def main() -> int:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=here / "rss_sources.yaml")
    parser.add_argument("--out", type=Path, default=here / ".." / "data" / "raw" / "news_raw.jsonl")
    args = parser.parse_args()

    sources = yaml.safe_load(args.sources.read_text(encoding="utf-8"))["sources"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = load_existing_ids(args.out)

    new_items: list[dict] = []
    for src in sources:
        try:
            root = fetch_feed(src["url"])
        except (requests.RequestException, ET.ParseError) as exc:
            print(f"[WARN] {src['outlet']}: no se pudo leer ({exc})", file=sys.stderr)
            continue
        items = parse_items(root, src["outlet"], src["region"])
        fresh = [it for it in items if it["id"] not in seen]
        seen.update(it["id"] for it in fresh)
        new_items.extend(fresh)
        print(f"[OK] {src['outlet']}: {len(items)} items, {len(fresh)} nuevos")

    with args.out.open("a", encoding="utf-8") as f:
        for it in new_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"\nTotal nuevos: {len(new_items)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
