# Recolección y preprocesamiento de datos

Pipeline de recolección propia que alimenta el notebook de M1. Corre en cualquier laptop del equipo (no necesita GPU ni Colab). Ver [`../docs/dataset.md`](../docs/dataset.md) para la descripción completa del dataset resultante — este README es sobre **cómo correr los scripts**, no sobre el dataset en sí.

## Orden de ejecución

```bash
cd proyecto1/data_collection
pip install -r ../../requirements-data.txt   # requests, pyyaml (además de lo que ya trae el repo)

python collect_rss.py        # 1. Descarga los feeds de rss_sources.yaml -> ../data/raw/news_raw.jsonl
python weak_label.py         # 2. Limpieza mínima + léxico/regex        -> ../data/processed/weak_labeled.jsonl
python build_gold_seed.py    # 3. (una sola vez) seed sintético         -> ../data/gold/gold_seed_synthetic.jsonl
```

El notebook (`../notebooks/`) hace el resto: split train/val, verificación del gold set y fine-tuning. No hace falta correr nada más a mano antes de abrir el notebook.

## Qué hace cada script

- **`collect_rss.py`** — lee `rss_sources.yaml`, descarga cada feed, guarda título + resumen corto + link + fecha. Deduplica por hash del link entre corridas (append-only: correrlo de nuevo solo agrega lo nuevo). **Nunca** descarga el artículo completo — ver la nota de licencia en `docs/dataset.md`.
- **`rss_sources.yaml`** — la lista de feeds, con comentarios de qué se probó y se descartó (y por qué) el 2026-08-09. Antes de agregar una fuente nueva, verifíquenla con `curl` — varios feeds "oficiales" que probamos responden 200 pero están vacíos o abandonados.
- **`lexicon.yaml`** — el léxico/regex que define el baseline de nivel 2 del proyecto y etiqueta el conjunto de entrenamiento (supervisión débil). Está pensado para editarse: si el notebook o la verificación manual encuentran un patrón que se les escapa, agréguenlo aquí primero.
- **`weak_label.py`** — aplica `lexicon.yaml` sobre `data/raw/`, limpia HTML/espacios, y reporta la distribución de clases resultante (primer chequeo de sesgo).
- **`build_gold_seed.py`** — genera un seed **sintético** (equipos/jugadores ficticios) para las categorías que la ventana de recolección actual casi no tiene. Léanlo antes de usarlo: explica por qué existe y qué límites tiene.
- **`gold_corrections.py`** — registro de la primera pasada de verificación manual sobre el conjunto de validación real (no un script que se "corre", es documentación ejecutable de las correcciones encontradas).

## Para sumar más datos

Correr `collect_rss.py` de nuevo en cualquier momento — es seguro, solo agrega fragmentos nuevos. La meta del equipo es 4.000-6.000 fragmentos (`Context.md`); este corte inicial trae 251. Recomendado: correrlo una vez al día durante las próximas semanas (a mano o con un pipeline programado de GitLab CI) para cubrir jornadas reales de liga, no solo pretemporada.
