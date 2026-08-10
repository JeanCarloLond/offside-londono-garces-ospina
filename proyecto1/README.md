# Proyecto 1 — Entrega M1: Fine-tuning baseline con LoRA

> **Offside** lee noticias de fútbol y le dice a un apostador recreativo qué información ya "sabe" el mercado y cuál todavía no — este módulo entrena la primera versión del clasificador que hace esa lectura.

SI4006 · Tópicos Especiales y Aplicaciones en IA · Universidad EAFIT · Equipo Offside.

## Qué hay en esta carpeta

```text
proyecto1/
├── notebooks/
│   └── M1_finetuning_LoRA_offside.ipynb   # el entregable: corre de punta a punta en Colab
├── data_collection/                         # recolección RSS + limpieza + supervisión débil
│   ├── README.md                             # cómo correr cada script, en qué orden
│   ├── collect_rss.py / weak_label.py / build_gold_seed.py / gold_corrections.py
│   ├── rss_sources.yaml / lexicon.yaml
├── data/
│   ├── raw/            # (gitignored) crudo del scraping, nunca se commitea
│   ├── processed/       # weak_labeled.jsonl — train, supervisión débil
│   └── gold/            # gold_seed_synthetic.jsonl + gold_verified.jsonl — validación
└── docs/
    ├── dataset.md        # descripción completa del dataset (fuente, tamaño, licencia, sesgos)
    └── results.md         # tabla de resultados + lectura honesta
```

## Modelo base y tarea

**Encoder — `dccuchile/bert-base-spanish-wwm-cased` (BETO)**, afinado con **LoRA** para clasificar
fragmentos de noticias de fútbol por categoría del hecho (baja confirmada, sanción, duda física,
regreso, cambio táctico, declaración, rumor, irrelevante). Por qué esa familia y ese modelo — incluida
la comparación de fertilidad del tokenizador que descarta las alternativas — está en
[`docs/results.md`](docs/results.md).

## Resultado, en una tabla

| Baseline / modelo | F1-macro (val) |
|---|---:|
| Clase mayoritaria | 0.10 |
| BETO zero-shot (sin fine-tuning) | 0.01 |
| **LoRA fine-tuned** | **0.25** |

Tabla completa, métrica secundaria, y por qué el baseline de léxico (que da 1.00) no cuenta como
comparación honesta en este corte: [`docs/results.md`](docs/results.md).

## Cómo correrlo

1. Abrir [`notebooks/M1_finetuning_LoRA_offside.ipynb`](notebooks/M1_finetuning_LoRA_offside.ipynb) en Colab (botón dentro del notebook).
2. Entorno de ejecución → Cambiar tipo de entorno → **GPU T4**.
3. Ejecutar todas las celdas en orden. Clona el repo, no necesita subir nada a mano.

El dataset ya viene recolectado y versionado (`data/processed/`, `data/gold/`). Para ampliarlo o
recolectar de nuevo, ver [`data_collection/README.md`](data_collection/README.md).

## Estado y próximos pasos

Este es un **primer corte real**, recolectado en pretemporada (agosto 2026) — pequeño y con
limitaciones honestamente documentadas en [`docs/dataset.md`](docs/dataset.md), no el dataset final
del proyecto (meta: 4.000-6.000 fragmentos). El pipeline completo (LoRA, baselines, comparación) ya
funciona de punta a punta; lo que falta para un resultado sólido es más texto real de temporada
regular y una segunda pasada de anotación manual entre los tres integrantes — el plan está al final de
`docs/dataset.md` y `docs/results.md`.
