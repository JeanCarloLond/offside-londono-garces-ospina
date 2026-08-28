# Proyecto 1 — Offside (M1: fine-tuning con LoRA · M2: evaluación)

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
├── eval/                                    # M2: harness de evaluación
│   ├── eval_set.json                         # 21 ejemplos: input / esperado / criterio
│   ├── judge_rubric.yaml                     # rúbrica 1-5 del juez, versionada
│   ├── harness.py                            # 3 dimensiones -> scorecard
│   ├── judge.py / bias_check.py              # juez LLM y su sesgo medido
│   └── scorecard_baseline.csv                # el scorecard del baseline
├── data/
│   ├── raw/            # (gitignored) crudo del scraping, nunca se commitea
│   ├── processed/       # weak_labeled.jsonl — train, supervisión débil
│   └── gold/            # gold_seed_synthetic.jsonl + gold_verified.jsonl — validación
└── docs/
    ├── dataset.md        # descripción completa del dataset (fuente, tamaño, licencia, sesgos)
    ├── results.md         # tabla de resultados + lectura honesta
    ├── eval-set.md        # semilla del eval set (S05)
    └── harness-m2.md      # M2: harness, rúbrica del juez, sesgo y scorecard
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

1. Descargar [`notebooks/M1_finetuning_LoRA_offside.ipynb`](notebooks/M1_finetuning_LoRA_offside.ipynb) y subirlo a [Colab](https://colab.research.google.com) (pestaña **Subir**).
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

## M2 · Harness de evaluación

Ningún benchmark público mide nuestro dominio, y los que existen se contaminan. Construimos el
nuestro: un **harness de tres dimensiones** sobre un eval set curado a mano, que produce un scorecard
del baseline y se puede volver a correr sobre cualquier sistema futuro — el RAG de M3 entra sin tocar
el harness.

| Dimensión | Qué es | Qué no ve |
|---|---|---|
| 1 · Métrica clásica | F1 macro sobre la categoría | La gravedad del error |
| 2 · LLM-as-a-judge | Qwen2.5-1.5B con rúbrica 1-5 anclada y versionada | Es pequeño y falible; su punto ciego está medido |
| 3 · Dominio | Tasa de señal accionable, con verificaciones deterministas | Matices: es una puerta de sí/no |

```bash
cd proyecto1/eval
python harness.py --sin-juez    # dims 1 y 3, instantáneo y sin GPU
python harness.py               # las tres
```

Detalle completo —eval set, rúbrica del juez, sesgo mitigado y lectura honesta del
scorecard— en [`docs/harness-m2.md`](docs/harness-m2.md).
