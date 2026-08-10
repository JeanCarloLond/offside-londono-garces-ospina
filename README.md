# Offside

> **Offside lee noticias de fútbol y le dice a un apostador recreativo qué información ya "sabe" el mercado y cuál todavía no** — para que decida con criterio y no solo con el titular que alcanzó a leer en el celular.

Proyecto integrador del curso SI4006 · Tópicos Especiales y Aplicaciones en IA · Universidad EAFIT.

## Equipo

| Integrante | Correo |
|---|---|
| Jean Carlo Londoño Ocampo (contacto) | jclondonoo@eafit.edu.co |
| Alejandro Garcés Ramírez | agarcesr@eafit.edu.co |
| Nicolás Ospina Torres | nospinat@eafit.edu.co |

## Estado del proyecto

- ✅ M1 · Arquitectura transformer y fine-tuning eficiente — ver [`proyecto1/`](proyecto1/)

## Estructura del repo

```text
.
├── proyecto1/              # Entregable M1 (y base de los siguientes módulos)
│   ├── notebooks/           # Notebook de fine-tuning, ejecutable en Colab
│   ├── data_collection/      # Scraping/recolección + limpieza + weak labeling
│   ├── data/                 # processed/ y gold/ se versionan; raw/ e interim/ no (ver .gitignore)
│   └── docs/                 # Descripción del dataset y reporte de resultados
├── .gitlab-ci.yml           # Pipeline de lint (rápido, sin GPU)
├── .pre-commit-config.yaml  # Mismos checks del CI, en local antes de commitear
├── pyproject.toml           # Config de ruff (Python + notebooks)
└── requirements-dev.txt     # Herramientas de desarrollo (lint, hooks) — no hace falta en Colab
```

## Buenas prácticas del repo

- **Lint automático**: cada push/MR corre `ruff` (Python + notebooks) y `pymarkdown` (Markdown) en un pipeline de GitLab CI de un solo job, sin GPU (`.gitlab-ci.yml`). Debe pasar antes de mergear a `main`.
- **Hooks locales opcionales, mismos checks que el CI**:

  ```bash
  pip install -r requirements-dev.txt
  pre-commit install
  ```

- **Nada de datos crudos ni artefactos de modelo en git**: `.gitignore` excluye `proyecto1/data/raw/`, checkpoints y pesos — ver por qué en [`proyecto1/docs/dataset.md`](proyecto1/docs/dataset.md#licencia-y-uso).
- **Ramas + Merge Requests**: trabajo en ramas cortas (`feature/...`, `fix/...`), MR a `main` con al menos una revisión del equipo antes de mergear.

## Cómo correr el entregable de M1

Ver [`proyecto1/README.md`](proyecto1/README.md) — incluye el link directo a Colab, el reporte de resultados y la descripción del dataset.
