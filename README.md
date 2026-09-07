# Offside

[![lint](https://github.com/JeanCarloLond/offside-londono-garces-ospina/actions/workflows/lint.yml/badge.svg)](https://github.com/JeanCarloLond/offside-londono-garces-ospina/actions/workflows/lint.yml)
[![Licencia MIT](https://img.shields.io/badge/licencia-MIT-blue.svg)](LICENSE)
[![Módulo](https://img.shields.io/badge/SI4006-M1%20·%20M2-success)](#entregables)

> **Offside lee noticias de fútbol y le dice a un apostador recreativo qué información ya "sabe" el mercado y cuál todavía no** — para que decida con criterio y no solo con el titular que alcanzó a leer en el celular.
>
> **Y para saber si eso funciona, lo medimos.** Nuestro harness de evaluación comprueba si la señal que entregamos —qué pasó, a quién le afecta y cuánto— le sirve al usuario para decidir mejor que si hubiera leído el titular por su cuenta. Una respuesta es buena cuando **acierta el tipo de hecho, no se equivoca en la dirección del impacto, no confunde un rumor con un hecho, y ante la duda calla en vez de inventar**.

Proyecto integrador del curso SI4006 · Tópicos Especiales y Aplicaciones en IA · Universidad EAFIT.

---

## Entregables

| Módulo | Qué es | Empiece por aquí | Artefactos |
|---|---|---|---|
| **M2** · Evaluación rigurosa | Harness de 3 dimensiones + scorecard del baseline, con la rúbrica del juez y su sesgo medido | **[Ir a M2 ↓](#m2--harness-de-evaluación)** · [`docs/harness-m2.md`](proyecto1/docs/harness-m2.md) | [eval set](proyecto1/eval/eval_set.json) · [rúbrica del juez](proyecto1/eval/judge_rubric.yaml) · [scorecard](proyecto1/eval/scorecard_baseline.csv) · [notebook](proyecto1/notebooks/M2_harness_evaluacion.ipynb) |
| **M1** · Fine-tuning con LoRA | BETO afinado sobre prensa deportiva, comparado contra tres baselines | **[Ir a M1 ↓](#m1--fine-tuning-con-lora)** · [`docs/results.md`](proyecto1/docs/results.md) | [notebook](proyecto1/notebooks/M1_finetuning_LoRA_offside.ipynb) · [dataset](proyecto1/docs/dataset.md) · [recolección](proyecto1/data_collection/) |

Los dos notebooks abren en Colab con un clic y clonan el repo solos:

[![M1 en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JeanCarloLond/offside-londono-garces-ospina/blob/main/proyecto1/notebooks/M1_finetuning_LoRA_offside.ipynb) M1 · fine-tuning
&nbsp;&nbsp;
[![M2 en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JeanCarloLond/offside-londono-garces-ospina/blob/main/proyecto1/notebooks/M2_harness_evaluacion.ipynb) M2 · harness

### Equipo

| Integrante | Correo |
|---|---|
| Jean Carlo Londoño Ocampo (contacto) | jclondonoo@eafit.edu.co |
| Alejandro Garcés Ramírez | agarcesr@eafit.edu.co |
| Nicolás Ospina Torres | nospinat@eafit.edu.co |

---

## M2 · Harness de evaluación

Ningún benchmark público mide nuestro dominio, y los que existen se contaminan. Construimos el
nuestro: un harness de **tres dimensiones** sobre 21 ejemplos curados a mano (13 adversariales), que
produce un scorecard del baseline y se puede volver a correr sobre cualquier sistema futuro — el RAG
de M3 entra sin tocar el harness.

| Dimensión | Qué mide | Qué **no** ve |
|---|---|---|
| **1 · Métrica clásica** | F1 macro sobre la categoría. Automática y reproducible | La **gravedad**: confundir una baja con una sanción cuesta lo mismo que invertir el signo |
| **2 · LLM-as-a-judge** | Si la señal le sirve al usuario, pesando el daño de cada error | Es un 1.5B falible. Su punto ciego está medido |
| **3 · Dominio** | Verificaciones duras y deterministas de los errores que no nos podemos permitir | Matices: es una puerta de sí/no |

```bash
cd proyecto1/eval
python harness.py --sin-juez    # dimensiones 1 y 3, instantáneo y sin GPU
python harness.py               # las tres
```

### La rúbrica del juez (versión 1)

El juez es `Qwen/Qwen2.5-1.5B-Instruct` y puntúa de 1 a 5 con estas anclas. Está versionada en
[`proyecto1/eval/judge_rubric.yaml`](proyecto1/eval/judge_rubric.yaml): cambiar un ancla cambia el
veredicto, así que el scorecard solo es comparable entre corridas con la misma versión.

| Nivel | Nombre | Qué respuesta lo merece |
|---:|---|---|
| **5** | Señal correcta y accionable | Acierta hecho, dirección del impacto, intensidad y equipo. El usuario puede actuar tal cual |
| **4** | Correcta con un matiz menor | Hecho y **dirección** correctos; falla la intensidad o el equipo |
| **3** | No engaña, pero tampoco sirve | No emite señal habiendo una, **o** acierta el signo con una categoría vecina. Deja al usuario como estaba |
| **2** | Señal engañosa | Inventa señal donde no la hay, o infla el impacto a alto. Añade daño |
| **1** | Señal dañina | **Invierte el signo**, o presenta un rumor como hecho confirmado |

El orden no es «cuánto acierta» sino **cuánto daño hace al decidir**: por eso *no emitir señal* (3)
puntúa por encima de *emitir una falsa* (2).

### El scorecard del baseline

| Sistema | D1 · F1 macro | D2 · juez (media) | D3 · tasa accionable |
|---|---:|---:|---:|
| Clase mayoritaria | 0,0400 | 2,52 | 0,19 |
| **Léxico/regex** | **0,4238** | 2,57 | **0,52** |
| LoRA afinado (M1) | 0,3208 | 2,52 | 0,33 |

| Sistema | Señal falsa de alto impacto | Signo invertido | Adversariales | Normales |
|---|---:|---:|---:|---:|
| Clase mayoritaria | 0 | 0 | 0,23 | 0,12 |
| Léxico/regex | **2** | **1** | 0,62 | 0,38 |
| LoRA afinado | 1 | 0 | 0,38 | 0,25 |

Datos completos en [`scorecard_baseline.csv`](proyecto1/eval/scorecard_baseline.csv) y, por ejemplo,
en [`scorecard_baseline.json`](proyecto1/eval/scorecard_baseline.json).

### La lectura, sin maquillar

**El léxico gana en la métrica clásica** (0,42 contra 0,32) y lo decimos. Pero consigue esa ventaja
**gritando**: emite dos señales falsas de alto impacto y una inversión de signo, los dos errores que
más le cuestan al usuario. El modelo afinado acierta menos pero **falla más barato**. Con una sola
métrica esa diferencia es invisible — ese es el argumento de por qué hay tres dimensiones.

**La dimensión 3 es la más severa** (0,52 en el mejor sistema) porque es la única con verificaciones
binarias: una inversión de signo tumba el ejemplo entero, sin nota parcial.

**En los adversariales el léxico va mejor (0,62) que en los normales (0,38)**, que suena al revés. La
explicación es incómoda: varias de las trampas son las que estudiamos en M1 y que nos llevaron a
parchear el léxico, mientras el bucket normal está lleno de bajas de temporada que nunca miramos. El
baseline está **sobreajustado a las trampas conocidas y ciego a los casos corrientes**.

**Y el hallazgo que más nos costó aceptar: nuestro juez discrimina poco.** Las medias de los tres
sistemas son 2,52 / 2,57 / 2,52. Medido con rigor: da 2,83 cuando la categoría es correcta y 2,38
cuando no (delta +0,45), y **en 9 de 21 ejemplos le pone la misma nota a una respuesta correcta y a
una incorrecta**. Un modelo de 1.5B en un solo forward pass no ejecuta la comparación que pide la
rúbrica. En M3 lo dejaremos razonar antes de puntuar.

> **Hoy ningún sistema sirve para el usuario, y ahora sabemos exactamente por qué.** Lo que sí
> tenemos es la vara fija: el mismo scorecard medirá el RAG de M3 sin cambiar una línea del harness.

### El sesgo del juez, reconocido y mitigado

**Sesgo: verbosidad** — los jueces LLM premian las respuestas largas aunque no aporten información.
Nos afecta de lleno de cara a M3, donde el RAG redactará señales más largas que el clasificador de
M1: si el juez premia la longitud, el RAG «ganaría» sin ser mejor.

**Detección** ([`bias_check.py`](proyecto1/eval/bias_check.py)): las mismas predicciones, renderizadas
con tres longitudes y exactamente la misma información.

| Condición | Longitud media | Nota media |
|---|---:|---:|
| `escueta` | 24 chars | 2,62 |
| `fija` (la que usamos) | 55 chars | 2,57 |
| `verbosa` | 352 chars | **3,05** |

**+0,43 puntos de rúbrica** solo por escribir más largo, y 7 de 21 ejemplos cambian de nivel.

**Mitigación:** `render_senal()` emite **todas** las señales con la misma plantilla de longitud fija,
sea cual sea el sistema. La longitud deja de ser una variable y el juez no tiene de dónde sacar la
preferencia. Es estructural, no un ajuste sobre el puntaje — no eliminamos el sesgo del modelo (eso no
se puede desde fuera), le quitamos la señal de la que se alimenta.

Detalle completo en [`proyecto1/docs/harness-m2.md`](proyecto1/docs/harness-m2.md).

---

## M1 · Fine-tuning con LoRA

**Encoder — `dccuchile/bert-base-spanish-wwm-cased` (BETO)**, afinado con LoRA para clasificar
fragmentos de prensa por categoría del hecho. Antes de fijar el modelo medimos la **fertilidad del
tokenizador** sobre 40 términos del dominio: BETO parte nuestro vocabulario en **2,24** tokens por
término, contra 2,88 del DistilBERT multilingüe y 3,48 del inglés.

| Sistema | F1 macro (validación de M1) |
|---|---:|
| Clase mayoritaria | 0,1033 |
| BETO zero-shot (sin afinar) | 0,0104 |
| **LoRA afinado** | **0,2468** |

LoRA mueve el **0,27 %** de los parámetros (301 064 de 110 M), con `r=8`, `alpha=16` y
`target_modules=[query, value]`.

Tras el M2 ampliamos el corpus con la liga ya en marcha —de **251 a 497** fragmentos, incorporando
prensa colombiana— y el F1 en validación del modelo reentrenado subió a **0,7046**. El detalle y la
lectura honesta están en [`docs/results.md`](proyecto1/docs/results.md), y el dataset con sus sesgos
en [`docs/dataset.md`](proyecto1/docs/dataset.md).

---

## El repositorio

```text
.
├── proyecto1/
│   ├── notebooks/            # Los dos entregables, ejecutables en Colab
│   ├── eval/                 # M2: eval set, rúbrica del juez, harness y scorecard
│   ├── data_collection/      # Recolección RSS + limpieza + supervisión débil
│   ├── data/                 # processed/ y gold/ se versionan; raw/ no (ver .gitignore)
│   └── docs/                 # Dataset, resultados de M1 y harness de M2
├── scripts/check_secrets.py  # Bloquea credenciales antes de que entren al repo
├── .github/workflows/        # Lint en GitHub Actions
├── .gitlab-ci.yml            # El mismo lint en GitLab
└── .pre-commit-config.yaml   # Los mismos checks, en local antes de commitear
```

### Buenas prácticas

- **Lint automático en cada push**: búsqueda de credenciales, `ruff` (Python y notebooks) y
  `pymarkdown`, en un job sin GPU. El repo vive en GitLab (trabajo diario) y en GitHub (entrega
  pública), así que los mismos checks están definidos en los dos.
- **Hooks locales opcionales**, idénticos al CI:

  ```bash
  pip install -r requirements-dev.txt
  pre-commit install
  ```

- **Nada de datos crudos ni pesos en git**: `.gitignore` excluye `proyecto1/data/raw/` y los
  checkpoints. El porqué está en [`docs/dataset.md`](proyecto1/docs/dataset.md#licencia-y-uso): no
  redistribuimos texto periodístico, solo fragmentos de sindicación con su enlace de vuelta.
- **Licencia separada para código y datos**: el código es MIT; el texto de prensa sigue siendo de
  cada medio, y así está dicho en [`LICENSE`](LICENSE).
- **Ramas cortas y revisión**: `feature/...` o `fix/...`, con revisión de otro integrante antes de
  mergear a `main`.
