# Descripción del dataset

> Corte vigente: **2026-08-28** (ampliado en temporada tras la entrega de M1). Este documento describe el dataset *tal como está hoy*, no el dataset final del proyecto (meta: 4.000–6.000 fragmentos + gold de 600–800, ver `Context.md`). Es un primer corte real, pequeño y honestamente limitado — ver la sección de sesgos.

## Fuente

Texto real de prensa deportiva en español, recolectado vía **RSS público** con [`data_collection/collect_rss.py`](../data_collection/collect_rss.py):

| Fuente (`source`) | Región | Fragmentos |
|---|---|---:|
| `marca` | España | 140 |
| `marca_premier` | España | 92 |
| `marca_bundesliga` | España | 81 |
| `marca_segunda` | España | 77 |
| `marca_laliga` | España | 58 |
| `marca_champions` | España | 32 |
| `eltiempo_deportes` | Colombia | 10 |
| `eltiempo_futbol_colombiano` | Colombia | 7 |

Guardamos **solo lo que el propio medio publica para sindicación** (título + resumen corto de 1–2 frases), nunca el cuerpo del artículo. Cada fila trae `link`, `published_at` y un hash del link — el rastro para verificar la fuente sin redistribuir el artículo (ver "Licencia y uso" abajo).

De las 11 fuentes candidatas verificadas manualmente (`data_collection/rss_sources.yaml`), solo estas 8 dan contenido real. `as.com` y `sport.es` responden 200 con XML válido pero canales vacíos (feeds abandonados, no rotos); el feed en inglés de ESPN se descartó por idioma. Detalle completo de qué se probó y por qué en los comentarios de `rss_sources.yaml`.

Además se generó un **seed sintético** (`data_collection/build_gold_seed.py`) de 30 fragmentos escritos por el equipo con clubes y jugadores ficticios — ver "Por qué hay datos sintéticos" abajo.

## Tamaño y splits

| Conjunto | Fragmentos | Cómo se arma |
|---|---:|---|
| `data/raw/news_raw.jsonl` (no versionado) | 497 | Salida cruda de `collect_rss.py`, deduplicada por hash de link |
| `data/processed/weak_labeled.jsonl` | 497 | Supervisión débil (léxico/regex, `weak_label.py`) |
| `data/gold/gold_seed_synthetic.jsonl` | 30 | Sintético, entidades ficticias, cubre categorías con poco volumen real |
| `data/gold/gold_verified.jsonl` | 48 | Verificación manual sobre el 20% más reciente por categoría |
| `eval/eval_set.json` (M2) | 21 | Eval set de dominio, curado a mano y **excluido del entrenamiento** |

**Split del entrenamiento vigente: 407 train / 100 validation**, tras excluir del corpus los 20 ids que forman el eval set (`Corpus: 497 -> 477`). El corte anterior, de pretemporada, era de 251 fragmentos y 227/54.

**Por qué así y no aleatorio:** la porción real se separa **por fecha** (`published_at`), últimos ~20% por categoría a validación — evita fuga temporal (entrenar con una noticia y validar con su eco de la misma tarde). La porción sintética no tiene fecha real, así que se separa con muestreo aleatorio estratificado por categoría, semilla fija (`SEED=42`).

## Idioma y licencia

- **Idioma:** español. 480 fragmentos de prensa española y 17 de prensa colombiana (El Tiempo) — ver el sesgo de variante lingüística, que sigue siendo fuerte aunque ya no sea total. Código, nombres de variables y esta documentación técnica en inglés/español mixto siguiendo la convención del equipo (`Context.md`).
- **Licencia del texto:** el contenido periodístico es propiedad de cada medio (Unidad Editorial/Marca, etc.). **No se redistribuye el artículo completo** en ningún momento — ni en el scraping ni en el repo. Lo que se versiona son fragmentos cortos (título + resumen de sindicación RSS, pensados por el propio medio para circular fuera de su sitio) más el link de vuelta a la fuente. `data/raw/` (que si pudiera acumular más contexto crudo) está en `.gitignore` y nunca se commitea. Uso estrictamente académico, no comercial, respetando los `robots.txt`/ToS de cada medio (User-Agent identificado en `collect_rss.py` con contacto del equipo).
- **Licencia del código:** MIT (`/LICENSE`), separada explícitamente de la licencia del texto.

## Tarea

- **Input:** un fragmento corto (`titulo. resumen`, limpiado de HTML) + metadata (`source`, `published_at`).
- **Output:** dos etiquetas categóricas sobre el mismo fragmento:
  - **Categoría del hecho** (8 clases): `baja_confirmada`, `sancion_suspension`, `duda_fisica`, `regreso_alta`, `cambio_tactico`, `declaracion_contexto`, `rumor_no_confirmado`, `irrelevante`.
  - **Impacto** (5 clases): `negativo_alto`, `negativo_bajo`, `neutro`, `positivo_bajo`, `positivo_alto`.
- Es clasificación de texto corto, no generación — de ahí la elección de un encoder (BETO) y no un modelo generativo. Ver `docs/results.md` para la justificación completa modelo↔tarea.

## Anotación (cómo se etiquetó cada conjunto)

1. **Train (supervisión débil):** `weak_label.py` aplica el léxico/regex de `lexicon.yaml` fragmento por fragmento. Rápido y barato, pero es exactamente el "baseline de nivel 2" del proyecto — sabemos que tiene errores sistemáticos (ver ejemplo abajo).
2. **Validation (gold, porción real):** los mismos fragmentos que el léxico etiquetaría, pero **releídos uno por uno** por el equipo. La primera pasada encontró 2 errores del léxico sobre 49 fragmentos (~96% de acuerdo), documentados con su motivo en `gold_corrections.py`; ambos se corrigieron y sirvieron para mejorar el léxico. El conjunto resultante (`data/gold/gold_verified.jsonl`) fue revisado por los tres integrantes antes de la entrega.
3. **Validation (porción sintética):** la categoría se fija al escribir el fragmento (`build_gold_seed.py`), no hay ambigüedad que anotar.

### Un error real que encontramos (y qué dice de la calidad del léxico)

Al verificar a mano, 2 de 49 fragmentos reales estaban mal etiquetados por el léxico: en ambos casos, el resumen RSS mezclaba **dos noticias distintas** en un solo fragmento (ej. un fichaje + "el meta de Las Palmas tuvo que pasar por el quirófano tras un choque fortuito"), y el léxico se quedaba con la mitad irrelevante. Corregimos el léxico (agregamos patrones para "pasó por el quirófano" y "adelantan su vuelta al trabajo") y quedó ~96% de acuerdo con la lectura manual en este corte — razonable para un baseline de regex, y una limitación real que un modelo con más contexto (fine-tuneado) debería poder resolver mejor que un match de substring.

## Por qué hay datos sintéticos en el gold set

El primer scraping se corrió el **2026-08-09, en plena pretemporada europea**, y de las 8 categorías solo cubría 4 con algo de volumen: `duda_fisica`, `cambio_tactico` y `rumor_no_confirmado` no aparecían en absoluto.

La recolección se repitió el **2026-08-27, con la liga ya en marcha**, y el corpus pasó de 251 a 497 fragmentos. Aparecieron los primeros ejemplos reales de `duda_fisica` y de alineaciones confirmadas, y rojas en partido de liga (no solo en amistoso). Aun así el desbalance sigue siendo severo —`irrelevante` es el 76 %— y `rumor_no_confirmado` **sigue sin aparecer ni una vez**: los únicos rumores que llegan por RSS son de fichajes, porque los de disponibilidad viven en el cuerpo del artículo o en redes, que no scrapeamos.

Para poder mostrar el pipeline de LoRA entrenando y evaluando las 8 categorías en este entregable, `build_gold_seed.py` agrega 30 fragmentos **escritos por el equipo, con clubes y jugadores ficticios a propósito** (nunca se le atribuye una lesión/sanción real a una persona real que no la tuvo — la misma razón ética que documentamos en `Context.md`, sección 7). Quedan marcados `source="synthetic_seed"` / `label_method="manual_synthetic"` en todo momento, nunca se mezclan silenciosamente con datos reales, y el modelo final debe evaluarse sobre todo con más texto real de temporada regular en cuanto esté disponible.

**Esto es un andamio, no el dataset final.** Ver "Próximos pasos" abajo.

## Sesgos y limitaciones conocidas

- **Volumen por clase (el más importante):** aunque el corpus se duplicó al recolectar en temporada, las clases accionables siguen siendo minoritarias: 4 `sancion_suspension`, 3 `baja_confirmada`, 2 `regreso_alta`, 1 `duda_fisica` y 0 `rumor_no_confirmado` frente a 378 `irrelevante`. El modelo aprende bien lo que abunda y falla en lo que importa.
- **Cobertura por club:** Marca es prensa española: sobrerrepresenta Real Madrid, Barcelona y LaLiga frente a equipos pequeños o ligas latinoamericanas — ver `Context.md`, sesgo de cobertura ya anticipado por el equipo.
- **Variante lingüística:** 480 de 497 fragmentos son de prensa española y solo 17 de prensa colombiana. El léxico sigue construido sobre vocabulario ibérico, así que el sesgo persiste aunque las fuentes latinoamericanas ya estén integradas. El eval set lo mide explícitamente con `OFF-15`, una roja en Liga BetPlay.
- **Desbalance de clases severo:** `irrelevante` es ~77-98% del corpus real según la pasada de recolección — ver `docs/results.md` para por qué esto hace que accuracy sea una métrica engañosa y F1 macro la elegida.
- **Fragmentos compuestos:** algunos resúmenes RSS mezclan dos noticias en un solo campo `summary` (ver el error real documentado arriba) — el léxico las etiqueta por la primera coincidencia, lo cual puede perder información.
- **Etiqueta de impacto es una heurística, no un análisis de a qué equipo afecta:** `impact_default` en el léxico asigna un impacto típico por categoría, no sabe leer "a quién perjudica" cuando el fragmento menciona a dos equipos. Es una limitación conocida del baseline (nivel 2) que un modelo con más contexto debería superar — y una que el propio equipo ya había anticipado en `Context.md`.
- **Datos sintéticos en el gold set** (ver sección anterior): entidades ficticias, vocabulario deliberadamente cercano al léxico — probablemente **infla** el desempeño reportado del baseline de léxico en esas clases (ver la nota metodológica en `docs/results.md`). No es un problema de fuga de datos con el modelo entrenado, pero sí limita cuánto podemos confiar en el número de esas clases específicas hasta reemplazarlas por texto real.

## Cómo se amplía

`collect_rss.py` es *append-only* y deduplica por hash del enlace, así que se puede correr todos los días sin duplicar nada: cada corrida solo añade lo nuevo. Es como se pasó de 251 a 497 fragmentos entre el corte de pretemporada y el de temporada regular.

Al ampliar hay que tener en cuenta dos cosas: los ids que forman el eval set (`eval/eval_set.json`) deben seguir excluidos del entrenamiento —`train_holdout_model.py` aborta si alguno se cuela— y conviene revisar si aparecen ejemplos reales de las clases que hoy están vacías, para retirar progresivamente el seed sintético.
