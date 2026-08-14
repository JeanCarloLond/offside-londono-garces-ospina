# Descripción del dataset

> Versión del corte usado en M1: **2026-08-09**. Este documento describe el dataset *tal como está hoy*, no el dataset final del proyecto (meta: 4.000–6.000 fragmentos + gold de 600–800, ver `Context.md`). Es un primer corte real, pequeño y honestamente limitado — ver la sección de sesgos.

## Fuente

Texto real de prensa deportiva en español, recolectado vía **RSS público** con [`data_collection/collect_rss.py`](../data_collection/collect_rss.py):

| Fuente (`source`) | Feed | Región |
|---|---|---|
| `marca` | marca.com/rss/futbol.xml | España |
| `marca_laliga` | LaLiga (marca) | España |
| `marca_segunda` | Segunda División (marca) | España |
| `marca_champions` | Champions League (marca) | España |
| `marca_premier` | Premier League (marca) | España |
| `marca_bundesliga` | Bundesliga (marca) | España |

Guardamos **solo lo que el propio medio publica para sindicación** (título + resumen corto de 1–2 frases), nunca el cuerpo del artículo. Cada fila trae `link`, `published_at` y un hash del link — el rastro para verificar la fuente sin redistribuir el artículo (ver "Licencia y uso" abajo).

De las 9 fuentes candidatas verificadas manualmente (`data_collection/rss_sources.yaml`), solo estas 6 dan contenido real. `as.com` y `sport.es` responden 200 con XML válido pero canales vacíos (feeds abandonados, no rotos); el feed en inglés de ESPN se descartó por idioma. Detalle completo de qué se probó y por qué en los comentarios de `rss_sources.yaml`.

Además se generó un **seed sintético** (`data_collection/build_gold_seed.py`) de 30 fragmentos escritos por el equipo con clubes y jugadores ficticios — ver "Por qué hay datos sintéticos" abajo.

## Tamaño y splits

| Conjunto | Fragmentos | Cómo se arma |
|---|---:|---|
| `data/raw/news_raw.jsonl` (no versionado) | 251 | Salida cruda de `collect_rss.py`, deduplicada por hash de link |
| `data/processed/weak_labeled.jsonl` (train) | 251 → 227 en train | Supervisión débil (léxico/regex, `weak_label.py`) |
| `data/gold/gold_seed_synthetic.jsonl` | 30 | Sintético, team-authored, cubre categorías ausentes en el scrape real |
| `data/gold/gold_verified.jsonl` (val, porción real) | 48 | Verificación manual (lectura fragmento a fragmento) sobre el 20% más reciente por categoría del conjunto real |

**Split final del notebook de M1: 227 train / 54 validation** (227 = 221 reales + 6 sintéticos held-in; 54 = 48 reales verificados + 6 sintéticos held-out).

**Por qué así y no aleatorio:** la porción real se separa **por fecha** (`published_at`), últimos ~20% por categoría a validación — evita fuga temporal (entrenar con una noticia y validar con su eco de la misma tarde). La porción sintética no tiene fecha real, así que se separa con muestreo aleatorio estratificado por categoría, semilla fija (`SEED=42`).

## Idioma y licencia

- **Idioma:** español (España, en este corte — ver sesgo de variante lingüística). Código, nombres de variables y esta documentación técnica en inglés/español mixto siguiendo la convención del equipo (`Context.md`).
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

El scraping se corrió el **2026-08-09, en plena pretemporada europea**. La prensa de estos días está dominada por fichajes y resultados de amistosos — no por partes médicos ni sanciones de temporada regular. Resultado honesto: de las 8 categorías, el scrape real solo cubre 4 con volumen (`irrelevante`, `declaracion_contexto`, y unos pocos `baja_confirmada`/`sancion_suspension`/`regreso_alta`); `duda_fisica`, `cambio_tactico` y `rumor_no_confirmado` casi no aparecen todavía.

Para poder mostrar el pipeline de LoRA entrenando y evaluando las 8 categorías en este entregable, `build_gold_seed.py` agrega 30 fragmentos **escritos por el equipo, con clubes y jugadores ficticios a propósito** (nunca se le atribuye una lesión/sanción real a una persona real que no la tuvo — la misma razón ética que documentamos en `Context.md`, sección 7). Quedan marcados `source="synthetic_seed"` / `label_method="manual_synthetic"` en todo momento, nunca se mezclan silenciosamente con datos reales, y el modelo final debe evaluarse sobre todo con más texto real de temporada regular en cuanto esté disponible.

**Esto es un andamio, no el dataset final.** Ver "Próximos pasos" abajo.

## Sesgos y limitaciones conocidas

- **Ventana temporal (el más importante en este corte):** datos de pretemporada, agosto de 2026. El modelo entrenado aquí prácticamente no ha visto ejemplos reales de `duda_fisica`, `cambio_tactico` ni `rumor_no_confirmado` — solo los sintéticos. Esperamos que el rendimiento en esas clases mejore sustancialmente en cuanto se recolecte texto de jornadas de liga reales (partes médicos, sanciones de árbitro, alineaciones probables).
- **Cobertura por club:** Marca es prensa española: sobrerrepresenta Real Madrid, Barcelona y LaLiga frente a equipos pequeños o ligas latinoamericanas — ver `Context.md`, sesgo de cobertura ya anticipado por el equipo.
- **Variante lingüística:** 100% de las fuentes activas son de España (`region: es`). El léxico usa vocabulario ibérico ("sancionado", no necesariamente "suspendido" como se diría en otra variante); pendiente sumar una fuente latam (ver `rss_sources.yaml`, TODO).
- **Desbalance de clases severo:** `irrelevante` es ~77-98% del corpus real según la pasada de recolección — ver `docs/results.md` para por qué esto hace que accuracy sea una métrica engañosa y F1 macro la elegida.
- **Fragmentos compuestos:** algunos resúmenes RSS mezclan dos noticias en un solo campo `summary` (ver el error real documentado arriba) — el léxico las etiqueta por la primera coincidencia, lo cual puede perder información.
- **Etiqueta de impacto es una heurística, no un análisis de a qué equipo afecta:** `impact_default` en el léxico asigna un impacto típico por categoría, no sabe leer "a quién perjudica" cuando el fragmento menciona a dos equipos. Es una limitación conocida del baseline (nivel 2) que un modelo con más contexto debería superar — y una que el propio equipo ya había anticipado en `Context.md`.
- **Datos sintéticos en el gold set** (ver sección anterior): entidades ficticias, vocabulario deliberadamente cercano al léxico — probablemente **infla** el desempeño reportado del baseline de léxico en esas clases (ver la nota metodológica en `docs/results.md`). No es un problema de fuga de datos con el modelo entrenado, pero sí limita cuánto podemos confiar en el número de esas clases específicas hasta reemplazarlas por texto real.

## Próximos pasos (antes de M2)

1. Correr `collect_rss.py` repetidamente durante las próximas semanas (liga ya en marcha) para acumular volumen real en las categorías hoy vacías — es append-only y deduplica por hash, así que correrlo todos los días es seguro.
2. Sumar 1-2 fuentes latam para atacar el sesgo de variante lingüística.
3. Ampliar el conjunto dorado hacia la meta de 600-800 fragmentos, con doble anotación sobre una muestra y métrica de acuerdo entre anotadores (`Context.md`).
4. Retirar progresivamente el seed sintético a medida que haya suficiente texto real por categoría.
