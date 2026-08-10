# Reporte de resultados — M1

> Corrida completa el 2026-08-09 (CPU local, validación antes de Colab — ver la nota de procedencia
> en la primera celda del notebook). Números y outputs genuinos, no inventados. Fuente completa:
> [`../notebooks/M1_finetuning_LoRA_offside.ipynb`](../notebooks/M1_finetuning_LoRA_offside.ipynb).

## Modelo base y familia

**Encoder — `dccuchile/bert-base-spanish-wwm-cased` (BETO).** La tarea central de Offside es
clasificación de texto corto (categoría del hecho + impacto), no generación: un encoder que atiende a
toda la frase a la vez y produce una representación para clasificar es la herramienta correcta para
este objetivo de pre-entrenamiento (MLM ↔ comprensión), mientras que pedirle esto a un decoder
generativo lo forzaría a "hablar" una etiqueta en vez de simplemente devolverla.

Entre BETO y DistilBERT multilingüe (la alternativa liviana considerada), medimos la **fertilidad del
tokenizador** sobre un glosario de 40 términos del dominio (vocabulario futbolístico + apellidos de
jugadores hispanos):

| Tokenizador | Fertilidad media (tokens/término) |
|---|---:|
| BETO (es) | **2.24** |
| DistilBERT multilingüe | 2.88 |
| DistilBERT inglés | 3.48 |

BETO parte el vocabulario del dominio en menos tokens — casos extremos como "regreso a la
convocatoria" son 4 tokens en BETO contra 9 en DistilBERT inglés. Con esto, BETO queda fijado como
modelo base.

## Baseline

Tres niveles, evaluados sobre el mismo conjunto de validación (54 fragmentos: 48 reales + 6
sintéticos, ver [`dataset.md`](dataset.md)):

1. **Clase mayoritaria** (`irrelevante`) — el piso.
2. **Léxico/regex** (`data_collection/lexicon.yaml`) — el mismo que etiqueta el train con supervisión
   débil.
3. **BETO zero-shot** — el modelo base con cabeza de clasificación sin entrenar. Es el baseline
   recomendado explícitamente por la guía de M1 ("el mismo modelo base sin fine-tuning") y el que
   usamos como comparación principal.

## Tabla de resultados

| Baseline / modelo | F1-macro (val) | Precisión "impacto alto" (val) |
|---|---:|---:|
| 1. Clase mayoritaria | 0.1033 | n/d (0 predicciones positivas) |
| 2. Léxico/regex | 1.0000 ⚠️ ver nota | 1.0000 ⚠️ ver nota |
| 3. BETO zero-shot (sin fine-tuning) | 0.0104 | 0.0769 |
| **4. LoRA fine-tuned (BETO + LoRA)** | **0.2468** | **0.5000** |

⚠️ **El 1.0000 del léxico está inflado y lo explicamos, no lo ocultamos:** 6 de las 8 categorías en
este conjunto de validación están representadas por un único ejemplo sintético que nosotros mismos
escribimos usando el vocabulario disparador del léxico (`build_gold_seed.py`) — es circular por
construcción. Sobre la porción real de la validación (irrelevante + declaracion_contexto, 48 de 54
filas) el léxico también acierta casi todo, pero ahí sí es una medición honesta (~96% de acuerdo con
la lectura manual en la primera pasada, ver `dataset.md`). La comparación que sí sostiene el peso de
esta entrega es **fine-tuned vs. zero-shot**: es la que pide la guía de M1 explícitamente.

## Lectura honesta

**¿Mejoró?** Sí, con claridad frente a los baselines no inflados: **F1-macro pasa de 0.0104
(zero-shot) a 0.2468 con LoRA** — más de 20×, y de 0.1033 a 0.2468 frente a la clase mayoritaria. En
la métrica secundaria (precisión sobre impacto alto), pasa de 0.077 (casi al azar) a 0.50.

**¿Cuánto, en la práctica?** El modelo aprendió bien las dos categorías con volumen real de
entrenamiento: `irrelevante` (F1 0.97, 157 ejemplos de train) y `declaracion_contexto` (F1 1.00, 42
ejemplos de train). **Sigue fallando por completo (F1 0.00) en las 6 categorías con 3-7 ejemplos de
entrenamiento cada una** — confirmado además con 3 frases nuevas escritas a mano, completamente fuera
del set de validación, donde el modelo falla las tres (dos las predice como `baja_confirmada`, una
como `irrelevante`; ver la última sección del notebook).

**¿Por qué pasa esto?** No es un problema del pipeline (LoRA está bien configurado, entrena y
converge — el `eval_loss` baja de 0.99 a 0.33 en 10 épocas) sino del **tamaño del dataset por
categoría**: el corte se recolectó en pretemporada (agosto 2026), cuando la prensa deportiva casi no
produce noticias de bajas médicas, sanciones o cambios tácticos de temporada regular. Es exactamente
el sesgo de "ventana temporal" que documentamos en `dataset.md` antes de entrenar nada — el resultado
del notebook lo confirma empíricamente en vez de contradecirlo.

## Ejemplos cualitativos

Ver la sección 7 del notebook para el detalle completo (8 ejemplos del set de validación + 3 frases
nuevas escritas a mano). Resumen:

| Texto (recortado) | Real | Predicho | ¿OK? |
|---|---|---|---|
| "Los nombres propios del mercado del Cádiz..." | irrelevante | irrelevante | ✅ |
| "El Barça tiene prisa: espera un pronto desenlace con el fichaje de Rodri..." | irrelevante | irrelevante | ✅ |
| "El defensor del Real Betis fue expulsado con roja directa..." (frase nueva) | sancion_suspension | baja_confirmada | ❌ |
| "Simeone confirmó en rueda de prensa que evalúa un cambio de sistema..." (frase nueva) | declaracion_contexto/cambio_tactico | baja_confirmada | ❌ |
| "El delantero recibe el alta médica y vuelve a la convocatoria..." (frase nueva) | regreso_alta | irrelevante | ❌ |

## Qué hace falta para que esto sea un resultado sólido (no solo un pipeline que funciona)

1. **Más texto real en las categorías raras** — correr `data_collection/collect_rss.py`
   repetidamente durante la temporada regular de liga (no solo pretemporada). Es el paso de mayor
   impacto esperado.
2. **Doble anotación real del gold set** entre los tres integrantes — hoy `data/gold/gold_verified.jsonl`
   solo tiene una primera pasada asistida por IA (ver `dataset.md`, sección "Anotación").
3. Con más datos, repetir esta misma comparación (los 3 baselines + LoRA) y ver si el delta se
   sostiene o crece — la expectativa honesta es que crezca, porque hoy el modelo literalmente no ha
   visto ejemplos suficientes de la mayoría de las clases.
