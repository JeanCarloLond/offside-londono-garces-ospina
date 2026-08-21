# Eval set de dominio y harness de evaluación (M2)

> Entregable de la Sesión 5. Responde las tres cosas que pide el módulo: **(1)** diez ejemplos gold de nuestro dominio con su salida esperada, **(2)** qué hace *buena* a una respuesta en nuestro dominio, y **(3)** el harness ejecutable que lo convierte en un scorecard.
>
> Artefactos: [`../eval/eval_set.jsonl`](../eval/eval_set.jsonl) · [`../eval/rubric.yaml`](../eval/rubric.yaml) · [`../eval/harness.py`](../eval/harness.py)

---

## Por qué construimos un eval set propio

Los benchmarks públicos (MMLU, GSM8K, MT-Bench) miden conocimiento general, y ninguno mide nuestro dominio. Peor: se **contaminan** — si las preguntas del test se filtraron al entrenamiento, el modelo las acierta porque las vio, no porque razone, y el puntaje sube sin que el sistema mejore.

Un eval set propio es la única medida que no está contaminada (la hicimos nosotros), que mide lo que a nuestro usuario le importa, y que sirve de vara fija el resto del semestre: el mismo set evaluará el RAG de M3 y el componente visual de M4.

---

## 1 · Los diez ejemplos gold

Son noticias **reales** de nuestro corpus (agosto 2026, prensa deportiva), etiquetadas a mano por el equipo. El texto no se transcribe: [`build_eval_set.py`](../eval/build_eval_set.py) lo extrae del corpus por `id`, así que es byte a byte el que recolectamos, y cada ejemplo arrastra su enlace de origen.

| ID | Input (fragmento, recortado) | Salida esperada | Equipo | Caso difícil | Léxico |
|---|---|---|---|---|---|
| `OFF-01` | El Getafe comunica "una grave lesión" en la rodilla derecha de Uche. El club confirma que el nigeriano se perderá toda… | `baja_confirmada` / `negativo_alto` | Getafe | declaración que contiene un hecho | ❌ |
| `OFF-02` | Chaira, baja sensible para el Oviedo. El marroquí se perderá entre dos y tres meses de competición… | `baja_confirmada` / `negativo_alto` | Real Oviedo | falso negativo total | ❌ |
| `OFF-03` | El Newcastle se lleva el Naranja. Remontó el golazo de Ugrinic en un partido en el que Gayà fue expulsado por una dura… | `irrelevante` / `neutro` | Valencia | vocabulario disparador fuera de contexto | ❌ |
| `OFF-04` | Mudryk, 615 días después: "Este ha sido el periodo más difícil de mi carrera". Fue suspendido por la FA por dopaje en… | `regreso_alta` / `positivo_alto` | Chelsea | tiempo verbal, signo invertido | ❌ |
| `OFF-05` | Álex Pastor reaparece once meses después. El central tiene una contrarreloj por delante para recuperar la forma | `regreso_alta` / `positivo_bajo` | — | regreso sin vocabulario de regreso | ❌ |
| `OFF-06` | Flick los tiene enchufados: Joan García, Eric y Gordon adelantan su vuelta al trabajo. Se ejercitaron este domingo… | `regreso_alta` / `positivo_bajo` | FC Barcelona | impacto sobrestimado | ❌ |
| `OFF-07` | Joan Jordán, en Lisboa para cerrar su fichaje por el Estrela Amadora. El meta de Las Palmas tuvo que pasar por el quir… | `baja_confirmada` / `negativo_alto` | Las Palmas | fragmento compuesto | ✅ |
| `OFF-08` | La UEFA cambia el ciclo de amarillas en las competiciones europeas. La organización efectuará cambios en la normativa… | `irrelevante` / `neutro` | ninguno | distractor normativo | ✅ |
| `OFF-09` | Larcamón anticipa refuerzos "en breve" tras las dudas defensivas del Sporting… | `declaracion_contexto` / `neutro` | Sporting de Gijón | homonimia de "dudas" | ✅ |
| `OFF-10` | Manuel Neuer insinúa que esta podría ser su última temporada. El guardameta alemán afirmó que estuvo a punto de retira… | `declaracion_contexto` / `neutro` | Bayern Múnich | frontera declaración/rumor | ✅ |

La columna **Léxico** indica si nuestro baseline de regex se equivoca en ese ejemplo. Cada fila lleva en el JSONL un campo `why` que explica por qué esa es la respuesta correcta.

### Cómo cumplimos los cuatro criterios de un buen eval set

**Representativo.** Son fragmentos reales del flujo que nuestro usuario lee: titulares y entradillas de prensa deportiva, tal cual llegan por RSS. No están reescritos ni simplificados.

**Con salida esperada.** Cada input trae categoría, impacto y equipo afectado. La tripleta es la unidad de decisión del sistema, no solo la etiqueta.

**Cubre casos difíciles** — y esto es lo que decidió la selección. No son diez ejemplos al azar: se eligieron buscando dónde se rompe la clasificación. Los seis primeros son fallos demostrados del baseline, cada uno de un tipo distinto:

- *Declaración que contiene un hecho* (`OFF-01`): las comillas del titular disparan `declaracion_contexto` y el léxico se detiene antes de leer "se perderá toda la temporada". Distinguir el continente del contenido es exactamente lo que un modelo con contexto debería hacer mejor que un regex.
- *Falso negativo total* (`OFF-02`): la noticia más accionable del corpus — una baja de dos a tres meses — es invisible para el baseline, porque "baja sensible" solo está en el ajuste de sentimiento, no como patrón de categoría.
- *Vocabulario disparador fuera de contexto* (`OFF-03`): hay una expulsión, pero en un amistoso de pretemporada. No arrastra sanción liguera, así que no cambia la disponibilidad para ningún partido apostable. El léxico grita "alto impacto" sobre puro ruido.
- *Tiempo verbal con el signo invertido* (`OFF-04`): la noticia es que el jugador **vuelve** tras 615 días; la suspensión está en pasado y es el contexto. El léxico lee "suspendido" y devuelve impacto negativo alto cuando la respuesta correcta tiene el signo contrario.
- *Regreso sin vocabulario de regreso* (`OFF-05`): "reaparece once meses después" no coincide con ninguna fórmula del léxico. Además el impacto es positivo **bajo**: el propio texto dice que tiene una contrarreloj para recuperar la forma.
- *Impacto sobrestimado* (`OFF-06`): la categoría la acierta, la intensidad no. Adelantar la vuelta a los entrenamientos no es estar disponible para jugar.

Los cuatro últimos son **distractores**: el léxico acierta, pero por poco. Existen a propósito — un eval set formado solo por fallos también está sesgado, y no detectaría una regresión que rompa lo que hoy sí funciona. `OFF-07` es además un test de regresión explícito del parche que le pusimos al léxico cuando encontramos ese error a mano.

**No contaminado.** Son noticias de agosto de 2026 etiquetadas por nosotros; no salen de ningún benchmark público. Ver la sección de contaminación más abajo para el matiz importante.

---

## 2 · Qué hace "buena" a una respuesta en nuestro dominio

La rúbrica completa, con pesos, está en [`rubric.yaml`](../eval/rubric.yaml). El principio que la ordena es este:

> Una respuesta es buena si el usuario, actuando solo con ella, decide mejor que si hubiera leído el titular por su cuenta. De ahí se sigue que un error que le hace **actuar sobre información falsa** es más grave que un error que simplemente **no le aporta nada**: en el segundo caso queda como estaba, en el primero decide peor por culpa nuestra.

Esa asimetría es la que reparte los pesos:

| Criterio | Peso | Por qué pesa lo que pesa |
|---|---:|---|
| **C2 · Signo del impacto correcto** | 0.25 | El error más caro. Confundir una buena noticia con una mala empuja al usuario en la dirección contraria. `OFF-04` es exactamente este caso |
| **C1 · Categoría correcta** | 0.20 | La base: si el tipo de hecho está mal, todo lo que se construye encima está mal |
| **C4 · No presenta un rumor como un hecho** | 0.20 | Funcional y ético: propaga información falsa sobre personas identificables. Es la razón por la que `rumor_no_confirmado` existe como clase |
| **C5 · No inventa señal de alto impacto** | 0.20 | Precisión, no exhaustividad. Preferimos que se nos escape una noticia a inventar una |
| **C3 · Intensidad correcta** | 0.10 | Degrada la señal pero no la invierte, por eso pesa menos que el signo |
| **C6 · Equipo afectado correcto** | 0.05 | Declarado pero sin evaluar: el modelo de M1 clasifica, no extrae equipo. Llega en M2/M3 |

A esto se suman tres **criterios estructurales** que se cumplen por diseño del sistema y no por el modelo: **trazabilidad** (toda salida va con enlace y marca temporal, para que el usuario verifique en vez de confiar), **abstención explícita** (bajo umbral de confianza el sistema responde "sin señal clara" en vez de forzar una etiqueta) y **no recomendar apostar** (el sistema informa; no calcula montos ni promete rentabilidad).

En resumen, y en una frase defendible ante cualquiera: **una respuesta buena acierta el tipo de hecho, no se equivoca en la dirección del impacto, no confunde un rumor con un hecho, y ante la duda calla en vez de inventar.**

---

## 3 · El harness y el scorecard del baseline

[`harness.py`](../eval/harness.py) combina las tres dimensiones que pide el módulo sobre el eval set. Corre en menos de un segundo, sin GPU y sin dependencias de ML:

```bash
cd proyecto1/eval
python harness.py                    # baselines
python harness.py --json scorecard.json
```

| Dimensión | Qué es aquí | Estado |
|---|---|---|
| **1 · Métrica clásica** | **F1 macro** sobre categoría (más accuracy solo como contraste). Automática, barata, reproducible | ✅ |
| **2 · Rúbrica / juez** | Los criterios C1–C5 de `rubric.yaml`, evaluados de forma determinista | ✅ parcial — el **LLM-as-a-judge** llega en S06 y recibirá esta misma rúbrica como prompt |
| **3 · De dominio** | Desglose por tipo de caso difícil, y recuento de los errores que más caros salen: señal falsa de alto impacto, señal perdida y signo invertido | ✅ |

### Scorecard del baseline

```text
DIMENSIÓN 1 · MÉTRICA CLÁSICA
sistema          F1 macro    accuracy
mayoritaria        0.0417      0.2000
lexico             0.2750      0.5000

DIMENSIÓN 2 · RÚBRICA
sistema            C1      C2      C3      C4      C5    GLOBAL
mayoritaria      0.20    0.40    0.40     n/d     n/d     0.327
lexico           0.50    0.50    0.50     n/d    0.50     0.500

DIMENSIÓN 3 · ERRORES CAROS
lexico:  señal FALSA de alto impacto: 2 (OFF-03, OFF-06)
         señal PERDIDA de alto impacto: 2 (OFF-01, OFF-02)
         signo del impacto INVERTIDO: 1 (OFF-04)
```

`C4` sale `n/d` porque el eval set actual no contiene ningún ejemplo cuya verdad sea `rumor_no_confirmado` — no había ninguno claro en el corpus de pretemporada. El harness lo reporta como no medible en vez de dar un 1.0 gratis; ampliar el set con rumores reales es la primera tarea pendiente.

### El hallazgo que justifica todo el ejercicio

Sobre el conjunto de validación de M1, el baseline de léxico sacaba **F1 macro 1.0000**. Sobre este eval set de dominio saca **0.2750**.

El léxico no ha cambiado. Lo que cambió es que dejamos de medirlo con ejemplos que él mismo etiquetó y empezamos a medirlo con casos elegidos para ser difíciles. Ese 1.00 era un artefacto de un conjunto de validación fácil y parcialmente circular; el 0.275 es la medición honesta.

Es exactamente la lección del módulo: **un número alto puede venir del test, no del sistema.** Y es la mejor prueba de por qué hacía falta este eval set.

---

## Contaminación: el eval set debe excluirse del entrenamiento

Los diez ejemplos se curaron a partir del mismo corpus que alimenta el train, así que el modelo de M1 los vio durante el entrenamiento. Medirlo contra ellos sería medir memoria, no generalización — justo la contaminación que infla los leaderboards.

El harness lo detecta y lo reporta:

```text
CONTAMINACIÓN / HOLD-OUT
  10 de los 10 ejemplos del eval set están en el corpus que alimenta
  el entrenamiento. DEBEN excluirse del train.
```

La solución está implementada: `harness.eval_corpus_ids()` expone los ids, y [`train_holdout_model.py`](../eval/train_holdout_model.py) reentrena excluyéndolos, **abortando si alguno se cuela**. Ese es el adaptador que se puede evaluar honestamente:

```bash
python train_holdout_model.py
python harness.py --adapter ../m1_lora_adapter_holdout
```

---

## Un bug de reproducibilidad que encontró el harness

Al añadir el modelo afinado al scorecard apareció algo raro: **el mismo adaptador daba un número distinto en cada corrida** (F1 macro 0.0000, luego 0.0312, luego 0.0500). Un modelo cargado desde disco y evaluado en modo `eval()` debería ser determinista.

**La causa.** BETO es un checkpoint de *masked language modeling*: no incluye `bert.pooler`. Cuando se carga como `AutoModelForSequenceClassification`, transformers avisa de ello (`bert.pooler.dense.weight | MISSING`) y **lo inicializa al azar**. Ese pooler está entre el encoder y la cabeza de clasificación, así que forma parte de la función del modelo. Durante el fine-tuning con LoRA queda congelado en esos valores aleatorios, y la cabeza aprende a leer *ese* pooler concreto — pero el adaptador solo guarda las matrices LoRA y el `classifier`. Al recargar aparece un pooler distinto, la cabeza entrenada ya no encaja con él, y las predicciones se degradan de forma silenciosa.

**Cómo lo confirmamos.** Fijando la misma semilla del entrenamiento (`torch.manual_seed(42)`) antes de construir el modelo, el pooler se reconstruye idéntico y el resultado vuelve a ser determinista y reproducible. Es la prueba de que la fuente de aleatoriedad era esa y no otra: sin semilla los números bailan, con la semilla del train salen siempre iguales.

**El arreglo.** No depender de una semilla —eso es frágil— sino hacer el adaptador autocontenido:

```python
LoraConfig(..., modules_to_save=["classifier", "pooler"])
```

Y para que el fallo no pueda volver en silencio, el harness ahora **revisa el `adapter_config.json` antes de cargar** y avisa si el adaptador no persiste el pooler.

**Por qué importa más allá de este caso.** Un modelo que rinde distinto cada vez que se carga no es evaluable, y el síntoma no se parece a un bug: se parece a "el modelo es malo". Sin el harness pidiendo el mismo número dos veces, esto habría pasado desapercibido y habríamos reportado como resultado lo que era ruido de inicialización.

---

## Pendientes hacia la entrega M2

1. **Ampliar a rumores reales** para poder medir C4, hoy no evaluable.
2. **LLM-as-a-judge** (S06): la rúbrica ya está escrita y es el prompt; falta el juez.
3. **Abstención calibrada**: implementar el umbral de confianza que hace posible el criterio estructural E2.
4. **Extracción de equipo afectado** para activar C6.
5. **Crecer el set** más allá de diez ejemplos, manteniendo la proporción de casos difíciles.
