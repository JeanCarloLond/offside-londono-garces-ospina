# M3 · RAG que recupera, actúa y se evalúa

Este documento es el reporte largo del Módulo 3. El resumen, con las dos tablas y la lectura corta,
está en el [README](../../README.md#m3--rag-que-recupera-actúa-y-se-evalúa). Todo número de aquí sale
de un archivo de [`rag/resultados/`](../rag/resultados/) y se regenera con los scripts de
[`rag/`](../rag/).

## 1 · Corpus

| | |
|---|---|
| Fuente | Feeds RSS de **Marca** (portada, Primera, Segunda, Champions, Premier, Bundesliga) y **El Tiempo** (deportes y fútbol colombiano), recolectados por nuestro `collect_rss.py` |
| Cortes | Agosto de 2026 (pretemporada e inicio de liga) y **14 de octubre de 2026** (liga en marcha, +258 fragmentos) |
| Tamaño | **748 documentos = 748 chunks** (7 descartados por tener menos de 40 caracteres) |
| Fechas | 2026-05-09 a 2026-10-14 (el grueso, agosto y octubre) |
| Longitud | mediana 162 caracteres, p90 225, máximo 564 |
| Licencia | El texto es de cada medio. Guardamos solo lo que publican para sindicación (titular + entradilla) con el enlace de vuelta, nunca el artículo. Ver [`dataset.md`](dataset.md#idioma-y-licencia). El código es MIT |
| Archivo | [`data/rag/corpus.jsonl`](../data/rag/corpus.jsonl), construido por [`rag/ingest.py`](../rag/ingest.py) desde el crudo (no versionado) |

El corpus del RAG no reemplaza a `data/processed/weak_labeled.jsonl`, que es el train de M1/M2 y
queda intacto para que esos números sigan siendo reproducibles.

Encontramos un problema en los datos: en algunos ítems el feed de Marca pega el resumen de otra
noticia. El de Aaron Ochoa tiene la URL y el titular del Málaga, pero el resumen habla del
técnico del Girona; el mismo resumen del portero de Las Palmas aparece bajo cuatro titulares distintos.
El eval set de M2 heredó el primero (OFF-12 está etiquetado como «Girona»). No lo corregimos a mano
porque es ruido real del corpus; en los resultados se ve qué preguntas afecta.

## 2 · Pipeline de indexación

```text
collect_rss.py ─► ingest.py ─► corpus.jsonl ─► indice.py
 (RSS crudo)      limpia HTML,   748 docs       chunk: 1 doc = 1 chunk
                  dedup, guarda                 embed: multilingual-e5-small
                  fecha, medio y                store: Chroma (coseno) + BM25 en memoria
                  sección de la URL
```

`rag/ingest.py` necesita el crudo (`data/raw/`), que no se versiona por derechos de autor; por eso
se versiona su salida, `corpus.jsonl`, y el resto del pipeline parte de ahí.
`python rag/indice.py --forzar` reconstruye el índice en menos de un minuto en CPU. El índice no se
versiona (`rag/_chroma/` está en `.gitignore`), así quien clone el repo comprueba que el pipeline
corre de principio a fin.

Lo que se indexa es la sección de la URL más el texto: en Marca la sección es el club, y en muchos
fragmentos el club solo aparece ahí («Quique abre el casting para la banda derecha» no dice que es el
Alavés). Lo que ve el generador lleva además la fecha y el medio delante, porque «estará de baja tres
meses» no significa nada sin saber cuándo se publicó.

## 3 · Las tres decisiones

**Chunks: un fragmento RSS por chunk, sin solapamiento.** Nuestros fragmentos ya son la unidad mínima
con sentido (titular más entradilla, mediana de 162 caracteres) y en ellos el sujeto y el hecho suelen
quedar en frases distintas: «*Chaira, baja sensible para el Oviedo. El marroquí se perderá entre dos y
tres meses*». Partirlos separaría al jugador de su lesión, y como ningún chunk corta una frase no hay
nada que solapar.

**Embeddings: `intfloat/multilingual-e5-small`.** Medido sobre nuestras 23 preguntas con respuesta,
con el mismo recuperador denso ([`embeddings_m3.csv`](../rag/resultados/embeddings_m3.csv)):

| Modelo | Parámetros | recall@1 | recall@4 | MRR | Sin el relevante en el top-50 |
|---|---:|---:|---:|---:|---:|
| **multilingual-e5-small** | 118 M | 0,83 | **0,96** | 0,90 | 0 |
| paraphrase-multilingual-MiniLM-L12-v2 | 118 M | 0,48 | 0,61 | 0,53 | 4 |
| multilingual-e5-base | 278 M | 0,91 | 0,96 | 0,94 | 0 |

El modelo de paráfrasis rinde mucho peor porque nuestras preguntas y los titulares no son paráfrasis: «¿Cuenta
el Sevilla con Guridi para jugar en el Bernabéu?» frente a «El Sevilla prepara la visita al Bernabéu
sin Guridi». e5-base gana en recall@1 y en MRR (0,94 frente a 0,90), pero con el reranker e5-small
llega a 0,96, y e5-base tiene más del doble de parámetros en una GPU que también tiene que cargar el
generador y el juez.

**top-k = 4.** Cada pregunta se responde con una o dos noticias. Con k = 4 el documento correcto llega
en 22 de 23 preguntas, y subir a k = 10 no rescata la que falta (llega en la posición 11). Más chunks
no traen información nueva y sí distractores, que confunden a un generador de 1.5B (pasa en
Bentancur y Celta, ver el diagnóstico).

## 4 · El eval set

[`eval/eval_set_m3.json`](../eval/eval_set_m3.json), generado y validado por
[`eval/build_eval_set_m3.py`](../eval/build_eval_set_m3.py). Nuestra tarea de M1 era de clasificación,
así que escribimos preguntas nuevas sobre el corpus sin cambiar la vara: cada una tiene la misma
respuesta estructurada que el harness de M2 ya sabe medir (categoría del hecho, impacto, equipo).

| | |
|---|---|
| Casos | 27, con `input`, `esperado` y `criterio` (más `referencia` y `referencia_afirmaciones` para RAGAS y `docs_relevantes` para diagnosticar la recuperación) |
| Fuera del corpus | 4 (Mbappé y el Clásico, lesión de Lamine Yamal, cuota del Betis-Oporto, Balón de Oro 2025): lo esperado es la válvula, que en el lenguaje del harness es «sin señal» (`irrelevante · neutro · ninguno`) |
| Necesitan calcular fechas | 3 (Aaron Ochoa el 1 de noviembre y el 15 de diciembre, Chaira el 29 de noviembre) |
| Adversariales | 17: conflictos temporales (Musiala, Vinicius), pares mínimos (Vinicius en Liga frente a Champions; Ochoa antes y después de su regreso estimado), negación (Beitia), nombre parcial (Nico), club solo en la URL (Ángel Pérez), titular metafórico (Svilar), pregunta que presupone el hecho (Erik Lira) |
| Categorías | las 8 |
| Fecha de la consulta | fija, 2026-10-14 (`config.FECHA_CONSULTA`), para que la tabla salga igual otro día |

Los casi-aciertos fuera del corpus son a propósito: «¿Kylian Mbappé se pierde el Clásico?» comparte
literalmente «se pierde el Clásico» con el titular de Vinicius.

Además hay un dev set, [`eval/dev_set_m3.json`](../eval/dev_set_m3.json) tiene 14 preguntas sobre otras
noticias (3 sin respuesta). Es donde se ajustaron el prompt base y el verificador. Ningún prompt ni
umbral se ajustó mirando el eval set.

## 5 · El RAG ingenuo y su válvula

`sistema_rag(pregunta)` ([`rag/sistemas.py`](../rag/sistemas.py)): top-4 por similitud densa, prompt
aumentado, `Qwen/Qwen2.5-1.5B-Instruct` con decodificación greedy. El modelo devuelve un JSON con la
respuesta en prosa y la señal; el parseo es tolerante (bloques ```json, comillas simples, comas
colgando, etiquetas mal escritas) y, si no hay JSON recuperable, la señal cae a «sin señal».

Versiones del prompt base, medidas sobre el dev set
([`ajuste_prompt_m3.csv`](../rag/resultados/ajuste_prompt_m3.csv)). Es el mismo para todas las filas
de la tabla, así que había que fijarlo antes de comparar:

| Versión | Qué cambia | Categoría correcta | Calla teniendo respuesta | Calla sin respuesta | Qué pasa |
|---|---|---:|---:|---:|---|
| v1 | La válvula como regla: «si no está, responde exactamente "No encuentro…"» | 8/14 | 5/11 | 3/3 | Copia la frase de la válvula aun con la noticia correcta en la posición 1 |
| v2 | + dos ejemplos resueltos con clubes ficticios | 3/14 | 8/11 | 3/3 | Imita el ejemplo («Lizancos está *sancionado*») en vez de leer |
| **v3** | Sin ejemplos; primer campo `noticia` (el número de la que responde, o 0) | **9/14** | **0/11** | 0/3 | Contesta todo… y nunca calla: a Pedri le atribuye la lesión de Guridi |
| v4 | + «comprueba que la noticia nombra al jugador» | 8/14 | 3/11 | 0/3 | No mejora a v3: calla de más y sigue sin callar cuando debe |

En el eval set la v1 confirma el diagnóstico: calla en 4 de 4 sin respuesta, pero también en 12 de las
23 que sí la tienen (fila `rag_ingenuo_prompt_v1` de [`scorecard_m3.csv`](../rag/resultados/scorecard_m3.csv)).

Con un generador de 1.5B no encontramos un prompt que haga las dos cosas: o calla de más o no calla
nunca. Congelamos v3, la que mejor responde, para todas las versiones. El ingenuo queda con su válvula de prompt
(`noticia = 0`) y lo que esa válvula no consigue se resolvió con una pieza aparte (sección 7).

## 6 · Diagnóstico del ingenuo

Primero, sin generar nada: ¿llega el chunk? ([`recuperacion_m3.csv`](../rag/resultados/recuperacion_m3.csv))

| Recuperador | recall@1 | recall@4 | MRR |
|---|---:|---:|---:|
| dense (el ingenuo) | 0,83 | 0,96 | 0,90 |
| BM25 | 0,91 | 0,96 | 0,93 |
| híbrido (RRF) | 0,87 | 0,96 | 0,92 |
| dense + reranker | 0,96 | 0,96 | 0,96 |
| **híbrido + reranker** | **0,96** | 0,96 | **0,96** |
| HyDE | 0,65 | 0,87 | 0,74 |
| multi-query | 0,65 | 0,78 | 0,74 |

La recuperación casi nunca es el cuello de botella en este corpus. Los fragmentos son cortos y
distintos entre sí, y nuestras preguntas casi siempre nombran al jugador. Solo M3-11 (el portero de la
Roma) no llega: el titular es metafórico («El rapero con guantes que quiere silenciar al Madrid») y ni
dense (posición 11), ni BM25 (23), ni HyDE (8), ni multi-query (15) lo traen al top-4.

Luego, con el generador: los fallos del ingenuo sobre el eval set, agrupados con la pregunta del
enunciado (¿no llegó el chunk, llegó y el modelo lo ignoró, o el dato no está?):

| Tipo de falla | Casos | ¿Llegó el chunk? |
|---|---|---|
| Inventa en preguntas sin respuesta (válvula 0/4) | Mbappé ← le traslada la sanción de Vinicius; Lamine; Balón de Oro; cuota | no hay chunk que llegar |
| Un distractor va delante del relevante | Musiala (relevante en posición 2: responde con la noticia de agosto, «el Bayern no descarta a Musiala para la Supercopa»), Celta (posición 2: la respuesta habla de otra noticia), Bentancur (mezcla «expulsado a los 3 minutos» de la Bundesliga) | sí, pero no primero |
| No llega | Svilar → inventa «Bernardo Silva» a partir de otro chunk | no |
| Hay que calcular una fecha | Ochoa el 15 de diciembre y Chaira el 29 de noviembre: repiten «tres meses» / «entre dos y tres meses» sin decir si llegan | sí |
| La etiqueta está mal con el chunk correcto en la posición 1 | Vinicius, Isaac Romero y Ángel Pérez (sanción → «baja»), Erik Lira (un rumor presentado como baja), Nico, Laporte, Beitia, Sadiq, Arrasate | sí; es un error de generación |

## 7 · Técnicas: qué falla ataca cada una

| Técnica | Falla que ataca | Evidencia del diagnóstico |
|---|---|---|
| **Reranking** con `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` sobre un pool de 20 | Un distractor va delante del relevante | Musiala, Celta y Ochoa pasan de la posición 2 a la 1; MRR 0,90 → 0,96 |
| **Búsqueda híbrida** BM25 + dense con RRF (k = 60) | Nombres propios raros y términos exactos | BM25 pone primero el relevante más a menudo que dense (recall@1 0,91 frente a 0,83); con el reranker encima, su aporte medido en este eval set es nulo (ver la lectura) |
| **Válvula verificada**: el mismo Qwen como clasificador sí/no por logits, chunk a chunk | Inventa en preguntas sin respuesta | El ingenuo activó la válvula 0 de 4 veces |
| *HyDE y multi-query* | *El chunk no llega* | Descartadas: no rescatan M3-11 y empeoran el resto (MRR 0,74), además de añadir una llamada al LLM |

Por qué un verificador y no un umbral del reranker: lo primero que probamos fue callar cuando el mejor
chunk puntúa bajo en el cross-encoder. No separa ([`valvula_dev_m3.csv`](../rag/resultados/valvula_dev_m3.csv)):
el cross-encoder (entrenado en mMARCO, pregunta → pasaje que contiene la respuesta) puntúa bajo las
preguntas de disponibilidad que exigen inferir («¿Puede jugar Sergio Álvarez?» frente a «sufre una
rotura muscular»). El mismo Qwen, en cambio, usado como clasificador de una sola decisión («¿es esta
noticia la fuente para responder?») y leyendo los logits de sí/no, acierta 13 de 14 en el dev set con
el umbral natural de 0,5. El que falla es Julián Alvarez, y es discutible: la noticia sí habla de él y
del Atlético. La primera versión del verificador rechazaba las preguntas de fechas (la noticia no
menciona la fecha por la que se pregunta) y la corregimos, también sobre el dev set, preguntando si es
«la fuente» aunque haya que deducir la respuesta.

Además de callar, el verificador filtra: solo pasan al generador los chunks que aprueba.

## 8 · La capa agéntica

**La herramienta:** `calcular_fecha(fecha_inicio, cantidad, unidad, fecha_objetivo)`
([`rag/herramientas.py`](../rag/herramientas.py)). Las bajas llegan como duraciones relativas a la
fecha de la noticia y el apostador pregunta por un día concreto. Un modelo de 1.5B se equivoca en esa
aritmética de calendario; una función no. El resultado incluye una «lectura» en español («el
2026-11-01 es 26 días ANTES del fin del plazo: para esa fecha el jugador TODAVÍA estará de baja»).

Esquema, parseo tolerante y tope de pasos. El esquema (nombre, descripción y parámetros, en formato
de *function calling*) va en el prompt. Lo que propone el modelo se parsea con tolerancia: etiquetas
`<tool_call>` o JSON desnudo, `herramienta`/`argumentos` o `name`/`arguments`, comillas simples,
argumentos como string JSON, nombres aproximados de la herramienta, fechas en español («27 de agosto
de 2026»), cantidades en palabras («tres») y rangos («2-3»: se usa el máximo, la lectura prudente).
Si los argumentos no son válidos, el error vuelve al modelo para que corrija, con un máximo de 3 pasos.
Las pruebas están en [`rag/test_parseo.py`](../rag/test_parseo.py) y corren en el CI.

Cuándo se usa. Intentamos tres veces que lo decidiera el LLM:

1. Con la plantilla nativa de tools de Qwen, el sistema recibe un bloque en inglés y el modelo
   respondía «0» (ninguna noticia) incluso a preguntas que sin herramienta contestaba bien.
2. Con la herramienta descrita dentro de nuestro prompt, no la llamó nunca: la orden «devuelve solo el
   JSON final» gana siempre.
3. Con un paso de planificación aparte, en el que el LLM decide si llamar: 0 de 4 llamadas cuando
   hacía falta (prueba con noticias ficticias, [`prueba_planificador_m3.csv`](../rag/resultados/prueba_planificador_m3.csv)).
   Leyendo esa decisión por logits tampoco separa: da 0,93 a «¿Qué lesión tiene Lunar?», que no
   necesita ningún cálculo. (Los intentos 1 y 2 y esta lectura por logits se midieron durante el
   desarrollo y no están versionados como CSV; el intento 3 sí, en el modo `llm` de la prueba.)

Este modelo no sabe decidir cuándo usar la herramienta, así que eso lo decide una regla: la pregunta
nombra una fecha concreta y algún chunk trae una duración con número. Los argumentos (fecha de la
noticia, cantidad, unidad y fecha objetivo) los propone el LLM. Con ese reparto, en la misma prueba,
la herramienta se llama en 4 de 4 casos que la necesitan y en 0 de 4 que no. Una de las
cuatro sale con el año equivocado («15 de marzo» → 2026-03-15, anterior a la noticia): la herramienta
lo rechaza, el error vuelve al modelo, el modelo repite el mismo error y el tope de pasos corta el
bucle al tercer intento.

La respuesta final sale del mismo prompt base del RAG avanzado, con el resultado de la herramienta
añadido como una entrada más del bloque de noticias. Entre el avanzado y el sistema final lo único que
cambia es esa entrada.

¿El generador usa el resultado? Poco. La primera versión añadía el resultado sin fecha y, en el
eval set, en Chaira la herramienta calculó bien y la respuesta repitió «entre dos y tres meses». Le
pusimos la fecha de la consulta («2026-10-14 · cálculo de Offside»), para que la regla del prompt base
«si dos noticias se contradicen, manda la más reciente» jugara a su favor. Medido con los casos
ficticios: el generador menciona el cálculo en 0 de 4 respuestas sin fecha y en 1 de 4 con fecha
(Ferro), y en ninguno cambia la señal cuando el plazo ya se cumplió (Lunar sigue «de baja» el día que ya
debería estar disponible). En Chaira lo sigue ignorando: la información correcta llega y el 1.5B no la usa, el mismo problema
de generación de la sección 9.

## 9 · Resultados

### La tabla evolutiva

Mismo eval set (27 preguntas), mismo harness, mismo generador y mismo prompt base. La latencia es la
mediana por consulta, de punta a punta, en una GTX 1650 Ti de 4 GB; en una T4 es menor, lo que cuenta
es la comparación entre filas. D2 es el juez de M2 tal cual; D2·v2 es el juez revisado tras el
feedback (ver más abajo).

| Sistema | D1 · F1 macro | D2 · juez v1 | D2·v2 | D3 · tasa accionable | Latencia mediana | Calla sin respuesta | Calla teniendo respuesta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Línea base M2 · LoRA de M1 (sin RAG) | 0,060 | 2,52 | 2,59 | 0,15 | < 0,1 s | — | — |
| Línea base M2 · léxico (sin RAG) | 0,110 | 2,52 | 3,37 | 0,19 | < 0,1 s | — | — |
| Mismo LLM, sin recuperación | 0,026 | 2,59 | 2,70 | 0,11 | 9,3 s | 3/4 | 19/23 |
| RAG ingenuo (sesión 7) | 0,139 | 2,59 | 3,96 | 0,37 | 15,1 s | 0/4 | 0/23 |
| RAG avanzado (híbrida + reranker + válvula verificada) | 0,289 | 2,93 | 3,93 | 0,52 | 22,1 s | 3/4 | 4/23 |
| Sistema final (+ herramienta) | 0,289 | 2,93 | 3,93 | 0,52 | 22,2 s | 3/4 | 4/23 |

Las líneas base de M2 son clasificadores de fragmentos: aquí reciben la pregunta como si fuera una
noticia y no tienen acceso al corpus.

Ablación, una técnica a la vez ([`scorecard_m3.csv`](../rag/resultados/scorecard_m3.csv)):

| Sistema | D1 | D2 · juez v1 | D3 | Latencia | Calla sin respuesta | Calla teniendo respuesta |
|---|---:|---:|---:|---:|---:|---:|
| RAG ingenuo | 0,139 | 2,59 | 0,37 | 15,1 s | 0/4 | 0/23 |
| + solo híbrida | 0,266 | 2,67 | 0,41 | 15,5 s | 0/4 | 0/23 |
| + solo reranker | 0,255 | 2,70 | 0,52 | 16,3 s | 1/4 | 0/23 |
| + solo válvula verificada | 0,266 | 2,89 | 0,48 | 21,6 s | 3/4 | 3/23 |
| + solo HyDE (descartada) | 0,185 | 2,48 | 0,33 | 26,1 s | 0/4 | 0/23 |
| Avanzado sin híbrida | 0,294 | 2,93 | 0,52 | 22,6 s | 3/4 | 4/23 |
| Avanzado sin reranker | 0,226 | 2,93 | 0,48 | 18,7 s | 3/4 | 3/23 |
| Avanzado sin válvula verificada | 0,194 | 2,63 | 0,48 | 13,5 s | 0/4 | 0/23 |
| Avanzado completo | 0,289 | 2,93 | 0,52 | 22,1 s | 3/4 | 4/23 |

### RAGAS

Las cuatro métricas están implementadas en [`ragas_local.py`](../rag/ragas_local.py) con las
definiciones de RAGAS, y el juez es `Qwen/Qwen2.5-1.5B-Instruct` leyendo logits de sí/no. No usamos la
librería `ragas` (no la probamos): le pide al juez salidas JSON en cada paso, y nuestro 1.5B, como
generador, devolvía JSON incompletos con frecuencia. Leyendo logits, cada veredicto es una pasada
determinista sin texto que parsear.

Se calcula sobre las 23 preguntas con respuesta en el corpus. Cada caso lleva la pregunta, la
respuesta, los chunks que vio el generador y la referencia escrita a mano. La faithfulness se calcula
sobre las respuestas que no son la válvula (`n`), porque «no encuentro esa información» no tiene
afirmaciones que verificar. `context precision (ids)` es la misma métrica calculada con los documentos
relevantes anotados a mano, y sirve para auditar al juez. Siguiendo la sesión 10, la calculamos también
para cada técnica sola: el reranker debería mover la precision y la híbrida el recall.

| Sistema | Faithfulness | Answer relevancy | Context precision | Context recall | Context precision (ids) |
|---|---:|---:|---:|---:|---:|
| RAG ingenuo | 0,80 (n=23) | 0,86 | 0,83 | 0,88 | 0,87 |
| + solo híbrida | 0,85 (n=23) | 0,86 | 0,86 | 0,89 | 0,91 |
| + solo reranker | 0,89 (n=23) | 0,86 | 0,84 | 0,88 | 0,95 |
| + solo válvula verificada | 0,85 (n=20) | 0,74 | 0,82 | 0,74 | 0,79 |
| RAG avanzado | 0,88 (n=19) | 0,71 | 0,83 | 0,74 | 0,83 |
| Sistema final | 0,88 (n=19) | 0,71 | 0,83 | 0,75 | 0,83 |

**Sanity check del juez RAGAS** (8 afirmaciones de verdad conocida sobre clubes ficticios): **8/8**.

### El juez, revisado tras el feedback de M2

El feedback de M2 señaló tres cosas: el juez casi no separaba respuestas buenas de malas; el
`criterio` que recibe casi le dice la etiqueta correcta, así que es un juez con referencia; y
`render_senal()` solo neutraliza la verbosidad mientras todo son etiquetas, pero el RAG escribe texto.
Lo atendimos así, sin tocar la columna D2, que sigue siendo el juez de M2 para poder comparar con esa
entrega:

1. Un juez v2 ([`judge_razonado.py`](../eval/judge_razonado.py),
   [`judge_rubric_v2.yaml`](../eval/judge_rubric_v2.yaml)) que puede escribir un análisis antes
   de que se lean los logits del puntaje, y que ve la respuesta en texto además de la señal.
2. El sanity check de M2 repetido dos veces: con el criterio completo y con un criterio «ciego» que
   describe qué mirar pero no nombra la categoría ni el impacto. Si el juez solo acierta con el
   completo, depende de la etiqueta que le damos.
3. Una plantilla nueva, `render_senal_y_texto()` en el harness, que añade la prosa recortada a 240
   caracteres, y una prueba de verbosidad con texto
   ([`evaluar_juez_v2.py`](../rag/evaluar_juez_v2.py)).

Sanity check ([`sanity_juez_v2.json`](../eval/sanity_juez_v2.json)), cinco señales de calidad
conocida sobre el mismo caso. Esperado: 5, 4, 3, 3, 1.

| Juez | Criterio completo | Criterio ciego | Signo invertido (esperado 1) |
|---|---|---|---|
| v1 de M2: Qwen2.5-1.5B, sin razonar | 4/5 en esta corrida (3/5 en M2, en CPU) | — | 3: no lo ve |
| v2 · 1.5B razonando | 1/5 | 1/5 | 2 con el completo; con el ciego pone 1 a todo |
| v2 · 1.5B sin razonar | 2/5 | 2/5 | 1, pero pone 1 a todo lo que no es perfecto |
| v2 · 3B razonando | 2/5 | 1/5 | 1 con el completo; 4 con el ciego |
| v2 · 3B sin razonar | 2/5 | 0/5 | 1 con el completo; 4 con el ciego |

En exactitud el que más acierta sigue siendo el v1. Razonar no mejoró al 1.5B: en sus análisis llega
a citar un «texto» que la señal no trae. Con el
criterio completo todas las variantes v2 ven el signo invertido, que la v1 no veía. Con el criterio
ciego ninguna lo ve. El profesor tenía razón: lo que detecta la inversión es la etiqueta que le damos
en el criterio, no el juez.

Aplicamos a las seis filas principales el 3B razonando (cuantizado a 4 bits para que quepa en 4 GB),
que es lo que se sugirió y la única variante que gradúa en vez de dar 5 o 1 (columna D2·v2,
[`scorecard_m3_juez_v2.csv`](../rag/resultados/scorecard_m3_juez_v2.csv)):

- Separa mucho más que la v1: 4,37 cuando la categoría es correcta frente a 3,03 cuando no (delta
  +1,34; la v1, +0,80), y solo en 1 de 27 preguntas le da la misma nota a una respuesta correcta y a
  una incorrecta (la v1, en 3).
- Pero al ver la prosa se deja llevar por lo que suena bien. Al ingenuo le pone un 5 cuando presenta el
  rumor de Erik Lira como baja confirmada, que es el error más caro de la rúbrica, y otro 5 cuando
  responde el Balón de Oro de memoria. Y castiga más la válvula (1 o 2) que la v1 (3). Por eso deja al
  ingenuo (3,96) a la par del avanzado (3,93), cuando todo lo demás dice que el avanzado es mejor.
- Es inestable con las señales vacías. La rúbrica dice que no emitir señal vale 3 siempre; en las 21
  preguntas con respuesta en las que el léxico no da señal, el v2 le pone desde 1 hasta 5 (media 3,38).
  De ahí sale el 3,37 del léxico, por encima del LLM sin recuperación.
- Verbosidad: con el recorte de 240 caracteres, añadir una frase de relleno no cambia la media (4,10
  en los dos casos), pero sí la nota de 11 de las 20 respuestas, en las dos direcciones. Sin recorte,
  la media sube a 4,35. El recorte quita el sesgo en promedio, no el ruido caso a caso
  ([`sesgo_verbosidad_texto_m3.csv`](../rag/resultados/sesgo_verbosidad_texto_m3.csv)).

Conclusión: el juez v2 separa mejor, pero premia la prosa plausible y castiga callar, que es lo
contrario de lo que pide la rúbrica. No lo adoptamos para tomar decisiones. Lo reportamos como
columna, y las conclusiones de este módulo se apoyan en la D1, la D3 y RAGAS. La dimensión 3 sigue
siendo la que cuenta el signo invertido y el rumor presentado como hecho.

### La lectura: qué movió qué, cuánto costó, qué falla queda

Antes de leer decimales: son 27 preguntas, así que un acierto vale 0,037 de accuracy, y en el F1 macro
una sola pregunta puede valer una clase entera (Erik Lira es el único caso de rumor). Un generador de
1.5B además cambia de respuesta cuando cambia el orden de los distractores. Por eso lo que sigue se
apoya en los casos y no solo en las medias. Y una categoría correcta no garantiza una respuesta
correcta: el harness mide la señal y RAGAS mide la prosa.

**1. ¿Qué movió qué?**

- Recuperar es lo que más mueve, y el avanzado mejora al ingenuo en las tres dimensiones del harness:
  F1 de 0,139 a 0,289, juez v1 de 2,59 a 2,93 y tasa accionable de 0,37 a 0,52. Sin noticias, el
  mismo LLM calla en 19 de las 23 preguntas que tienen respuesta. Las líneas base de M2 reciben la
  pregunta como si fuera una noticia y no saben nada de lo que pasó esta semana.
- El reranker es la técnica que más sube la tasa accionable por sí sola (0,37 → 0,52), y quitarlo del
  avanzado baja el F1 a 0,226 y la tasa a 0,48. En RAGAS mueve lo que la sesión 10 dice que debe
  mover: la context precision calculada con los documentos anotados pasa de 0,87 a 0,95, y la
  faithfulness de 0,80 a 0,89 porque llegan menos distractores. Gana la categoría en Musiala (pone
  delante la noticia de octubre), Ochoa, Svilar y Balón de Oro. Dos de esos aciertos lo son solo de
  categoría: en Musiala responde con otra noticia del Bayern y en Svilar el documento no llega.
- La búsqueda híbrida sola sube el F1 de 0,139 a 0,266, pero casi todo viene de una pregunta, Erik
  Lira, el único rumor, que vale una clase entera del F1 macro (además gana Mbappé y pierde
  Bentancur). En RAGAS sube algo la precision (0,83 → 0,86) y apenas el recall (0,88 → 0,89, una
  afirmación). Dentro del avanzado, quitarla no cambia nada (0,294 sin ella frente a 0,289; D2 y D3
  iguales). Su efecto en este eval set está dentro del ruido. La mantenemos porque es la segunda
  técnica de la lista y cuesta milisegundos.
- La válvula verificada es la pieza que más pesa en el avanzado. Quitarla baja el F1 de 0,289 a 0,194
  y el juez v1 de 2,93 a 2,63, y deja la válvula en 0 de 4. A cambio calla en 4 de las 23 preguntas
  que sí tenían respuesta (punto 3), y por eso el context recall del avanzado baja a 0,74.
- HyDE empeora todo (F1 0,185, juez v1 2,48) y cuesta 11 s más. La descartamos.
- El agente no mejora el harness: las tres dimensiones del avanzado y del final son iguales. La
  herramienta calcula bien el regreso de Chaira y RAGAS lo registra (el context recall de esa pregunta
  pasa de 0,75 a 1,0), pero el generador ignora el resultado y repite «entre dos y tres meses». Las
  otras dos preguntas que la necesitaban (las de Ochoa) no llegan a la herramienta porque la válvula
  las corta antes. La mantenemos porque solo se dispara cuando hace falta (0 llamadas innecesarias en
  24 preguntas), el cálculo queda registrado y es lo que mostrará la interfaz del M4. Si la
  quitáramos, el harness no cambiaría.

**2. ¿Cuánto costó?**

| Paso | Latencia mediana | Qué cuesta |
|---|---:|---|
| RAG ingenuo | 15,1 s | casi todo es generar el JSON con un 1.5B en una GPU de 4 GB |
| + reranker | +1,2 s | 20 pares por el cross-encoder, en CPU |
| + híbrida | +0,3 s | BM25 tarda milisegundos; el resto es variación de la generación |
| + válvula verificada | +6,5 s | 4 pasadas del modelo, una por chunk. Cuando calla, la consulta termina en 9,1 s sin generar |
| + herramienta | +0,2 s de mediana | solo se dispara en 1 de 27 consultas, pero esa pasa de 22 s a 36 s |
| HyDE (descartada) | +11,0 s | una generación más antes de buscar |

El avanzado tarda un 46 % más que el ingenuo (22,1 s frente a 15,1 s), casi todo por la válvula. Para
nuestro usuario creemos que compensa: el apostador consulta antes del partido, no en directo, y lo que
se paga con esa latencia es no recibir una baja inventada. HyDE no compensa. Para la versión eficiente
del M5 hay margen: las cuatro verificaciones caben en una sola pasada y el JSON de salida puede ser más
corto.

**3. ¿Qué falla queda y de qué tipo?** Cruzamos el harness con RAGAS pregunta por pregunta en las 23
con respuesta del sistema final ([`ragas_m3_por_caso.csv`](../rag/resultados/ragas_m3_por_caso.csv)):

| Tipo | Casos | Qué se ve en RAGAS |
|---|---:|---|
| Categoría correcta | 9 | De esas, el impacto solo es correcto en 2: el modelo pone «negativo_bajo» a casi todo. La dimensión 3 no castiga quedarse corto; el juez v1 sí (nivel 4) |
| Generación: etiqueta equivocada con el contexto correcto | 8 | context recall y faithfulness altos, la categoría mal: Vinicius en la Roma (alineación → «regreso»), Isaac Romero y Ángel Pérez (sanción → «baja»), Nico Williams (baja → «duda»), Laporte y Beitia (duda → «baja»), Arrasate (su «vuelta» leída como regreso) y Chaira (ignora la herramienta) |
| La válvula calla de más | 4 | sin contexto: Erik Lira (el verificador confunde «la respuesta es que no» con «no es la fuente»), los dos de Ochoa (el resumen RSS cruzado Málaga/Girona) y Svilar (el documento no llegó) |
| Generación: inventa en la prosa | 1 | Sadiq: «ha vuelto a estar bien» cuando la noticia dice que espera pruebas (faithfulness 0,33) |
| Error del juez de RAGAS | 1 | Celta: la respuesta es correcta y el juez no la reconoce (context recall 0) |

Hay además un error de prosa que no detectan ni el harness ni RAGAS: en Nico Williams la respuesta
mezcla su baja con la indisposición de Laporte.

La falla que queda es de generación, no de recuperación. El recuperador trae el documento correcto al
top-4 en 22 de 23 preguntas; después de la válvula le llega contexto al generador en 19, y la
faithfulness de lo que responde es 0,88. Aun así etiqueta mal 8. Etiquetar es lo que hace el
clasificador de M1, así que el siguiente paso natural es separar las dos tareas: el RAG recupera y
redacta, y un clasificador afinado etiqueta el chunk recuperado. El context recall baja del ingenuo
(0,88) al final (0,75). De esos 3 puntos sobre 23, 2 vienen de la válvula (Erik Lira y los dos de Ochoa
se quedan sin contexto) y 1 del juez en Celta y de un filtrado parcial en Vinicius-Roma.

**4. ¿Qué pasó con las preguntas sin respuesta en el corpus?** El RAG ingenuo inventó en las cuatro: a
«¿Mbappé se pierde el Clásico?» respondió con la sanción de Vinicius, dijo que Lamine «estuvo hasta
hace poco lesionado», respondió el Balón de Oro de memoria («el inglés fue el último…») y al
Betis-Oporto le atribuyó «un impacto positivo». El sistema final activa la válvula en 3 de 4 (Mbappé,
cuota y Balón de Oro). Falla en Lamine Yamal: el verificador deja pasar «Pues el día que Lamine esté
fino» (P = 0,999) y el generador inventa «Lamine está cerca de estar disponible». El LLM sin
recuperación también calla en 3 de 4, pero porque calla en casi todo (19 de 23 con respuesta).

**Qué aprendimos.** Con un generador de 1.5B no encontramos un prompt que supiera a la vez responder y
callar, y el modelo tampoco supo decidir cuándo usar la herramienta (0 de 4). Las dos decisiones las
sacamos del generador: la primera la toma un clasificador sí/no por logits y la segunda una regla. El
problema del feed (resúmenes cruzados entre noticias) que vimos al construir el corpus terminó
costando dos preguntas del eval set.

**Un error nuestro.** La primera corrida completa usó por descuido la versión v4 del prompt base en
lugar de la v3 que habíamos elegido con el dev set. Lo detectamos en una revisión antes de entregar y
repetimos todo con v3. Las conclusiones de fondo no cambiaron, pero las cifras sí: con v4 el avanzado
parecía peor que el ingenuo en F1. Los números de este README son los de v3.

## 10 · Reproducir

Todo corre en Colab gratis (T4) sin claves de pago; los modelos son abiertos y se ejecutan en local. El notebook
[`M3_RAG_agentico.ipynb`](../notebooks/M3_RAG_agentico.ipynb) clona el repo, instala
[`requirements-m3.txt`](../../requirements-m3.txt) y reconstruye el índice; con `REPRODUCIR = True`
vuelve a correr todo lo de abajo. Desde una terminal:

```bash
pip install -r requirements-m3.txt
cd proyecto1/rag
python indice.py --forzar              # ingest -> chunk -> embed -> store
python diagnostico_recuperacion.py     # embeddings_m3.csv, recuperacion_m3.csv (--con-llm: + HyDE/multi-query)
python ajuste_prompt.py                # versiones del prompt base sobre el dev set
python calibrar_umbral.py              # candidatos a válvula sobre el dev set
python prueba_planificador.py          # el agente con noticias ficticias
python correr_sistemas.py              # fase 1: todos los sistemas responden el eval set
python evaluar_m3.py                   # fase 2: harness de M2 + RAGAS -> scorecard_m3.csv, ragas_m3.csv
python ../eval/judge_razonado.py --modelo Qwen/Qwen2.5-3B-Instruct   # sanity check del juez v2
python evaluar_juez_v2.py --modelo Qwen/Qwen2.5-3B-Instruct          # juez v2 y sesgo de verbosidad con texto
python test_parseo.py                  # parseo tolerante y herramienta (también corre en el CI)
```

Semilla 42, decodificación greedy, versiones exactas de librerías y commit fijo de cada modelo de
Hugging Face (`rag/config.py`). Los resultados versionados se produjeron en una GTX 1650 Ti (4 GB,
fp16) con Python 3.12; en otra GPU las latencias cambian y, por la aritmética de punto flotante de cada
GPU, alguna respuesta greedy puede cambiar en un token. Nos pasó: entre la corrida de diagnóstico del RAG
ingenuo y la definitiva, con el mismo prompt y los mismos pesos, cambió 1 de 27 respuestas (M3-22); lo atribuimos a la aritmética fp16 de
los kernels de la GPU y no lo investigamos más. Las respuestas exactas de cada sistema están en
[`rag/resultados/respuestas/`](../rag/resultados/respuestas/), así que el harness y RAGAS se pueden
recalcular sobre ellas sin volver a generar.

## 11 · Limitaciones

- 27 preguntas: un acierto son 0,037 de accuracy; las conclusiones se apoyan en los casos, no en
  la segunda cifra decimal.
- Escribimos el eval set conociendo el corpus, lo que explica en parte que la recuperación sea tan
  fácil (casi todas las preguntas nombran al jugador). Un usuario real preguntará peor.
- El juez de RAGAS es el mismo modelo base que genera. No sabe qué sistema respondió, pero el
  riesgo de autopreferencia existe. Su sanity check es 8/8 y su context precision coincide con la
  calculada con documentos anotados a mano, y aun así comete errores: en Celta no reconoce que
  «Pablo Durán está lesionado» se deduce de «una convocatoria sin Pablo Durán ni Borja Iglesias
  lesionados». Antes descomponía las respuestas mezclando la pregunta (de «Isaac Romero fue
  expulsado» sacaba «juega con el Atlético Madrid»); eso lo corregimos quitándole la pregunta.
- El dev set es pequeño (14 preguntas, 3 sin respuesta) y en DEV-14 el verificador «acierta» abriendo
  sobre un documento que no es el de Lizancos.
- Es un solo corpus de prensa, casi todo español, con poca presencia latinoamericana (36 de 748 fragmentos de El
  Tiempo), y resúmenes RSS que a veces pertenecen a otra noticia.
- Las latencias son de una GPU de portátil: la comparación entre filas vale; los segundos absolutos no.
