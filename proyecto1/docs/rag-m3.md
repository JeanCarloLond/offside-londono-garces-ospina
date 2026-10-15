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

El corpus del RAG **no** reemplaza a `data/processed/weak_labeled.jsonl`: ese es el train de M1/M2 y
lo dejamos intacto para que sus números sigan siendo reproducibles.

**Un problema de datos que encontramos aquí.** En algunos ítems el feed de Marca pega el resumen de
otra noticia. El de Aaron Ochoa tiene la URL y el titular del Málaga, pero el resumen habla del
técnico del Girona; el mismo resumen del portero de Las Palmas aparece bajo cuatro titulares distintos.
El eval set de M2 heredó el primero (OFF-12 está etiquetado como «Girona»). No lo limpiamos a mano:
es ruido real del corpus y en la sección de resultados se ve dónde muerde.

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
versiona (`rag/_chroma/` está en `.gitignore`), precisamente para que quien clone el repo compruebe
que el pipeline corre de principio a fin.

Lo que se indexa es la sección de la URL más el texto: en Marca la sección es el club, y en muchos
fragmentos el club **solo** está ahí («Quique abre el casting para la banda derecha» no dice que es el
Alavés). Lo que ve el generador lleva además la fecha y el medio delante, porque «estará de baja tres
meses» no significa nada sin saber cuándo se publicó.

## 3 · Las tres decisiones

**Chunks: un fragmento RSS por chunk, sin solapamiento.** Nuestros fragmentos ya son la unidad mínima
con sentido —titular más entradilla, mediana de 162 caracteres— y en ellos el sujeto y el hecho suelen
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

El modelo de paráfrasis se hunde porque nuestras preguntas y los titulares no son paráfrasis: «¿Cuenta
el Sevilla con Guridi para jugar en el Bernabéu?» frente a «El Sevilla prepara la visita al Bernabéu
sin Guridi». e5-base gana en recall@1 y en MRR (0,94 frente a 0,90), justo lo que el reranker
iguala (0,96 con e5-small), con más del doble de parámetros
—y Colab gratis tiene que cargar también el generador y el juez—.

**top-k = 4.** Cada pregunta se responde con una o dos noticias. Con k = 4 el documento correcto llega
en 22 de 23 preguntas, y subir a k = 10 no rescata la que falta (llega en la posición 11). Más chunks
no traen información nueva y sí distractores, que es justo lo que confunde a un generador de 1.5B (lo
vimos en el diagnóstico: Bentancur, Celta).

## 4 · El eval set

[`eval/eval_set_m3.json`](../eval/eval_set_m3.json), generado y validado por
[`eval/build_eval_set_m3.py`](../eval/build_eval_set_m3.py). Nuestra tarea de M1 era de clasificación,
así que escribimos preguntas nuevas sobre el corpus, **sin cambiar la vara**: cada una tiene la misma
respuesta estructurada que el harness de M2 ya sabe medir (categoría del hecho, impacto, equipo).

| | |
|---|---|
| Casos | **27**, con `input`, `esperado` y `criterio` (más `referencia` y `referencia_afirmaciones` para RAGAS y `docs_relevantes` para diagnosticar la recuperación) |
| Fuera del corpus | **4** (Mbappé y el Clásico, lesión de Lamine Yamal, cuota del Betis-Oporto, Balón de Oro 2025): lo esperado es la válvula, que en el lenguaje del harness es «sin señal» (`irrelevante · neutro · ninguno`) |
| Necesitan calcular fechas | 3 (Aaron Ochoa el 1 de noviembre y el 15 de diciembre, Chaira el 29 de noviembre) |
| Adversariales | 17: conflictos temporales (Musiala, Vinicius), pares mínimos (Vinicius en Liga frente a Champions; Ochoa antes y después de su regreso estimado), negación (Beitia), nombre parcial (Nico), club solo en la URL (Ángel Pérez), titular metafórico (Svilar), pregunta que presupone el hecho (Erik Lira) |
| Categorías | las 8 |
| Fecha de la consulta | fija, 2026-10-14 (`config.FECHA_CONSULTA`), para que la tabla salga igual otro día |

Los casi-aciertos fuera del corpus son a propósito: «¿Kylian Mbappé se pierde el Clásico?» comparte
literalmente «se pierde el Clásico» con el titular de Vinicius.

**Dev set aparte.** [`eval/dev_set_m3.json`](../eval/dev_set_m3.json) tiene 14 preguntas sobre otras
noticias (3 sin respuesta). Es donde se ajustaron el prompt base y el verificador. Ningún prompt ni
umbral se ajustó mirando el eval set.

## 5 · El RAG ingenuo y su válvula

`sistema_rag(pregunta)` ([`rag/sistemas.py`](../rag/sistemas.py)): top-4 por similitud densa, prompt
aumentado, `Qwen/Qwen2.5-1.5B-Instruct` con decodificación greedy. El modelo devuelve un JSON con la
respuesta en prosa y la señal; el parseo es tolerante (bloques ```json, comillas simples, comas
colgando, etiquetas mal escritas) y, si no hay JSON recuperable, la señal cae a «sin señal».

**El prompt base, versión por versión** (sobre el dev set,
[`ajuste_prompt_m3.csv`](../rag/resultados/ajuste_prompt_m3.csv)). Es el mismo para todas las filas de
la tabla, así que había que fijarlo antes de comparar nada:

| Versión | Qué cambia | Categoría correcta | Calla teniendo respuesta | Calla sin respuesta | Qué pasa |
|---|---|---:|---:|---:|---|
| v1 | La válvula como regla: «si no está, responde exactamente "No encuentro…"» | 8/14 | 5/11 | 3/3 | Copia la frase de la válvula aun con la noticia correcta en la posición 1 |
| v2 | + dos ejemplos resueltos con clubes ficticios | 3/14 | 8/11 | 3/3 | Imita el ejemplo («Lizancos está *sancionado*») en vez de leer |
| **v3** | Sin ejemplos; primer campo `noticia` (el número de la que responde, o 0) | **9/14** | **0/11** | 0/3 | Contesta todo… y nunca calla: a Pedri le atribuye la lesión de Guridi |
| v4 | + «comprueba que la noticia nombra al jugador» | 8/14 | 3/11 | 0/3 | Ni calla cuando debe ni deja de callar cuando no |

En el eval set la v1 confirma el diagnóstico: calla en 4 de 4 sin respuesta, pero también en 12 de las
23 que sí la tienen (fila `rag_ingenuo_prompt_v1` de [`scorecard_m3.csv`](../rag/resultados/scorecard_m3.csv)).

**Con un generador de 1.5B, la válvula por prompt es un dial, no un interruptor.** Congelamos v3,
que es la que mejor responde, para todas las versiones. El ingenuo queda con su válvula de prompt
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

**La recuperación casi nunca es el cuello de botella en nuestro corpus.** Los fragmentos son cortos y
distintos entre sí, y nuestras preguntas casi siempre nombran al jugador. Solo M3-11 (el portero de la
Roma) no llega: el titular es metafórico («El rapero con guantes que quiere silenciar al Madrid») y ni
dense (posición 11), ni BM25 (23), ni HyDE (8), ni multi-query (15) lo traen al top-4.

Luego, con el generador: los fallos del ingenuo sobre el eval set, agrupados por la pregunta del
enunciado —¿no llegó el chunk, llegó y el modelo lo ignoró, o el dato no está?—:

| Tipo de falla | Casos | ¿Llegó el chunk? |
|---|---|---|
| Inventa en preguntas sin respuesta (válvula 0/4) | Mbappé ← le traslada la sanción de Vinicius; Lamine; Balón de Oro; cuota | no hay chunk que llegar |
| Un distractor va delante del relevante | Musiala (relevante en posición 2, delante noticias de agosto), Celta (posición 2: la respuesta habla de otra noticia), Bentancur (mezcla «expulsado a los 3 minutos» de la Bundesliga) | sí, pero no primero |
| No llega | Svilar → inventa «Bernardo Silva» a partir de otro chunk | no |
| Hay que calcular una fecha | Ochoa el 15 de diciembre (responde «0»), Chaira el 29 de noviembre | sí |
| La etiqueta está mal con el chunk correcto en la posición 1 | Vinicius, Isaac Romero y Ángel Pérez (sanción → «baja»), Laporte, Beitia, Sadiq, Arrasate | sí — es generación |

## 7 · Técnicas: qué falla ataca cada una

| Técnica | Falla que ataca | Evidencia del diagnóstico |
|---|---|---|
| **Reranking** con `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` sobre un pool de 20 | Un distractor va delante del relevante | Musiala, Celta y Ochoa pasan de la posición 2 a la 1; MRR 0,90 → 0,96 |
| **Búsqueda híbrida** BM25 + dense con RRF (k = 60) | Nombres propios raros y términos exactos | BM25 pone primero el relevante más a menudo que dense (recall@1 0,91 frente a 0,83); con el reranker encima, su aporte medido en este eval set es **nulo** (lo decimos en la lectura) |
| **Válvula verificada**: el mismo Qwen como clasificador sí/no por logits, chunk a chunk | Inventa en preguntas sin respuesta | El ingenuo activó la válvula 0 de 4 veces |
| *HyDE y multi-query* | *El chunk no llega* | **Descartadas**: no rescatan M3-11 y empeoran el resto (MRR 0,74), además de añadir una llamada al LLM |

**Por qué un verificador y no un umbral del reranker.** Probamos primero lo obvio: si el mejor chunk
puntúa bajo en el cross-encoder, callar. No separa ([`valvula_dev_m3.csv`](../rag/resultados/valvula_dev_m3.csv)):
el cross-encoder (entrenado en mMARCO, pregunta → pasaje que contiene la respuesta) puntúa bajo las
preguntas de disponibilidad que exigen inferir —«¿Puede jugar Sergio Álvarez?» frente a «sufre una
rotura muscular»—. El mismo Qwen, en cambio, usado como clasificador de una sola decisión («¿es esta
noticia la fuente para responder?») y leyendo los logits de sí/no, acierta 13 de 14 en el dev set con
el umbral natural de 0,5. El que falla es Julián Alvarez, y es discutible: la noticia sí habla de él y
del Atlético. La primera versión del verificador rechazaba las preguntas de fechas (la noticia no
menciona la fecha por la que se pregunta) y la corregimos, también sobre el dev set, preguntando si es
«la fuente» aunque haya que deducir la respuesta.

Además de callar, el verificador **filtra**: solo pasan al generador los chunks que aprueba.

## 8 · La capa agéntica

**La herramienta:** `calcular_fecha(fecha_inicio, cantidad, unidad, fecha_objetivo)`
([`rag/herramientas.py`](../rag/herramientas.py)). Las bajas llegan como duraciones relativas a la
fecha de la noticia y el apostador pregunta por un día concreto. Esa aritmética de calendario un 1.5B
la hace mal; una función pura no se equivoca. El resultado incluye una «lectura» en español («el
2026-11-01 es 26 días ANTES del fin del plazo: para esa fecha el jugador TODAVÍA estará de baja»).

**Esquema, parseo tolerante y tope de pasos.** El esquema (nombre, descripción y parámetros, en formato
de *function calling*) va en el prompt. Lo que propone el modelo se parsea con tolerancia: etiquetas
`<tool_call>` o JSON desnudo, `herramienta`/`argumentos` o `name`/`arguments`, comillas simples,
argumentos como string JSON, nombres aproximados de la herramienta, fechas en español («27 de agosto
de 2026»), cantidades en palabras («tres») y rangos («2-3»: se usa el máximo, la lectura prudente).
Si los argumentos no son válidos, el error vuelve al modelo para que corrija; **como máximo 3 pasos**.
Las pruebas están en [`rag/test_parseo.py`](../rag/test_parseo.py) y corren en el CI.

**Cuándo se usa, y por qué el cuándo no lo decide el LLM.** Lo intentamos tres veces:

1. Con la plantilla nativa de tools de Qwen, el sistema recibe un bloque en inglés y el modelo
   respondía «0» (ninguna noticia) incluso a preguntas que sin herramienta contestaba bien.
2. Con la herramienta descrita dentro de nuestro prompt, no la llamó nunca: la orden «devuelve solo el
   JSON final» gana siempre.
3. Con un paso de planificación aparte, en el que el LLM decide si llamar: **0 de 4** llamadas cuando
   hacía falta (prueba con noticias ficticias, [`prueba_planificador_m3.csv`](../rag/resultados/prueba_planificador_m3.csv)).
   Leyendo esa decisión por logits tampoco separa: da 0,93 a «¿Qué lesión tiene Lunar?», que no
   necesita ningún cálculo. (Los intentos 1 y 2 y esta lectura por logits se midieron durante el
   desarrollo y no están versionados como CSV; el intento 3 sí, en el modo `llm` de la prueba.)

Este modelo no sabe decidir **cuándo** usar una herramienta. Así que el cuándo lo decide una regla
verificable —la pregunta nombra una fecha concreta **y** algún chunk trae una duración con número— y el
**qué** lo propone el LLM: la fecha de la noticia, la cantidad, la unidad y la fecha objetivo. Con ese
reparto, en la misma prueba: **4 de 4** llamadas cuando hace falta y **0 de 4** cuando no. Una de las
cuatro sale con el año equivocado («15 de marzo» → 2026-03-15, anterior a la noticia): la herramienta
lo rechaza, el error vuelve al modelo, el modelo repite el mismo error y **el tope de pasos corta el
bucle al tercer intento**. Es exactamente el caso para el que existe el tope.

La respuesta final sale del mismo prompt base del RAG avanzado, con el resultado de la herramienta
añadido como una entrada más del bloque de noticias. Entre el avanzado y el sistema final lo único que
cambia es esa entrada.

**¿El generador usa el resultado?** Poco. La primera versión añadía el resultado sin fecha y, en el
eval set, en Chaira la herramienta calculó bien y la respuesta repitió «entre dos y tres meses». Le
pusimos la fecha de la consulta («2026-10-14 · cálculo de Offside»), para que la regla del prompt base
«si dos noticias se contradicen, manda la más reciente» jugara a su favor. Medido con los casos
ficticios: el generador menciona el cálculo en **0 de 4** respuestas sin fecha y en **2 de 4** con
fecha, y en ninguno cambia la señal cuando el plazo ya se cumplió (Lunar sigue «de baja» el día que ya
debería estar disponible). En Chaira lo sigue ignorando. Es la misma falla de generación de la sección
9: la información correcta llega y el 1.5B no la usa.

## 9 · Resultados

### La tabla evolutiva

Mismo eval set (27 preguntas), mismo harness, mismo generador y mismo prompt base. Latencia mediana
por consulta, de punta a punta, en una GTX 1650 Ti de 4 GB (en la T4 de Colab es menor; lo que importa
es la comparación entre filas).

| Sistema | D1 · F1 macro | D2 · juez (1-5) | D3 · tasa accionable | Latencia mediana | Válvula (calla sin respuesta) | Calla teniendo respuesta |
|---|---:|---:|---:|---:|---:|---:|
| Línea base M2 · LoRA de M1 (sin RAG) | 0,060 | 2,52 | 0,15 | < 0,1 s | — | — |
| Línea base M2 · léxico (sin RAG) | 0,110 | 2,52 | 0,19 | < 0,1 s | — | — |
| Mismo LLM, sin recuperación | 0,083 | 2,59 | 0,19 | 4,6 s | 4/4 | 21/23 |
| **RAG ingenuo** (sesión 7) | 0,272 | 2,74 | 0,48 | 15,3 s | 0/4 | 1/23 |
| **RAG avanzado** (híbrida + reranker + válvula verificada) | 0,252 | 2,93 | 0,52 | 21,8 s | 3/4 | 4/23 |
| **Sistema final** (+ herramienta) | 0,252 | 2,93 | 0,52 | 22,0 s | 3/4 | 4/23 |

Las líneas base de M2 son clasificadores de fragmentos: aquí reciben la pregunta como si fuera una
noticia, sin acceso al corpus. Están para mostrar que, sin recuperar, ningún sistema sabe nada de esta
semana.

**Ablación: una técnica a la vez** ([`scorecard_m3.csv`](../rag/resultados/scorecard_m3.csv))

| Sistema | D1 | D2 | D3 | Latencia | Válvula | Calla teniendo respuesta |
|---|---:|---:|---:|---:|---:|---:|
| RAG ingenuo | 0,272 | 2,74 | 0,48 | 15,3 s | 0/4 | 1/23 |
| + solo híbrida | 0,256 | 2,70 | 0,41 | 17,2 s | 0/4 | 1/23 |
| + solo reranker | 0,335 | 2,74 | 0,56 | 16,8 s | 1/4 | 1/23 |
| + solo válvula verificada | 0,229 | 2,96 | 0,48 | 21,8 s | 3/4 | 3/23 |
| + solo HyDE (descartada) | 0,167 | 2,44 | 0,33 | 25,4 s | 0/4 | 1/23 |
| Avanzado sin híbrida | 0,258 | 2,93 | 0,52 | 23,3 s | 3/4 | 4/23 |
| Avanzado sin reranker | 0,226 | 2,93 | 0,48 | 21,8 s | 3/4 | 3/23 |
| Avanzado sin válvula verificada | 0,335 | 2,78 | 0,56 | 16,2 s | 1/4 | 1/23 |
| **Avanzado completo** | 0,252 | 2,93 | 0,52 | 21,8 s | 3/4 | 4/23 |

### RAGAS

Las cuatro métricas, implementadas en [`ragas_local.py`](../rag/ragas_local.py) con las
definiciones de RAGAS y como juez **`Qwen/Qwen2.5-1.5B-Instruct`** leyendo logits sí/no. No usamos
la librería `ragas` porque le pide al juez salidas JSON estructuradas en cada paso, y nuestro propio
1.5B, como generador, devolvía JSON incompletos con frecuencia; leyendo logits cada veredicto es un
forward pass determinista sin nada que parsear. Sobre las 23 preguntas con respuesta en el corpus;
cada caso lleva pregunta, respuesta, los chunks que vio el generador y la referencia escrita a mano.
Faithfulness se calcula sobre las respuestas que no son la válvula (`n`): «no encuentro esa
información» no tiene afirmaciones que verificar. `context precision (ids)` es la misma métrica
calculada con los documentos relevantes anotados a mano: sirve para auditar al juez.

| Sistema | Faithfulness | Answer relevancy | Context precision | Context recall | Context precision (ids) |
|---|---:|---:|---:|---:|---:|
| RAG ingenuo | 0,78 (n=22) | 0,82 | 0,82 | 0,88 | 0,87 |
| RAG avanzado | 0,84 (n=19) | 0,71 | 0,83 | 0,74 | 0,83 |
| Sistema final | 0,89 (n=19) | 0,71 | 0,83 | 0,75 | 0,83 |

**Sanity check del juez RAGAS** (8 afirmaciones de verdad conocida sobre clubes ficticios): **8/8**.

### La lectura: qué movió qué, cuánto costó, qué falla queda

Una advertencia antes de leer decimales: son 27 preguntas, así que **un acierto vale 0,037 de
accuracy** y un generador de 1.5B cambia de opinión cuando cambia el orden de los distractores.
Diferencias de uno o dos casos están dentro de ese ruido; lo que sigue se apoya en los casos, no solo
en las medias.

**1 · ¿Qué movió qué?**

- **Recuperar es lo que más mueve.** Sin noticias, el mismo LLM calla en 21 de 23 preguntas que tienen
  respuesta (D3 0,19); con el RAG ingenuo, D3 sube a 0,48 y D1 de 0,08 a 0,27. Las líneas base de M2
  no saben nada de esta semana: reciben la pregunta como si fuera una noticia.
- **El reranker es la única técnica de recuperación que mueve el harness.** Solo, sube D1 de 0,272 a
  0,335 y D3 de 0,48 a 0,56; quitarlo del avanzado los baja a 0,226 y 0,48. Gana justo los casos del
  diagnóstico: Bentancur (ya no mezcla la expulsión de la Bundesliga), Musiala (pasa al frente la
  noticia de octubre) y Celta.
- **La búsqueda híbrida no mueve nada medible.** Con el reranker encima es idéntica con o sin ella
  (D2 y D3 iguales; D1 0,252 frente a 0,258). Sola resta un acierto neto, y los casos que cambian
  (pierde Ochoa y Betis, gana Mbappé) no tienen que ver con nombres propios: es el ruido del
  generador. La dejamos porque es la segunda técnica de la lista que exige el enunciado y cuesta
  milisegundos, pero **en este eval set no se ha ganado su sitio**, y lo decimos.
- **La válvula verificada no sube la métrica clásica: compra no inventar.** Pasa la válvula de 0/4 a
  3/4 y es lo que más sube al juez (D2 2,74 → 2,96, porque la rúbrica castiga más inventar que
  callar). A cambio calla en 3 de las 23 preguntas que sí tenían respuesta (el ingenuo, en 1), y una
  es Erik Lira, la única de rumor del eval set: el F1 macro pierde una clase entera (0,272 → 0,229). Quitarla del avanzado sube D1 y D3 y baja
  D2. **La dejamos**: para un apostador, que el sistema afirme «Mbappé se pierde el Clásico» es el
  error más caro posible, y es exactamente el que la válvula evita.
- **HyDE empeora todo** (D1 0,167, D2 2,44) y cuesta 10 s más. Descartada.
- **El agente no mejora el harness.** Las respuestas del avanzado y del final son idénticas. La
  herramienta funciona —en Chaira calcula bien el regreso y RAGAS lo ve: el context recall de ese caso
  sube de 0,75 a 1,0— pero el generador ignora el resultado y repite «entre dos y tres meses». Las otras
  dos preguntas que la necesitaban (Ochoa) no llegan a la herramienta porque la válvula las corta antes
  (ver punto 3). La dejamos porque solo se dispara cuando hace falta (0 llamadas innecesarias en 24
  preguntas, 0 de 4 en la prueba aislada), el cálculo queda registrado y es lo que mostrará la interfaz
  del M4. Si se quitara, el harness no cambiaría: lo decimos tal cual.

**2 · ¿Cuánto costó?**

| Paso | Latencia mediana | Qué cuesta |
|---|---:|---|
| RAG ingenuo | 15,3 s | casi todo es generar el JSON con un 1.5B en una GPU de 4 GB |
| + reranker | +1,5 s | 20 pares por el cross-encoder, en CPU |
| + híbrida | ≈ 0 | BM25 son milisegundos. Lo medido (+1,9 s sola, −1,5 s dentro del avanzado) es ruido de la generación |
| + válvula verificada | +6,5 s | 4 pasadas del modelo, una por chunk. Cuando calla, la consulta termina en 8,9 s sin generar |
| + herramienta | +0,2 s de mediana | solo se dispara en 1 de 27 consultas, pero esa pasa de 21 s a 35 s |
| HyDE (descartada) | +10 s | una generación extra antes de buscar |

El avanzado tarda un 42 % más que el ingenuo (21,8 s frente a 15,3 s). **Para nuestro usuario vale la
pena**: el apostador consulta antes del partido, no en directo, y lo que compra esa latencia es no
recibir una baja inventada. Lo que no vale la pena es HyDE. La versión eficiente del M5 tiene dónde
cortar: las cuatro verificaciones caben en una sola pasada y el JSON de salida puede ser más corto.

**3 · ¿Qué falla queda y de qué tipo?** Cruzando el harness con RAGAS, pregunta por pregunta, sobre las
23 con respuesta del sistema final ([`ragas_m3_por_caso.csv`](../rag/resultados/ragas_m3_por_caso.csv)):

| Tipo | Casos | Señal en RAGAS |
|---|---:|---|
| Bien | 8 | — |
| **Generación: etiqueta equivocada** con el contexto correcto y la respuesta fiel | 8 | context recall y faithfulness altos, la categoría mal. Sobre todo sanción → «baja» (Vinicius, Isaac Romero, Ángel Pérez), duda → «baja» (Laporte) y palabras trampa («vuelta» de Arrasate → regreso) |
| **La válvula calla de más** | 4 | context recall 0 porque no pasa ningún chunk: Erik Lira (el verificador confunde «la respuesta es que no» con «no es la fuente»), los dos de Ochoa (el resumen RSS cruzado Málaga/Girona) y Svilar (el chunk ni siquiera llegó) |
| **Generación infiel** | 2 | faithfulness 0,4 y 0,2 con contexto completo: Beitia y Sadiq |
| Discrepancia del juez | 1 | Celta: la respuesta nombra a los jugadores equivocados y el juez la da por fiel |

**La falla que queda es de generación, no de recuperación**: el documento correcto llega en 22 de 23
preguntas, la faithfulness de lo que se responde es 0,89, y aun así el 1.5B pone mal la etiqueta en 8.
Etiquetar es justo lo que hace el clasificador de M1; separar «recuperar y redactar» (el RAG) de
«etiquetar el chunk recuperado» (un clasificador afinado) es el siguiente paso natural. El context
recall baja del ingenuo (0,88) al avanzado (0,74) no porque se recupere peor, sino porque la válvula
deja sin contexto a 4 preguntas: es el precio medido de no inventar.

**4 · ¿Qué pasó con las preguntas sin respuesta en el corpus?** El RAG ingenuo **inventó en las
cuatro**: a Mbappé le trasladó la sanción de cuatro partidos de Vinicius, el Balón de Oro lo respondió
de memoria («el inglés fue el último…») y al Betis-Oporto le inventó «un impacto positivo por su
historia». El sistema final **activa la válvula en 3 de 4** (Mbappé, cuota, Balón de Oro). Falla en
Lamine Yamal: el verificador deja pasar «Pues el día que Lamine esté fino» (P = 0,999) y el generador
inventa «Lamine está cerca de estar disponible». El LLM sin recuperación también calla en las cuatro,
pero porque calla en casi todo (21 de 23 con respuesta): una válvula que siempre está cerrada no es
una válvula.

**Lo que no sabíamos al empezar y ahora sí.** Con un generador de 1.5B, la válvula por prompt es un
dial y no un interruptor (o calla de más, o nunca calla), y el modelo tampoco sabe decidir cuándo usar
una herramienta (0 de 4). Las dos decisiones salieron del generador: la primera la toma ahora un
clasificador sí/no por logits y la segunda una regla verificable. Y el problema de datos del feed
(resúmenes cruzados entre noticias) que encontramos al construir el corpus terminó costando dos
preguntas del eval set.

**Sobre el juez de la dimensión 2.** En M2 dijimos que en M3 lo dejaríamos razonar antes de puntuar.
No lo hicimos, a propósito: cambiar el juez es cambiar la vara, y el enunciado pide la misma para
todas las filas. Es el juez de M2 tal cual (Qwen2.5-1.5B, rúbrica v1) y sigue viendo solo la señal
renderizada con la plantilla de longitud fija, así que la prosa más larga del RAG no le da puntos por
verbosidad. En este eval set discrimina mejor que en M2 —3,26 cuando la categoría es correcta frente
a 2,48 cuando no (delta +0,79, antes +0,45), y solo en 2 de 27 preguntas le da la misma nota a una
respuesta correcta y a una incorrecta (antes 9 de 21)— y conserva su punto ciego: al signo invertido
le pone un 3. Por eso ese error lo sigue contando la dimensión 3.

## 10 · Reproducir

Todo corre en Colab gratis (T4) sin claves: modelos abiertos y locales. El notebook
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
python test_parseo.py                  # parseo tolerante y herramienta (también corre en el CI)
```

Semilla 42, decodificación greedy, versiones exactas de librerías y commit fijo de cada modelo de
Hugging Face (`rag/config.py`). Los resultados versionados se produjeron en una GTX 1650 Ti (4 GB,
fp16) con Python 3.12; en otra GPU las latencias cambian y, por la aritmética de punto flotante de cada
GPU, alguna respuesta greedy puede cambiar en un token. Las respuestas exactas de cada sistema están en
[`rag/resultados/respuestas/`](../rag/resultados/respuestas/), así que el harness y RAGAS se pueden
recalcular sobre ellas sin volver a generar.

## 11 · Limitaciones

- **27 preguntas.** Un acierto son 0,037 de accuracy; las conclusiones se apoyan en los casos, no en
  la segunda cifra decimal.
- **Escribimos el eval set conociendo el corpus.** Eso explica en parte que la recuperación sea tan
  fácil (casi todas las preguntas nombran al jugador). Un usuario real preguntará peor.
- **El juez de RAGAS es el mismo modelo base que genera.** No sabe qué sistema respondió, pero el
  riesgo de autopreferencia existe. Su sanity check es 8/8 y su context precision coincide con la
  calculada con documentos anotados a mano (0,83 frente a 0,83), y aun así en Celta da por fiel una
  respuesta que nombra a los jugadores equivocados. Al descomponer respuestas, a veces mezcla la
  pregunta en la afirmación (Chaira).
- **El dev set es pequeño** (14 preguntas, 3 sin respuesta) y en DEV-14 el verificador «acierta» abriendo
  sobre un documento que no es el de Lizancos.
- **Un solo corpus de prensa española con poca presencia latinoamericana** (36 de 748 fragmentos de El
  Tiempo), y resúmenes RSS que a veces pertenecen a otra noticia.
- **Latencias de una GPU de portátil.** La comparación entre filas vale; los segundos absolutos no.
