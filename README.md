# Offside

[![lint](https://github.com/JeanCarloLond/offside-londono-garces-ospina/actions/workflows/lint.yml/badge.svg)](https://github.com/JeanCarloLond/offside-londono-garces-ospina/actions/workflows/lint.yml)
[![Licencia MIT](https://img.shields.io/badge/licencia-MIT-blue.svg)](LICENSE)
[![Módulo](https://img.shields.io/badge/SI4006-M1%20·%20M2%20·%20M3-success)](#entregables)

> **Offside responde las preguntas de un apostador recreativo sobre noticias de fútbol —«¿juega Vinicius el Clásico?»— buscando en la prensa que recolectamos, para que decida con información y no solo con el titular que alcanzó a leer en el celular.**
>
> Una respuesta es buena cuando **dice lo que dicen las noticias, cita de cuál lo saca, acierta el tipo de hecho y la dirección del impacto, no confunde un rumor con un hecho y, si no lo sabe, lo dice en vez de inventar**. Y para saber si lo consigue, lo medimos con la misma vara desde el Módulo 2.

Proyecto integrador del curso SI4006 · Tópicos Especiales y Aplicaciones en IA · Universidad EAFIT.

---

## Entregables

| Módulo | Qué es | Empiece por aquí | Artefactos |
|---|---|---|---|
| **M3** · RAG agéntico | Un RAG sobre 748 noticias reales: ingenuo, avanzado (híbrida + reranker + válvula verificada) y con una herramienta de fechas, medido con el harness de M2 y con RAGAS | **[Ir a M3 ↓](#m3--rag-que-recupera-actúa-y-se-evalúa)** · [`docs/rag-m3.md`](proyecto1/docs/rag-m3.md) | [eval set](proyecto1/eval/eval_set_m3.json) · [código](proyecto1/rag/) · [scorecard](proyecto1/rag/resultados/scorecard_m3.csv) · [RAGAS](proyecto1/rag/resultados/ragas_m3.csv) · [notebook](proyecto1/notebooks/M3_RAG_agentico.ipynb) |
| **M2** · Evaluación rigurosa | Harness de 3 dimensiones + scorecard del baseline, con la rúbrica del juez y su sesgo medido | **[Ir a M2 ↓](#m2--harness-de-evaluación)** · [`docs/harness-m2.md`](proyecto1/docs/harness-m2.md) | [eval set](proyecto1/eval/eval_set.json) · [rúbrica del juez](proyecto1/eval/judge_rubric.yaml) · [scorecard](proyecto1/eval/scorecard_baseline.csv) · [notebook](proyecto1/notebooks/M2_harness_evaluacion.ipynb) |
| **M1** · Fine-tuning con LoRA | BETO afinado sobre prensa deportiva, comparado contra tres baselines | **[Ir a M1 ↓](#m1--fine-tuning-con-lora)** · [`docs/results.md`](proyecto1/docs/results.md) | [notebook](proyecto1/notebooks/M1_finetuning_LoRA_offside.ipynb) · [dataset](proyecto1/docs/dataset.md) · [recolección](proyecto1/data_collection/) |

Los notebooks abren en Colab con un clic y clonan el repo solos:

[![M3 en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JeanCarloLond/offside-londono-garces-ospina/blob/main/proyecto1/notebooks/M3_RAG_agentico.ipynb) M3 · RAG
&nbsp;&nbsp;
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

## M3 · RAG que recupera, actúa y se evalúa

> ⚠️ **En curso (2026-10-14):** las tablas de esta sección son de una corrida que usó por error la
> versión v4 del prompt base en lugar de la v3 elegida con el dev set. Se está repitiendo todo con v3;
> las tablas, la lectura y `rag/resultados/` se actualizan en el siguiente commit.

Reporte completo en [`docs/rag-m3.md`](proyecto1/docs/rag-m3.md) · notebook ejecutable en
[`M3_RAG_agentico.ipynb`](proyecto1/notebooks/M3_RAG_agentico.ipynb) · código en
[`proyecto1/rag/`](proyecto1/rag/) · resultados en [`rag/resultados/`](proyecto1/rag/resultados/).

| | |
|---|---|
| **Corpus** | 748 fragmentos RSS reales (Marca + El Tiempo, agosto y 14 de octubre de 2026) = 748 chunks. Solo titular + entradilla de sindicación, con enlace; el texto es de cada medio |
| **Generador** | `Qwen/Qwen2.5-1.5B-Instruct`, greedy, **el mismo y con el mismo prompt base en todas las filas** |
| **Índice** | `multilingual-e5-small` + Chroma, y BM25 en memoria. Un fragmento = un chunk, sin solapamiento, top-k = 4 |
| **Eval set** | 27 preguntas con `input`, `esperado` y `criterio`; **4 sin respuesta en el corpus**, 3 que exigen calcular fechas, 17 adversariales |
| **Vara** | `harness(eval_set, sistema)` de M2 **sin tocar**, con su juez (Qwen2.5-1.5B, rúbrica v1), más RAGAS |

### El pipeline

```mermaid
flowchart LR
    subgraph IDX["Indexación (una vez)"]
        RSS["RSS de Marca y El Tiempo"] --> ING["ingest.py<br/>limpia, deduplica,<br/>guarda fecha y club"]
        ING --> CORP[("corpus.jsonl<br/>748 fragmentos")]
        CORP --> CH["chunk:<br/>1 fragmento = 1 chunk"]
        CH --> EMB["embed:<br/>multilingual-e5-small"]
        EMB --> CHROMA[("Chroma")]
        CH --> BM25[("BM25")]
    end

    Q(["Pregunta del apostador"]) --> DENSE["búsqueda densa"]
    CHROMA --> DENSE
    Q --> LEX["búsqueda léxica"]
    BM25 --> LEX
    DENSE --> RRF{{"★ Híbrida: fusión RRF"}}
    LEX --> RRF
    RRF --> RR{{"★ Reranker<br/>cross-encoder → top-4"}}
    RR --> VER{{"★ Válvula verificada<br/>Qwen sí/no por logits"}}
    VER -- "ningún chunk pasa" --> NO(["«No encuentro esa información<br/>en las noticias recolectadas»"])
    VER --> REG{"¿fecha en la pregunta<br/>y duración en el chunk?"}
    REG -- sí --> TOOL[["🛠 calcular_fecha<br/>el LLM propone el JSON<br/>(máx. 3 pasos)"]]
    TOOL --> GEN
    REG -- no --> GEN["Qwen2.5-1.5B-Instruct<br/>mismo prompt base"]
    GEN --> OUT(["respuesta con cita<br/>+ señal: categoría · impacto · equipo"])

    DENSE -. "RAG ingenuo: top-4 directo" .-> GEN
```

Las piezas con ★ son el RAG avanzado; 🛠 es la capa agéntica. La línea punteada es el RAG ingenuo:
búsqueda densa, top-4, generar.

### Por qué cada técnica: el diagnóstico del RAG ingenuo

Empezamos por la lista de fallos del ingenuo, no por el catálogo. Dos hallazgos lo ordenan todo:
**la recuperación casi nunca falla** (con búsqueda densa el documento correcto llega al top-4 en 22 de
23 preguntas) y **el generador de 1.5B no sabe callar a tiempo** (con el prompt que contesta bien,
inventa en las 4 preguntas sin respuesta; con el que calla, calla también en 12 de las 23 que sí la
tienen).

| Falla del ingenuo | Ejemplo | Qué la ataca |
|---|---|---|
| El chunk llega, pero un distractor va delante | «¿Cómo está Musiala?»: delante, noticias de agosto que lo daban de baja | **Reranker** (cross-encoder) |
| Nombres propios y términos exactos | dense pone ruido primero en Celta y Ochoa; BM25 no | **Búsqueda híbrida** BM25 + dense (RRF) |
| Inventa cuando no está en el corpus | «¿Mbappé se pierde el Clásico?» → le traslada la sanción de Vinicius | **Válvula verificada**: el mismo Qwen, sí/no por logits, chunk a chunk |
| Hay que calcular una fecha | «¿Puede contar el Oviedo con Chaira el 29 de noviembre?» | **Herramienta** `calcular_fecha` |
| El chunk no llega | el portero de la Roma: el titular es «El rapero con guantes…» | HyDE y multi-query: **probadas y descartadas**, empeoran la recuperación (MRR 0,90 → 0,74) |

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

**Ablación: una técnica a la vez** ([`scorecard_m3.csv`](proyecto1/rag/resultados/scorecard_m3.csv))

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

Las cuatro métricas, implementadas en [`ragas_local.py`](proyecto1/rag/ragas_local.py) con las
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
23 con respuesta del sistema final ([`ragas_m3_por_caso.csv`](proyecto1/rag/resultados/ragas_m3_por_caso.csv)):

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

### Uso de IA

Usamos **Claude Code** (Anthropic) como asistente de programación en este módulo, para:

- **Código:** la mayor parte de [`proyecto1/rag/`](proyecto1/rag/), el constructor del eval set
  ([`build_eval_set_m3.py`](proyecto1/eval/build_eval_set_m3.py)) a partir de noticias reales del
  corpus, y el notebook de M3.
- **Experimentos y depuración:** correr los diagnósticos, el ajuste de prompts sobre el dev set, las
  ablaciones y la evaluación; depurar el bucle del agente y el parseo tolerante.
- **Redacción:** la primera versión de esta sección y de [`docs/rag-m3.md`](proyecto1/docs/rag-m3.md).

**Lo que revisamos y cambiamos nosotros:** ⚠️ COMPLETAR ANTES DE ENTREGAR.

Los modelos que forman parte del sistema (Qwen2.5-1.5B como generador y juez, e5-small, el
cross-encoder) no cuentan aquí: esto es sobre cómo construimos la entrega.

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
│   ├── notebooks/            # Los entregables de M1, M2 y M3, ejecutables en Colab
│   ├── rag/                  # M3: indexación, sistemas RAG, herramienta, RAGAS y resultados/
│   ├── eval/                 # M2: harness, juez y eval sets (el de M3 incluido)
│   ├── data_collection/      # Recolección RSS + limpieza + supervisión débil
│   ├── data/                 # processed/, gold/ y rag/ se versionan; raw/ no (ver .gitignore)
│   └── docs/                 # Dataset, resultados de M1, harness de M2 y reporte de M3
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
  checkpoints. El porqué está en [`docs/dataset.md`](proyecto1/docs/dataset.md#idioma-y-licencia): no
  redistribuimos texto periodístico, solo fragmentos de sindicación con su enlace de vuelta.
- **Licencia separada para código y datos**: el código es MIT; el texto de prensa sigue siendo de
  cada medio, y así está dicho en [`LICENSE`](LICENSE).
- **Ramas cortas y revisión**: `feature/...` o `fix/...`, con revisión de otro integrante antes de
  mergear a `main`.
