# Harness de evaluación y scorecard del baseline (M2)

> **Qué evalúa este harness, en una frase:** mide si la señal que Offside le entrega a un apostador —qué pasó, a quién le afecta y cuánto— le sirve para decidir mejor que si hubiera leído el titular por su cuenta.
>
> Artefactos: [`../eval/eval_set.json`](../eval/eval_set.json) · [`../eval/judge_rubric.yaml`](../eval/judge_rubric.yaml) · [`../eval/harness.py`](../eval/harness.py) · [`../eval/scorecard_baseline.csv`](../eval/scorecard_baseline.csv)

---

## Qué es una buena respuesta en nuestro dominio

Antes del código va el criterio, porque si no se puede escribir, ningún juez lo puede medir.

> **Una respuesta es buena si el usuario, actuando solo con ella, decide mejor que si hubiera leído el titular por su cuenta.**

De ese principio sale todo lo demás, incluida la asimetría que ordena la escala del juez: un error que hace **actuar sobre información falsa** es más grave que uno que simplemente **no aporta nada**. En el segundo caso el usuario queda como estaba; en el primero decide peor por culpa nuestra.

En concreto, una respuesta buena **acierta el tipo de hecho, no se equivoca en la dirección del impacto, no confunde un rumor con un hecho, y ante la duda calla en vez de inventar.**

---

## 1 · El eval set de dominio

21 ejemplos, cada uno con los tres campos que pide el módulo: `input`, `esperado` y `criterio`.

| | |
|---|---:|
| Ejemplos | **21** |
| Adversariales o de borde | **13 (62 %)** — el mínimo pedido es 20 % |
| Controles fáciles | 3 |
| Categorías cubiertas | **8 / 8** |
| Redactados por el equipo | 1 (el resto es RSS real) |

### Cómo se construyó

El texto **no se transcribe a mano**: [`build_eval_set.py`](../eval/build_eval_set.py) lo extrae del corpus por su identificador, así que es byte a byte el que recolectamos, y cada ejemplo arrastra su enlace de origen. En el script solo viven las etiquetas gold, el criterio y el razonamiento.

No está contaminado: son noticias de agosto de 2026 etiquetadas por nosotros; no salen de ningún benchmark público.

### Los casos adversariales

El módulo pide al menos un 20 % de casos ambiguos, con trampa, fuera de dominio o donde el sistema suele fallar. Tenemos 62 %, y cada uno ejercita un modo de fallo distinto:

| Tipo | Ejemplo | Por qué es difícil |
|---|---|---|
| **Trampa: declaración que contiene un hecho** | `OFF-01` Getafe comunica «una grave lesión»… se pierde toda la temporada | Las comillas disparan «declaración» y se pierde el hecho que hay dentro |
| **Trampa: disparador fuera de contexto** | `OFF-03` roja en el Trofeo Naranja (amistoso) | Una roja en amistoso no arrastra sanción liguera: no hay señal que dar |
| **Trampa: par mínimo** | `OFF-11` roja directa en partido de LaLiga | Texto casi idéntico a `OFF-03` y respuesta **opuesta**. Lo único que las separa es el contexto de competición, que no está dicho explícitamente |
| **Trampa: tiempo verbal** | `OFF-04` «Mudryk, 615 días después… fue suspendido en 2024» | La noticia es que **vuelve**; la sanción está en pasado. El signo correcto es el contrario del que sugiere el vocabulario |
| **Trampa: fragmento compuesto** | `OFF-07` un fichaje + un portero operado en el mismo resumen | La señal está en la segunda mitad, no en el titular |
| **Trampa: distractor normativo** | `OFF-08` la UEFA cambia el ciclo de amarillas | Vocabulario de sanción sin ningún equipo afectado |
| **Trampa: homonimia** | `OFF-09` «dudas defensivas» del Sporting | «Dudas» aquí es táctico, no una molestia física |
| **Ambiguo: frontera declaración/rumor** | `OFF-10` Neuer «insinúa» que podría retirarse | Lenguaje de rumor, pero la fuente es el propio jugador |
| **Ambiguo: molestia o pretexto** | `OFF-14` el «caso Arribas»: no viajó por molestias que avivan rumores de salida | El mismo texto admite dos lecturas y la diferencia importa |
| **Ambiguo: ¿a quién sanciona?** | `OFF-17` Dimayor castiga con seis fechas a un «integrante» | La sanción es indiscutible; el sujeto es vago, así que el impacto no se puede afirmar alto |
| **Fuera de dominio** | `OFF-16` Camila Osorio en el US Open | Es tenis. Llega por el mismo feed, tiene nombres propios y calendario: todo se parece menos lo esencial |
| **Variante lingüística** | `OFF-15` roja en Liga BetPlay (Colombia) | Todo el entrenamiento es prensa española. Mide el sesgo de variante que documentamos en M1 |
| **Rumor con forma de baja** | `OFF-18` (redactado) «según fuentes… podría perderse el derbi» | Comparte casi todo el vocabulario con una baja; solo lo separan los marcadores de incertidumbre |

### Los tres controles, y por qué importan

`OFF-19`, `OFF-20` y `OFF-21` son **fáciles a propósito**: una baja explícita, un fichaje irrelevante y una declaración sin hecho dentro.

Sin ellos el eval set sería solo dificultad, y entonces no se podría distinguir *«el sistema es malo»* de *«los casos son imposibles»*. También son la red de seguridad ante regresiones: si un cambio futuro rompe **estos**, algo va muy mal.

### El ejemplo redactado

`OFF-18` es el único que no sale del corpus. El motivo está documentado en el código: **el corpus no contiene ni un solo rumor no confirmado sobre disponibilidad**. Los únicos rumores que llegan por RSS son de fichajes.

Es una limitación real del formato titular + entradilla, no un descuido: ese tipo de rumor vive en el cuerpo del artículo o en redes, que deliberadamente no scrapeamos. Sin ningún ejemplo de esa clase, el criterio «no presentar un rumor como un hecho» no se puede medir, y es de los que más pesan. Lo redactamos con **entidades ficticias a propósito**, como todo lo sintético del proyecto: nunca atribuimos una lesión o sanción a una persona real que no la tuvo.

---

## 2 · Las tres dimensiones

| # | Dimensión | Qué mide | Qué NO ve |
|---|---|---|---|
| **1** | **Métrica clásica** · F1 macro sobre la categoría | Acierto por clase, sin ponderar por frecuencia. Automática, barata, reproducible | **La gravedad.** Para F1, confundir una baja con una sanción cuesta lo mismo que decir que un jugador vuelve cuando lo sancionaron |
| **2** | **LLM-as-a-judge** · Qwen2.5-1.5B-Instruct, rúbrica 1-5 anclada | Si la señal le sirve al usuario, pesando el daño de cada tipo de error | Es un modelo pequeño y falible. Su punto ciego está medido, ver abajo |
| **3** | **Dominio** · tasa de señal accionable | Qué fracción pasa **todas** las verificaciones duras de nuestro dominio, de forma determinista | No juzga matices: es una puerta de sí/no |

Cada una tapa el hueco de la otra. Leer una sola da una imagen equivocada, y por eso el scorecard las muestra juntas.

### Por qué la dimensión 3 NO depende del juez

El módulo sugiere «juez ≥ 4» como posible criterio de dominio. Lo probamos y lo descartamos **con datos**: nuestro juez concentra dos tercios de sus notas en el 3 y no detecta la inversión de signo. Atarle la dimensión 3 le importaría ese punto ciego justo a la dimensión que existe para cubrirlo, y las tres dejarían de ser miradas independientes.

La dimensión 3 verifica, sin LLM de por medio:

1. **No invierte el signo** del impacto (el error más caro).
2. **No inventa señal de alto impacto** donde no la había.
3. **No presenta un rumor como hecho** confirmado.
4. Acierta al menos la **familia** de la categoría — para el usuario, una baja y una sanción significan lo mismo: ese jugador no está.

El criterio alternativo se calcula igualmente y se reporta como `tasa_accionable_estricta`, para que se vea qué habría dado y por qué no lo usamos.

---

## 3 · El juez y su rúbrica

La rúbrica está versionada en [`judge_rubric.yaml`](../eval/judge_rubric.yaml) (`version: 1`). Cambiar un ancla cambia el veredicto, así que el scorecard solo es comparable entre corridas con la misma versión — el harness la registra.

### La escala, anclada

| Nivel | Nombre | Qué respuesta lo merece |
|---:|---|---|
| **5** | Señal correcta y accionable | Acierta hecho, dirección, intensidad y equipo. El usuario puede actuar tal cual |
| **4** | Correcta con un matiz menor | Hecho y **dirección** correctos; falla la intensidad o el equipo |
| **3** | No engaña, pero tampoco sirve | No emite señal habiendo una, **o** acierta el signo con una categoría vecina. Deja al usuario como estaba |
| **2** | Señal engañosa | Inventa señal donde no la hay, o infla el impacto a alto. Añade daño |
| **1** | Señal dañina | **Invierte el signo**, o presenta un rumor como hecho. Empuja en la dirección contraria |

El orden de la escala no es «cuánto acierta» sino **cuánto daño hace al decidir**. Por eso no emitir señal (3) puntúa por encima de emitir una falsa (2).

### Dos decisiones de implementación

**El juez no ve la etiqueta gold.** Recibe el fragmento, el `criterio` en prosa y la señal emitida. Si le diéramos la etiqueta correcta se convertiría en un comparador de strings y no aportaría nada sobre la dimensión 1.

**El puntaje se lee de los logits, no del texto.** En vez de generar texto y buscarle un número con una regex —que falla cuando el modelo contesta «Puntaje: 4/5» o se pone a explicar— comparamos directamente los logits de los tokens `1`..`5` en la primera posición de la respuesta. Es determinista, no necesita muestreo, **no puede fallar el parseo**, y deja una distribución de probabilidad sobre los cinco niveles que dice cuán segura fue la decisión.

### Cómo llegamos a esta versión (y qué sigue sin funcionar)

`Judge.sanity_check()` le pasa al juez el **mismo fragmento** con cinco señales de calidad conocida y comprueba que las ordena. La primera versión no pasó, y eso guió el diseño:

| Versión del prompt | Aciertos exactos | Qué pasaba |
|---|---:|---|
| Anclas en orden descendente (5→1), descripciones largas | **1 / 5** | El juez se quedaba anclado en el 5 y le daba **5 a una señal con el signo invertido** |
| Procedimiento explícito, sin ejemplos | 2 / 5 | Corregía el signo pero colapsaba todo lo demás a 1 |
| **Anclas ascendentes + un demo resuelto por nivel** | **3 / 5** | La versión actual |

Lo que el juez **sí** hace de forma fiable es separar una señal útil de una que no aporta. Lo que **no** hace: reconocer la inversión de signo — la puntúa 3 con una confianza de ~0,46, es decir, dudando.

No lo escondemos ni lo forzamos con más prompt engineering. **Ese error concreto no se delega en el juez**: lo verifica la dimensión 3 de forma determinista. Es la razón arquitectónica de que existan tres dimensiones y no dos.

---

## 4 · Sesgo del juez: detección y mitigación

**Sesgo elegido: verbosidad** (el juez premia respuestas largas aunque no aporten información).

### Por qué este y no otro

Nos afecta de lleno de cara a M3: el RAG va a redactar señales con más texto que el clasificador de M1. Si el juez premia la longitud, el RAG «ganaría» sin ser mejor y la comparación entre módulos —que es todo el punto de tener una vara fija— quedaría rota.

Los otros dos sesgos clásicos no aplican a este diseño y lo decimos: **posición** es un problema de comparaciones por pares (A vs B) y aquí puntuamos cada señal por separado; **auto-preferencia** requiere que el sistema evaluado genere texto libre, y los nuestros emiten etiquetas.

### La detección

[`bias_check.py`](../eval/bias_check.py) toma las **mismas predicciones** y las renderiza con tres longitudes distintas y exactamente la misma información:

| Condición | Ejemplo | Longitud |
|---|---|---:|
| `escueta` | `baja_confirmada/negativo_alto` | ~30 chars |
| `fija` | `baja_confirmada \| impacto: negativo_alto \| equipo: Getafe` | ~70 chars |
| `verbosa` | la misma información envuelta en relleno cortés | ~300 chars |

Si la nota sube al crecer la longitud, el sesgo está y es medible: nada más cambió.

### La mitigación

`render_senal()` en el harness emite **todas** las señales con la misma plantilla de longitud fija, sea cual sea el sistema. Así la longitud deja de ser una variable y no puede llevar información: el juez no tiene de dónde sacar la preferencia.

Es una mitigación **estructural**, no un ajuste sobre el puntaje. No pretendemos eliminar el sesgo del modelo —eso no se puede desde fuera—: le quitamos la señal de la que se alimenta, y medimos cuánta ventaja habría tenido un sistema verboso si no lo hubiéramos hecho.

---

## 5 · Reproducibilidad

| Requisito | Cómo se cumple |
|---|---|
| Semilla | `SEED = 42` en `random`, `numpy`, `torch` y el `Trainer` |
| Juez determinista | Sin muestreo: se leen logits, no se genera texto |
| Metadata de modelos | El scorecard guarda el id del juez, su versión de rúbrica y el resultado del sanity check |
| Sin rutas rotas | Todo relativo a `proyecto1/eval/`; los datos vienen en el repo |
| Sin contaminación | El eval set se excluye del entrenamiento; `train_holdout_model.py` **aborta** si un id se cuela |
| Corre sin GPU | `--sin-juez` da las dimensiones 1 y 3 en menos de un segundo, sin descargar ningún LLM |

```bash
cd proyecto1/eval
python build_eval_set.py                              # regenera el eval set
python harness.py --sin-juez                          # dims 1 y 3, instantáneo
python harness.py                                     # + juez (descarga Qwen2.5-1.5B)
python harness.py --adapter ../m1_lora_adapter_holdout  # + el modelo de M1
python bias_check.py                                  # experimento de sesgo
```

### Contaminación

El eval set se curó del mismo corpus que alimenta el entrenamiento, así que **hay que excluirlo**. `harness.eval_corpus_ids()` expone los 20 identificadores (el 21º es el redactado) y `train_holdout_model.py` reentrena excluyéndolos, abortando si alguno se cuela.
