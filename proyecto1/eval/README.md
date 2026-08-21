# Evaluación — eval set de dominio y harness

Nuestro propio conjunto de evaluación y el harness que lo convierte en un scorecard. Ver [`../docs/eval-set.md`](../docs/eval-set.md) para el razonamiento completo; este README es sobre **cómo correrlo**.

## Orden de ejecución

```bash
cd proyecto1/eval
pip install -r ../../requirements-data.txt   # pyyaml; además scikit-learn

python build_eval_set.py     # regenera eval_set.jsonl desde el corpus
python harness.py            # scorecard del baseline — <1s, sin GPU
python harness.py --json scorecard.json      # además lo guarda en JSON
```

Para evaluar también el modelo afinado, **sin contaminación**:

```bash
python train_holdout_model.py                      # reentrena excluyendo el eval set
python harness.py --adapter ../m1_lora_adapter_holdout
```

## Qué hay aquí

| Archivo | Qué es |
|---|---|
| `eval_set.jsonl` | Los 10 ejemplos gold: input + salida esperada + por qué es difícil cada uno |
| `build_eval_set.py` | Construye el eval set. Las etiquetas gold y su razonamiento viven aquí; el texto se extrae del corpus por `id` para no transcribirlo a mano |
| `rubric.yaml` | Qué hace "buena" a una respuesta en nuestro dominio: criterios, pesos y el principio que los ordena. En S06 es el prompt del LLM-as-a-judge |
| `harness.py` | El harness: tres dimensiones → scorecard. Sistemas intercambiables (`--systems`, `--adapter`) |
| `train_holdout_model.py` | Reentrena BETO+LoRA excluyendo el eval set, para poder medirlo sin trampa |

## Las tres dimensiones

1. **Métrica clásica** — F1 macro (automática, barata). `accuracy` se imprime solo como contraste: sube con la clase mayoritaria y por eso no es la métrica principal.
2. **Rúbrica** — los criterios de `rubric.yaml`. Hoy deterministas (C1–C5); el juez LLM llega en S06.
3. **De dominio** — dónde falla, en el lenguaje del problema: desglose por tipo de caso difícil y recuento de los errores que más caros salen (señal falsa de alto impacto, señal perdida, signo invertido).

## Sobre la contaminación

El eval set se curó a partir del mismo corpus que alimenta el entrenamiento, así que **hay que excluirlo del train**. El harness lo detecta y lo avisa; `eval_corpus_ids()` expone los ids para excluirlos en una línea, y `train_holdout_model.py` aborta si alguno se cuela.

Si añaden ejemplos nuevos al eval set, hay que reentrenar antes de volver a comparar contra el modelo afinado.

## Para ampliar el eval set

Añadir una entrada a `CURATED` en `build_eval_set.py`: el `id` del fragmento en el corpus, la etiqueta gold, el tipo de caso difícil y **por qué** esa es la respuesta correcta. Ese `why` no es decorativo — es lo que hace defendible la etiqueta en los casos frontera.

Prioridad actual: **ejemplos de `rumor_no_confirmado` reales**, porque sin ellos el criterio C4 de la rúbrica no se puede medir.
