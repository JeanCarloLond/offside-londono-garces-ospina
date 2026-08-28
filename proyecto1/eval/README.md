# Evaluación — eval set de dominio y harness

El harness mide si la señal que Offside le entrega a un apostador le sirve para decidir mejor que si hubiera leído el titular por su cuenta. Ver [`../docs/harness-m2.md`](../docs/harness-m2.md) para el razonamiento completo; este README es sobre **cómo correrlo**.

## Cómo se corre

```bash
cd proyecto1/eval
pip install -r ../../requirements-data.txt      # pyyaml, requests

python build_eval_set.py        # regenera eval_set.json desde el corpus
python harness.py --sin-juez    # dimensiones 1 y 3 — instantáneo, sin GPU ni LLM
python harness.py               # + dimensión 2 (descarga Qwen2.5-1.5B-Instruct)
python bias_check.py            # experimento de sesgo de verbosidad
```

Para incluir el modelo afinado de M1 **sin contaminación**:

```bash
python train_holdout_model.py                            # reentrena excluyendo el eval set
python harness.py --adapter ../m1_lora_adapter_holdout
```

O todo de una vez: `bash run_all.sh`.

## Qué hay aquí

| Archivo | Qué es |
|---|---|
| `eval_set.json` | Los 21 ejemplos gold: `input`, `esperado` y `criterio`, más el tipo de caso difícil |
| `build_eval_set.py` | Construye el eval set. Las etiquetas y el razonamiento viven aquí; el texto se extrae del corpus por `id` para no transcribirlo |
| `judge_rubric.yaml` | **La rúbrica del juez, versionada**: escala 1-5 con anclas y un demo resuelto por nivel |
| `judge.py` | El juez LLM. Puntúa leyendo los logits de los tokens `1`..`5`, así que es determinista y no puede fallar el parseo. Incluye `sanity_check()` |
| `harness.py` | `harness(eval_set, sistema)` → scorecard con las tres dimensiones. Sistemas intercambiables |
| `bias_check.py` | Detección y mitigación del sesgo de verbosidad del juez |
| `train_holdout_model.py` | Reentrena BETO+LoRA excluyendo el eval set; aborta si un id se cuela |
| `scorecard_baseline.csv` | El scorecard del baseline — el entregable |
| `scorecard_baseline.json` | El mismo, con el detalle por ejemplo |
| `bias_report.json` | Resultado del experimento de sesgo |

## El contrato del sistema evaluado

Cualquier cosa que cumpla esto entra en el harness sin tocar nada:

```python
def sistema(entrada: dict) -> dict:
    """entrada    {"text": str, "source": str, "published_at": str | None}
    respuesta  {"category": str, "impact": str, "team": str}"""
```

Es lo que va a permitir que el RAG de M3 se evalúe con la misma vara sin modificar `harness.py`.

> Ojo al añadir un sistema: la señal se renderiza siempre con `render_senal()`, que usa una plantilla de longitud fija. Eso es la mitigación del sesgo de verbosidad — un sistema que redacte su propia señal más larga rompería la comparación.

## Las tres dimensiones

1. **Métrica clásica** — F1 macro sobre la categoría. Automática y reproducible, pero ciega a la gravedad del error.
2. **LLM-as-a-judge** — Qwen2.5-1.5B-Instruct con la rúbrica anclada. Pesa el daño, que es lo que F1 no ve.
3. **Dominio** — tasa de señal accionable: verificaciones duras y deterministas de los errores que no nos podemos permitir. **No depende del juez**, a propósito.

## Para ampliar el eval set

Añadir una entrada a `CURADOS` en `build_eval_set.py`: el `id` del fragmento en el corpus, la etiqueta gold, el `criterio` (qué haría buena a la respuesta **en ese caso**), si es adversarial y por qué es difícil.

Ese `criterio` no es decorativo: es literalmente lo que ve el juez, así que un criterio vago da una nota poco fiable.

Después de ampliarlo hay que **reentrenar** antes de volver a comparar contra el modelo afinado, porque los ids nuevos deben quedar fuera del train.
