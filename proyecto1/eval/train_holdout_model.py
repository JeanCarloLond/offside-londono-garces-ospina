"""Entrena BETO+LoRA **excluyendo el eval set**, para poder evaluarlo sin trampa.

Por qué existe aparte del notebook de M1
----------------------------------------
El eval set de dominio se curó a partir del mismo corpus que alimenta el
entrenamiento. El modelo de M1 vio 10 de esos 10 ejemplos durante el train, así
que medirlo contra ellos sería medir memoria, no generalización — exactamente la
contaminación que el módulo advierte que infla los leaderboards.

Este script repite el entrenamiento de M1 con una sola diferencia: descarta del
train los ids que forman el eval set. El adaptador resultante sí se puede
evaluar honestamente con `harness.py --adapter ...`.

Uso:
    python train_holdout_model.py                 # ~8 min en CPU, <1 min en GPU
    python harness.py --adapter ../m1_lora_adapter_holdout
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from harness import CATEGORIES, eval_corpus_ids
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def split_por_fecha(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Mismo split temporal del notebook de M1: evita fuga temporal."""
    por_cat = defaultdict(list)
    for r in records:
        por_cat[r["category"]].append(r)
    train, val = [], []
    for items in por_cat.values():
        ordenados = sorted(items, key=lambda r: r["published_at"] or "")
        n_val = max(1, int(len(ordenados) * 0.2)) if len(ordenados) >= 5 else 0
        if n_val == 0:
            train.extend(ordenados)
        else:
            train.extend(ordenados[:-n_val])
            val.extend(ordenados[-n_val:])
    return train, val


def main() -> int:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=here / ".." / "data" / "processed" / "weak_labeled.jsonl"
    )
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=here / ".." / "data" / "gold" / "gold_seed_synthetic.jsonl",
    )
    parser.add_argument("--eval-set", type=Path, default=here / "eval_set.json")
    parser.add_argument("--out", type=Path, default=here / ".." / "m1_lora_adapter_holdout")
    parser.add_argument("--base-model", default="dccuchile/bert-base-spanish-wwm-cased")
    parser.add_argument("--epochs", type=float, default=10)
    args = parser.parse_args()

    holdout = eval_corpus_ids(args.eval_set)

    with args.corpus.open(encoding="utf-8") as f:
        reales = [json.loads(line) for line in f if line.strip()]
    n_antes = len(reales)
    reales = [r for r in reales if r["id"] not in holdout]
    print(f"Corpus: {n_antes} -> {len(reales)} tras excluir el eval set ({len(holdout)} ids)")

    with args.synthetic.open(encoding="utf-8") as f:
        sinteticos = [json.loads(line) for line in f if line.strip()]

    train_real, val_real = split_por_fecha(reales)
    rng = random.Random(SEED)
    por_cat = defaultdict(list)
    for r in sinteticos:
        por_cat[r["category"]].append(r)
    train_syn, val_syn = [], []
    for items in por_cat.values():
        items = items[:]
        rng.shuffle(items)
        n_val = 1 if len(items) >= 3 else 0
        val_syn.extend(items[:n_val])
        train_syn.extend(items[n_val:])

    train_records = train_real + train_syn
    val_records = val_real + val_syn
    rng2 = random.Random(SEED)
    rng2.shuffle(train_records)
    rng2.shuffle(val_records)

    # Verificación dura: si un id del eval set se coló en el train, abortar.
    colados = [r["id"] for r in train_records if r["id"] in holdout]
    if colados:
        raise SystemExit(f"CONTAMINACIÓN: estos ids del eval set están en el train: {colados}")
    print(f"train={len(train_records)} val={len(val_records)} | sin contaminación del eval set")
    print("train:", Counter(r["category"] for r in train_records).most_common())

    cat2id = {c: i for i, c in enumerate(CATEGORIES)}
    all_labels = list(range(len(CATEGORIES)))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def to_tok(records):
        ds = Dataset.from_dict(
            {
                "text": [r["text"] for r in records],
                "label": [cat2id[r["category"]] for r in records],
            }
        )
        return ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=128), batched=True
        ).remove_columns(["text"])

    train_tok, val_tok = to_tok(train_records), to_tok(val_records)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(CATEGORIES)
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value"],
            # `pooler` va aquí por reproducibilidad, no por capacidad. BETO es un
            # checkpoint de MLM: no trae pooler, así que transformers lo inicializa
            # AL AZAR en cada carga. Si no se guarda con el adaptador, al recargar
            # aparece un pooler distinto y la cabeza entrenada deja de tener sentido
            # -> las predicciones cambian en cada proceso. Incluirlo en
            # modules_to_save lo persiste y hace el adaptador autocontenido.
            modules_to_save=["classifier", "pooler"],
        ),
    )
    model.print_trainable_parameters()

    def metricas(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "f1_macro": f1_score(labels, preds, average="macro", labels=all_labels, zero_division=0)
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(here / "_train_out"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            learning_rate=2e-4,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=20,
            seed=SEED,
            report_to=[],
            disable_tqdm=True,
        ),
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metricas,
    )
    trainer.train()
    print("eval final:", trainer.evaluate())

    model.save_pretrained(str(args.out))
    print(f"\nAdaptador guardado en {args.out}")
    print("Ahora:  python harness.py --adapter", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
