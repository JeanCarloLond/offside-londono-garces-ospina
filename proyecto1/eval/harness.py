"""Harness de evaluación de Offside — produce el scorecard del baseline.

Combina las **tres dimensiones** que pide el módulo, sobre nuestro propio eval
set de dominio (`eval_set.jsonl`):

  1. MÉTRICA CLÁSICA   → F1 macro (automática, barata, reproducible)
  2. RÚBRICA           → los criterios de `rubric.yaml`, evaluados de forma
                         determinista donde se puede. En S06 esta misma rúbrica
                         pasa a ser el prompt del LLM-as-a-judge; el hueco está
                         declarado abajo en `LlmJudge`.
  3. DE DOMINIO        → desglose por tipo de caso difícil y por el error que a
                         nuestro usuario le sale más caro (señal falsa de alto
                         impacto, y signo invertido).

Los sistemas evaluados son intercambiables (`--systems`). Por defecto corre los
dos baselines, que no necesitan GPU ni pesos y tardan menos de un segundo.

Uso:
    python harness.py                          # baselines: mayoritaria + léxico
    python harness.py --systems lexicon        # solo uno
    python harness.py --adapter ../m1_lora_adapter   # añade el modelo afinado
    python harness.py --json scorecard.json    # guarda el scorecard para comparar
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml
from sklearn.metrics import accuracy_score, f1_score

CATEGORIES = [
    "baja_confirmada",
    "sancion_suspension",
    "duda_fisica",
    "regreso_alta",
    "cambio_tactico",
    "declaracion_contexto",
    "rumor_no_confirmado",
    "irrelevante",
]
HECHOS_CONFIRMADOS = {"baja_confirmada", "sancion_suspension"}


# --------------------------------------------------------------------------
# Utilidades de impacto
# --------------------------------------------------------------------------
def signo(impacto: str) -> str:
    """negativo / neutro / positivo. Equivocar esto invierte la decisión."""
    if impacto.startswith("negativo"):
        return "negativo"
    if impacto.startswith("positivo"):
        return "positivo"
    return "neutro"


def nivel(impacto: str) -> str:
    """alto / bajo / neutro. Equivocar esto degrada la señal, no la invierte."""
    if impacto.endswith("_alto"):
        return "alto"
    if impacto.endswith("_bajo"):
        return "bajo"
    return "neutro"


def es_alto_impacto(impacto: str) -> bool:
    return impacto in ("negativo_alto", "positivo_alto")


# --------------------------------------------------------------------------
# Sistemas evaluables (intercambiables)
# --------------------------------------------------------------------------
class Predictor:
    """Un sistema que, dado un texto, devuelve (categoría, impacto)."""

    name = "base"

    def predict(self, text: str) -> tuple[str, str]:
        raise NotImplementedError


class MajorityPredictor(Predictor):
    """Responde siempre la clase mayoritaria. El piso absoluto."""

    name = "mayoritaria"

    def predict(self, text: str) -> tuple[str, str]:
        return "irrelevante", "neutro"


class LexiconPredictor(Predictor):
    """El léxico/regex de `lexicon.yaml`: el baseline que queremos superar."""

    name = "lexico"

    def __init__(self, lexicon_path: Path):
        cfg = yaml.safe_load(lexicon_path.read_text(encoding="utf-8"))
        self.rules = [
            (c["name"], c["impact_default"], [re.compile(p, re.I) for p in c["patterns"]])
            for c in cfg["categories"]
        ]
        self.boost = {
            label: [re.compile(p, re.I) for p in pats]
            for label, pats in cfg.get("sentiment_boost", {}).items()
        }

    def predict(self, text: str) -> tuple[str, str]:
        for name, impact_default, patterns in self.rules:
            if any(p.search(text) for p in patterns):
                impact = impact_default
                for boosted, pats in self.boost.items():
                    if any(p.search(text) for p in pats):
                        impact = boosted
                        break
                return name, impact
        return "irrelevante", "neutro"


class FineTunedPredictor(Predictor):
    """BETO + adaptador LoRA de M1. El impacto se deriva de la categoría.

    Importar torch/transformers es caro, así que se hace aquí dentro: el harness
    corre los baselines sin tener nada de ML instalado.
    """

    name = "lora_finetuned"

    def __init__(self, adapter_dir: Path, lexicon_path: Path, base_model: str):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.avisos = self._revisar_adaptador(adapter_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=len(CATEGORIES)
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_dir))
        self.model.eval()

        cfg = yaml.safe_load(lexicon_path.read_text(encoding="utf-8"))
        self.cat_impact = {c["name"]: c["impact_default"] for c in cfg["categories"]}
        self.cat_impact["irrelevante"] = "neutro"

    @staticmethod
    def _revisar_adaptador(adapter_dir: Path) -> list[str]:
        """Comprueba que el adaptador basta para reproducir el modelo entrenado.

        BETO es un checkpoint de MLM: no trae `bert.pooler`, así que transformers
        lo inicializa AL AZAR en cada carga. El pooler está entre el encoder y la
        cabeza de clasificación, o sea que forma parte de la función del modelo.
        Si el adaptador no lo guarda, al recargar sale un pooler distinto, la
        cabeza entrenada deja de encajar con él y las predicciones cambian en
        cada proceso — un fallo silencioso que parece "el modelo es malo".
        """
        cfg_path = adapter_dir / "adapter_config.json"
        if not cfg_path.exists():
            return [f"no encuentro {cfg_path}"]
        guardados = json.loads(cfg_path.read_text(encoding="utf-8")).get("modules_to_save") or []
        if not any("pooler" in m for m in guardados):
            return [
                "el adaptador NO guarda `pooler`: se inicializa al azar en cada carga, "
                "así que estas predicciones no son reproducibles. Reentrena con "
                "modules_to_save=['classifier', 'pooler'] (ver train_holdout_model.py)."
            ]
        return []

    def predict(self, text: str) -> tuple[str, str]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        category = CATEGORIES[int(logits.argmax(-1))]
        return category, self.cat_impact[category]


class LlmJudge:
    """Dimensión 2 completa: un LLM aplica `rubric.yaml` a cada respuesta.

    Pendiente de S06 (el módulo lo cubre esa semana). El hueco queda declarado
    a propósito en vez de improvisar un juez: la rúbrica ya está escrita y es
    exactamente el prompt que recibirá. Hasta entonces, `score_rubric()` evalúa
    de forma determinista los criterios que no necesitan juicio (C1-C5).
    """

    available = False


# --------------------------------------------------------------------------
# Las tres dimensiones
# --------------------------------------------------------------------------
def dim1_metrica_clasica(gold_cats: list[str], pred_cats: list[str]) -> dict:
    """F1 macro (la métrica principal del proyecto) + accuracy para contraste."""
    return {
        "f1_macro": f1_score(
            gold_cats, pred_cats, average="macro", labels=CATEGORIES, zero_division=0
        ),
        "accuracy": accuracy_score(gold_cats, pred_cats),
    }


def score_rubric(rows: list[dict], rubric: dict) -> dict:
    """Dimensión 2: puntúa los criterios deterministas de la rúbrica."""
    pesos = {c["id"]: c["weight"] for c in rubric["criterios"]}
    nombres = {c["id"]: c["nombre"] for c in rubric["criterios"]}

    c1 = [r["pred_cat"] == r["gold_cat"] for r in rows]
    c2 = [signo(r["pred_imp"]) == signo(r["gold_imp"]) for r in rows]
    c3 = [nivel(r["pred_imp"]) == nivel(r["gold_imp"]) for r in rows]

    # C4 solo aplica donde la verdad es un rumor; si no hay rumores en el eval
    # set, el criterio no se puede medir (y decirlo es mejor que dar un 1.0).
    rumores = [r for r in rows if r["gold_cat"] == "rumor_no_confirmado"]
    c4 = [r["pred_cat"] not in HECHOS_CONFIRMADOS for r in rumores]

    # C5 = precisión sobre impacto alto: de lo que marcamos como alto, cuánto lo era.
    marcados_alto = [r for r in rows if es_alto_impacto(r["pred_imp"])]
    c5 = [es_alto_impacto(r["gold_imp"]) for r in marcados_alto]

    def media(xs):
        return sum(xs) / len(xs) if xs else None

    detalle = {
        "C1": {"nombre": nombres["C1"], "score": media(c1), "n": len(c1)},
        "C2": {"nombre": nombres["C2"], "score": media(c2), "n": len(c2)},
        "C3": {"nombre": nombres["C3"], "score": media(c3), "n": len(c3)},
        "C4": {"nombre": nombres["C4"], "score": media(c4), "n": len(c4)},
        "C5": {"nombre": nombres["C5"], "score": media(c5), "n": len(c5)},
        "C6": {"nombre": nombres["C6"], "score": None, "n": 0},
    }

    # Media ponderada solo sobre los criterios que sí se pudieron medir.
    medibles = [(cid, d) for cid, d in detalle.items() if d["score"] is not None]
    peso_total = sum(pesos[cid] for cid, _ in medibles)
    global_score = (
        sum(pesos[cid] * d["score"] for cid, d in medibles) / peso_total if peso_total else None
    )
    return {"criterios": detalle, "score_global": global_score, "peso_cubierto": peso_total}


def dim3_dominio(rows: list[dict]) -> dict:
    """Dimensión de dominio: dónde falla, en el lenguaje de nuestro problema."""
    por_dificultad = {}
    for r in rows:
        d = por_dificultad.setdefault(r["difficulty"], {"n": 0, "ok": 0})
        d["n"] += 1
        d["ok"] += int(r["pred_cat"] == r["gold_cat"])

    señal_falsa = [
        r for r in rows if es_alto_impacto(r["pred_imp"]) and not es_alto_impacto(r["gold_imp"])
    ]
    señal_perdida = [
        r for r in rows if es_alto_impacto(r["gold_imp"]) and not es_alto_impacto(r["pred_imp"])
    ]
    signo_invertido = [
        r
        for r in rows
        if signo(r["pred_imp"]) != signo(r["gold_imp"])
        and "neutro" not in (signo(r["pred_imp"]), signo(r["gold_imp"]))
    ]
    return {
        "por_dificultad": por_dificultad,
        "senal_falsa_alto_impacto": [r["eval_id"] for r in señal_falsa],
        "senal_perdida_alto_impacto": [r["eval_id"] for r in señal_perdida],
        "signo_invertido": [r["eval_id"] for r in signo_invertido],
    }


# --------------------------------------------------------------------------
# Contaminación
# --------------------------------------------------------------------------
def check_holdout(eval_records: list[dict], corpus_path: Path) -> dict:
    """El eval set sale del mismo corpus que alimenta el train: hay que excluirlo.

    Un eval set contaminado no mide generalización, mide memoria. Esto reporta
    los ids que el entrenamiento debe dejar fuera.
    """
    eval_ids = {r["provenance"]["corpus_id"] for r in eval_records}
    if not corpus_path.exists():
        return {"eval_ids": sorted(eval_ids), "en_corpus": None}
    corpus_ids = set()
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus_ids.add(json.loads(line)["id"])
    return {"eval_ids": sorted(eval_ids), "en_corpus": sorted(eval_ids & corpus_ids)}


def eval_corpus_ids(eval_path: Path) -> set[str]:
    """Para que el código de entrenamiento excluya el eval set en una línea."""
    with eval_path.open(encoding="utf-8") as f:
        return {json.loads(line)["provenance"]["corpus_id"] for line in f if line.strip()}


# --------------------------------------------------------------------------
def evaluar(predictor: Predictor, eval_records: list[dict]) -> list[dict]:
    rows = []
    for rec in eval_records:
        pred_cat, pred_imp = predictor.predict(rec["input"]["text"])
        rows.append(
            {
                "eval_id": rec["eval_id"],
                "difficulty": rec["difficulty"],
                "gold_cat": rec["expected"]["category"],
                "gold_imp": rec["expected"]["impact"],
                "pred_cat": pred_cat,
                "pred_imp": pred_imp,
            }
        )
    return rows


def imprimir_scorecard(resultados: dict, eval_records: list[dict], holdout: dict) -> None:
    sistemas = list(resultados)
    ancho = max(len(s) for s in sistemas) + 2

    print("=" * 78)
    print("SCORECARD · Offside — eval set de dominio")
    print("=" * 78)
    print(f"Ejemplos: {len(eval_records)}  |  Sistemas: {', '.join(sistemas)}")
    dificiles = sum(1 for r in eval_records if r["lexicon_fails"])
    print(f"Casos donde el léxico falla por diseño del set: {dificiles}/{len(eval_records)}")

    print("\n" + "-" * 78)
    print("DIMENSIÓN 1 · MÉTRICA CLÁSICA (automática)")
    print("-" * 78)
    print(f"{'sistema':<{ancho}}{'F1 macro':>12}{'accuracy':>12}")
    for s in sistemas:
        d = resultados[s]["dim1"]
        print(f"{s:<{ancho}}{d['f1_macro']:>12.4f}{d['accuracy']:>12.4f}")
    print("\n  accuracy se muestra solo para contraste: sube con la clase mayoritaria")
    print("  y por eso no la usamos como métrica principal.")

    print("\n" + "-" * 78)
    print("DIMENSIÓN 2 · RÚBRICA DE DOMINIO (criterios de rubric.yaml)")
    print("-" * 78)
    ids = ["C1", "C2", "C3", "C4", "C5"]
    print(f"{'sistema':<{ancho}}" + "".join(f"{c:>8}" for c in ids) + f"{'GLOBAL':>10}")
    for s in sistemas:
        r = resultados[s]["dim2"]
        celdas = ""
        for c in ids:
            sc = r["criterios"][c]["score"]
            celdas += f"{'  n/d':>8}" if sc is None else f"{sc:>8.2f}"
        g = r["score_global"]
        print(f"{s:<{ancho}}" + celdas + (f"{g:>10.3f}" if g is not None else f"{'n/d':>10}"))
    print()
    for c in ids:
        print(f"  {c} = {resultados[sistemas[0]]['dim2']['criterios'][c]['nombre']}")
    print("  C6 (equipo afectado) y el juez LLM llegan en S06 — ver rubric.yaml.")

    print("\n" + "-" * 78)
    print("DIMENSIÓN 3 · VISTA DE DOMINIO (dónde falla, en lenguaje del problema)")
    print("-" * 78)
    dificultades = sorted({r["difficulty"] for r in eval_records})
    print(f"{'caso difícil':<44}" + "".join(f"{s:>16}" for s in sistemas))
    for d in dificultades:
        fila = f"{d:<44}"
        for s in sistemas:
            pd = resultados[s]["dim3"]["por_dificultad"].get(d, {"n": 0, "ok": 0})
            marcador = f"{pd['ok']}/{pd['n']}"
            fila += f"{marcador:>16}"
        print(fila)

    print("\n  Errores que más caros salen para el usuario:")
    for s in sistemas:
        d3 = resultados[s]["dim3"]
        print(f"    {s}:")
        print(
            f"      señal FALSA de alto impacto (el peor): "
            f"{len(d3['senal_falsa_alto_impacto'])} → {d3['senal_falsa_alto_impacto'] or '-'}"
        )
        print(
            f"      señal PERDIDA de alto impacto:         "
            f"{len(d3['senal_perdida_alto_impacto'])} → {d3['senal_perdida_alto_impacto'] or '-'}"
        )
        print(
            f"      signo del impacto INVERTIDO:           "
            f"{len(d3['signo_invertido'])} → {d3['signo_invertido'] or '-'}"
        )

    print("\n" + "-" * 78)
    print("CONTAMINACIÓN / HOLD-OUT")
    print("-" * 78)
    en_corpus = holdout["en_corpus"]
    if en_corpus:
        print(f"  {len(en_corpus)} de los {len(holdout['eval_ids'])} ejemplos del eval set están")
        print("  en el corpus que alimenta el entrenamiento. DEBEN excluirse del train:")
        print("  usar eval_corpus_ids() desde el código de entrenamiento.")
    else:
        print("  El eval set no se solapa con el corpus de entrenamiento.")


def main() -> int:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=here / "eval_set.jsonl")
    parser.add_argument("--rubric", type=Path, default=here / "rubric.yaml")
    parser.add_argument(
        "--lexicon", type=Path, default=here / ".." / "data_collection" / "lexicon.yaml"
    )
    parser.add_argument(
        "--corpus", type=Path, default=here / ".." / "data" / "processed" / "weak_labeled.jsonl"
    )
    parser.add_argument("--systems", nargs="*", default=["mayoritaria", "lexico"])
    parser.add_argument("--adapter", type=Path, default=None, help="dir del adaptador LoRA")
    parser.add_argument("--base-model", default="dccuchile/bert-base-spanish-wwm-cased")
    parser.add_argument("--json", type=Path, default=None, help="guardar scorecard en JSON")
    args = parser.parse_args()

    with args.eval_set.open(encoding="utf-8") as f:
        eval_records = [json.loads(line) for line in f if line.strip()]
    rubric = yaml.safe_load(args.rubric.read_text(encoding="utf-8"))

    predictores: list[Predictor] = []
    if "mayoritaria" in args.systems:
        predictores.append(MajorityPredictor())
    if "lexico" in args.systems:
        predictores.append(LexiconPredictor(args.lexicon))
    if args.adapter:
        predictores.append(FineTunedPredictor(args.adapter, args.lexicon, args.base_model))

    if not predictores:
        raise SystemExit("No hay sistemas que evaluar (--systems / --adapter).")

    for p in predictores:
        for aviso in getattr(p, "avisos", []):
            print(f"AVISO [{p.name}]: {aviso}")
            print()

    resultados = {}
    for p in predictores:
        rows = evaluar(p, eval_records)
        resultados[p.name] = {
            "dim1": dim1_metrica_clasica(
                [r["gold_cat"] for r in rows], [r["pred_cat"] for r in rows]
            ),
            "dim2": score_rubric(rows, rubric),
            "dim3": dim3_dominio(rows),
            "predicciones": rows,
        }

    holdout = check_holdout(eval_records, args.corpus)
    imprimir_scorecard(resultados, eval_records, holdout)

    if args.json:
        payload = {
            "eval_set": str(args.eval_set.name),
            "n_ejemplos": len(eval_records),
            "rubric_version": rubric["meta"]["version"],
            "sistemas": {
                s: {"dim1": r["dim1"], "dim2": r["dim2"], "dim3": r["dim3"]}
                for s, r in resultados.items()
            },
            "holdout": holdout,
        }
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nScorecard guardado en {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
