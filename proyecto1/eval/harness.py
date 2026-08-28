"""Harness de evaluación de Offside — tres dimensiones y un scorecard.

    harness(eval_set, sistema) -> Scorecard

El sistema evaluado es cualquier callable que respete el contrato de abajo. Hoy
lo cumplen la clase mayoritaria, el léxico y el modelo LoRA de M1; en M3 lo
cumplirá el RAG sin tocar este archivo.

Las tres dimensiones
--------------------
1. MÉTRICA CLÁSICA — F1 macro sobre la categoría. Automática, barata y
   reproducible, pero **ciega a la gravedad**: para F1, confundir una baja con
   una sanción cuesta exactamente lo mismo que decir que un jugador vuelve
   cuando en realidad lo sancionaron. Para nuestro usuario no es lo mismo.

2. JUEZ LLM — Qwen2.5-1.5B-Instruct puntúa 1-5 con las anclas de
   `judge_rubric.yaml`. Aporta lo que le falta a la dimensión 1: distingue un
   error vecino e inofensivo de uno que engaña. Su punto ciego está medido en
   `Judge.sanity_check()`.

3. DOMINIO — tasa de señal accionable: qué fracción de los ejemplos pasa TODAS
   las verificaciones duras de nuestro dominio. No depende del juez, y por eso
   cubre justo lo que el juez no ve.

Por qué las tres y no una
-------------------------
Cada una tapa el hueco de la otra. La 1 es objetiva pero no pesa el daño; la 2
pesa el daño pero es un modelo pequeño y falible; la 3 es determinista e
implacable con los dos errores que no nos podemos permitir. El scorecard las
muestra juntas a propósito: leer una sola da una imagen equivocada.

Uso:
    python harness.py                                  # baselines
    python harness.py --adapter ../m1_lora_adapter_holdout
    python harness.py --sin-juez                       # solo dims 1 y 3, sin descargar el LLM
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
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

# Categorías que para el usuario significan lo mismo ("ese jugador no está
# disponible"). Confundirlas entre sí es un error vecino, no uno que engañe.
FAMILIAS = [
    {"baja_confirmada", "sancion_suspension"},
    {"regreso_alta"},
    {"duda_fisica", "rumor_no_confirmado"},
    {"cambio_tactico", "declaracion_contexto"},
    {"irrelevante"},
]

# ---------------------------------------------------------------------------
# Contrato del sistema evaluado
#
#   sistema(input: dict) -> dict
#     input     {"text": str, "source": str, "published_at": str | None}
#     respuesta {"category": str, "impact": str, "team": str}
#
# Cualquier cosa que cumpla esto se puede pasar por el harness. Es lo que
# permite que en M3 el RAG entre sin modificar este archivo.
# ---------------------------------------------------------------------------
Sistema = Callable[[dict], dict]


def render_senal(respuesta: dict) -> str:
    """Renderiza la respuesta con una plantilla de longitud FIJA.

    Esto no es cosmético: es la mitigación del sesgo de verbosidad del juez.
    Si cada sistema pudiera redactar su señal a su manera, el juez premiaría al
    que escribe más largo, no al que acierta. Con una plantilla fija la longitud
    no lleva información y deja de ser una variable. `bias_check.py` mide que la
    mitigación funcione.
    """
    return (
        f"{respuesta['category']} | impacto: {respuesta['impact']} | "
        f"equipo: {respuesta.get('team') or 'ninguno'}"
    )


# ---------------------------------------------------------------------------
# Utilidades de impacto
# ---------------------------------------------------------------------------
def signo(impacto: str) -> str:
    if impacto.startswith("negativo"):
        return "negativo"
    if impacto.startswith("positivo"):
        return "positivo"
    return "neutro"


def es_alto_impacto(impacto: str) -> bool:
    return impacto in ("negativo_alto", "positivo_alto")


def misma_familia(a: str, b: str) -> bool:
    return any(a in fam and b in fam for fam in FAMILIAS)


# ---------------------------------------------------------------------------
# Sistemas evaluables
# ---------------------------------------------------------------------------
def sistema_mayoritaria(entrada: dict) -> dict:
    """Responde siempre la clase mayoritaria. El piso absoluto."""
    return {"category": "irrelevante", "impact": "neutro", "team": "ninguno"}


class SistemaLexico:
    """El léxico/regex de `lexicon.yaml`: el baseline que queremos superar."""

    def __init__(self, lexicon_path: Path):
        cfg = yaml.safe_load(lexicon_path.read_text(encoding="utf-8"))
        self.reglas = [
            (c["name"], c["impact_default"], [re.compile(p, re.I) for p in c["patterns"]])
            for c in cfg["categories"]
        ]
        self.boost = {
            label: [re.compile(p, re.I) for p in pats]
            for label, pats in cfg.get("sentiment_boost", {}).items()
        }

    def __call__(self, entrada: dict) -> dict:
        texto = entrada["text"]
        for nombre, impacto_def, patrones in self.reglas:
            if any(p.search(texto) for p in patrones):
                impacto = impacto_def
                for boosted, pats in self.boost.items():
                    if any(p.search(texto) for p in pats):
                        impacto = boosted
                        break
                return {"category": nombre, "impact": impacto, "team": "desconocido"}
        return {"category": "irrelevante", "impact": "neutro", "team": "ninguno"}


class SistemaLoRA:
    """BETO + adaptador LoRA de M1. El impacto se deriva de la categoría."""

    def __init__(self, adapter_dir: Path, lexicon_path: Path, base_model: str):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.avisos = self._revisar_adaptador(adapter_dir)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=len(CATEGORIES)
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_dir))
        self.model.eval()

        cfg = yaml.safe_load(lexicon_path.read_text(encoding="utf-8"))
        self.cat_impacto = {c["name"]: c["impact_default"] for c in cfg["categories"]}
        self.cat_impacto["irrelevante"] = "neutro"

    @staticmethod
    def _revisar_adaptador(adapter_dir: Path) -> list[str]:
        """El adaptador debe bastar para reproducir el modelo entrenado.

        BETO es un checkpoint de MLM: no trae `bert.pooler`, así que se
        inicializa al azar en cada carga. Si el adaptador no lo guarda, las
        predicciones cambian entre procesos — un fallo silencioso que parece
        "el modelo es malo". Lo encontramos midiendo dos veces el mismo número.
        """
        cfg_path = adapter_dir / "adapter_config.json"
        if not cfg_path.exists():
            return [f"no encuentro {cfg_path}"]
        guardados = json.loads(cfg_path.read_text(encoding="utf-8")).get("modules_to_save") or []
        if not any("pooler" in m for m in guardados):
            return [
                "el adaptador NO guarda `pooler`: se inicializa al azar en cada carga, así que "
                "estas predicciones no son reproducibles. Reentrena con "
                "modules_to_save=['classifier', 'pooler'] (ver train_holdout_model.py)."
            ]
        return []

    def __call__(self, entrada: dict) -> dict:
        inputs = self.tokenizer(
            entrada["text"], return_tensors="pt", truncation=True, max_length=128
        )
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        categoria = CATEGORIES[int(logits.argmax(-1))]
        return {
            "category": categoria,
            "impact": self.cat_impacto[categoria],
            "team": "desconocido",
        }


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
@dataclass
class Scorecard:
    sistema: str
    n_ejemplos: int
    d1_f1_macro: float
    d1_accuracy: float
    d2_juez_media: float | None
    d2_juez_distribucion: dict[int, int]
    d3_tasa_accionable: float
    d3_tasa_accionable_estricta: float | None
    d3_detalle: dict[str, int]
    adversariales: dict[str, float]
    normales: dict[str, float]
    filas: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def como_fila_csv(self) -> dict:
        return {
            "sistema": self.sistema,
            "n_ejemplos": self.n_ejemplos,
            "d1_f1_macro": round(self.d1_f1_macro, 4),
            "d1_accuracy": round(self.d1_accuracy, 4),
            "d2_juez_media": "" if self.d2_juez_media is None else round(self.d2_juez_media, 3),
            "d3_tasa_accionable": round(self.d3_tasa_accionable, 4),
            "d3_tasa_accionable_estricta": (
                ""
                if self.d3_tasa_accionable_estricta is None
                else round(self.d3_tasa_accionable_estricta, 4)
            ),
            "d3_senal_falsa_alto_impacto": self.d3_detalle["senal_falsa_alto_impacto"],
            "d3_signo_invertido": self.d3_detalle["signo_invertido"],
            "d3_rumor_como_hecho": self.d3_detalle["rumor_como_hecho"],
            "f1_macro_adversariales": round(self.adversariales["f1_macro"], 4),
            "f1_macro_normales": round(self.normales["f1_macro"], 4),
            "tasa_accionable_adversariales": round(self.adversariales["tasa_accionable"], 4),
            "tasa_accionable_normales": round(self.normales["tasa_accionable"], 4),
        }


def _f1(golds: list[str], preds: list[str]) -> float:
    if not golds:
        return float("nan")
    return f1_score(golds, preds, average="macro", labels=CATEGORIES, zero_division=0)


def _tasa(filas: list[dict]) -> float:
    return sum(f["accionable"] for f in filas) / len(filas) if filas else float("nan")


def harness(
    eval_set: list[dict],
    sistema: Sistema,
    nombre: str = "sistema",
    juez=None,
) -> Scorecard:
    """Corre las tres dimensiones sobre `eval_set` y devuelve el scorecard.

    `juez` es opcional: sin él se calculan las dimensiones 1 y 3, que no
    necesitan descargar ningún LLM. Es lo que permite correr el harness en
    cualquier laptop y dejar el juez para Colab.
    """
    filas = []
    for ej in eval_set:
        respuesta = sistema(ej["input"])
        gold = ej["esperado"]
        senal = render_senal(respuesta)

        # --- verificaciones duras de dominio (dimensión 3) ---
        signo_invertido = signo(respuesta["impact"]) != signo(gold["impact"]) and "neutro" not in (
            signo(respuesta["impact"]),
            signo(gold["impact"]),
        )
        senal_falsa = es_alto_impacto(respuesta["impact"]) and not es_alto_impacto(gold["impact"])
        rumor_como_hecho = (
            gold["category"] == "rumor_no_confirmado"
            and respuesta["category"] in HECHOS_CONFIRMADOS
        )
        familia_ok = misma_familia(respuesta["category"], gold["category"])

        fila = {
            "eval_id": ej["eval_id"],
            "adversarial": ej["adversarial"],
            "gold_category": gold["category"],
            "gold_impact": gold["impact"],
            "pred_category": respuesta["category"],
            "pred_impact": respuesta["impact"],
            "senal": senal,
            "signo_invertido": signo_invertido,
            "senal_falsa_alto_impacto": senal_falsa,
            "rumor_como_hecho": rumor_como_hecho,
            "familia_ok": familia_ok,
        }

        if juez is not None:
            res = juez.score(ej["input"]["text"], ej["criterio"], senal)
            fila["juez_nivel"] = res.nivel
            fila["juez_confianza"] = round(res.confianza, 3)

        # Una señal es ACCIONABLE si supera las tres verificaciones duras y
        # acierta al menos la familia de la categoría. DELIBERADAMENTE no
        # depende del juez.
        #
        # El módulo sugiere "juez >= 4" como posible criterio de dominio, y lo
        # probamos: nuestro juez concentra dos tercios de sus notas en el 3 y no
        # detecta la inversión de signo. Atarle la dimensión 3 le importaría ese
        # punto ciego justo a la dimensión que existe para cubrirlo, y las tres
        # dejarían de ser miradas independientes. Se calcula igualmente como
        # `accionable_estricto` para que se vea el criterio alternativo.
        accionable = not signo_invertido and not senal_falsa and not rumor_como_hecho and familia_ok
        fila["accionable"] = accionable
        if juez is not None:
            fila["accionable_estricto"] = accionable and fila["juez_nivel"] >= 4
        filas.append(fila)

    golds = [f["gold_category"] for f in filas]
    preds = [f["pred_category"] for f in filas]
    adv = [f for f in filas if f["adversarial"]]
    nor = [f for f in filas if not f["adversarial"]]
    niveles = [f["juez_nivel"] for f in filas if "juez_nivel" in f]

    return Scorecard(
        sistema=nombre,
        n_ejemplos=len(filas),
        d1_f1_macro=_f1(golds, preds),
        d1_accuracy=accuracy_score(golds, preds),
        d2_juez_media=(sum(niveles) / len(niveles)) if niveles else None,
        d2_juez_distribucion={n: niveles.count(n) for n in range(1, 6)} if niveles else {},
        d3_tasa_accionable=_tasa(filas),
        d3_tasa_accionable_estricta=(
            sum(f["accionable_estricto"] for f in filas) / len(filas)
            if filas and "accionable_estricto" in filas[0]
            else None
        ),
        d3_detalle={
            "senal_falsa_alto_impacto": sum(f["senal_falsa_alto_impacto"] for f in filas),
            "signo_invertido": sum(f["signo_invertido"] for f in filas),
            "rumor_como_hecho": sum(f["rumor_como_hecho"] for f in filas),
        },
        adversariales={
            "n": len(adv),
            "f1_macro": _f1([f["gold_category"] for f in adv], [f["pred_category"] for f in adv]),
            "tasa_accionable": _tasa(adv),
        },
        normales={
            "n": len(nor),
            "f1_macro": _f1([f["gold_category"] for f in nor], [f["pred_category"] for f in nor]),
            "tasa_accionable": _tasa(nor),
        },
        filas=filas,
    )


# ---------------------------------------------------------------------------
def eval_corpus_ids(eval_set_path: Path) -> set[str]:
    """Ids del corpus que forman el eval set, para excluirlos del entrenamiento."""
    datos = json.loads(eval_set_path.read_text(encoding="utf-8"))
    return {r["procedencia"]["corpus_id"] for r in datos if r["procedencia"].get("corpus_id")}


def diagnostico_juez(tarjetas: list[Scorecard]) -> dict | None:
    """¿Cuánto poder discriminante tiene el juez sobre ESTE eval set?

    Una nota media del juez no dice nada por sí sola: hay que saber si el juez
    premia acertar. Esto compara su nota cuando la categoría predicha era
    correcta contra cuando no lo era, agrupando todos los sistemas, y cuenta en
    cuántos ejemplos le da la MISMA nota a una respuesta correcta y a una
    incorrecta — que es la forma más directa de ver dónde deja de distinguir.
    """
    todas = [f for t in tarjetas for f in t.filas if "juez_nivel" in f]
    if not todas:
        return None
    ok = [f["juez_nivel"] for f in todas if f["pred_category"] == f["gold_category"]]
    mal = [f["juez_nivel"] for f in todas if f["pred_category"] != f["gold_category"]]
    por_ejemplo: dict[str, list[tuple[bool, int]]] = {}
    for f in todas:
        por_ejemplo.setdefault(f["eval_id"], []).append(
            (f["pred_category"] == f["gold_category"], f["juez_nivel"])
        )
    empates = [
        k
        for k, v in por_ejemplo.items()
        if len({c for c, _ in v}) > 1 and len({n for _, n in v}) == 1
    ]
    return {
        "n_correctas": len(ok),
        "n_incorrectas": len(mal),
        "media_correctas": (sum(ok) / len(ok)) if ok else None,
        "media_incorrectas": (sum(mal) / len(mal)) if mal else None,
        "delta": ((sum(ok) / len(ok)) - (sum(mal) / len(mal))) if ok and mal else None,
        "ejemplos_indistinguibles": empates,
        "n_ejemplos_indistinguibles": len(empates),
        "n_ejemplos": len(por_ejemplo),
    }


def imprimir_scorecard(
    tarjetas: list[Scorecard], eval_set: list[dict], sanity: dict | None
) -> None:
    ancho = max(len(t.sistema) for t in tarjetas) + 2
    n_adv = sum(1 for e in eval_set if e["adversarial"])

    print("=" * 78)
    print("SCORECARD · Offside")
    print("=" * 78)
    print(
        f"Eval set: {len(eval_set)} ejemplos "
        f"({n_adv} adversariales / borde, {100 * n_adv / len(eval_set):.0f}%)"
    )

    print("\n" + "-" * 78)
    print("DIMENSIÓN 1 · MÉTRICA CLÁSICA (automática)")
    print("-" * 78)
    print(f"{'sistema':<{ancho}}{'F1 macro':>11}{'accuracy':>11}")
    for t in tarjetas:
        print(f"{t.sistema:<{ancho}}{t.d1_f1_macro:>11.4f}{t.d1_accuracy:>11.4f}")
    print("\n  accuracy va solo como contraste: sube con la clase mayoritaria.")

    if any(t.d2_juez_media is not None for t in tarjetas):
        print("\n" + "-" * 78)
        print("DIMENSIÓN 2 · LLM-AS-A-JUDGE (rúbrica 1-5 anclada)")
        print("-" * 78)
        print(f"{'sistema':<{ancho}}{'media':>8}   distribución 1..5")
        for t in tarjetas:
            if t.d2_juez_media is None:
                continue
            dist = " ".join(f"{n}:{t.d2_juez_distribucion.get(n, 0)}" for n in range(1, 6))
            print(f"{t.sistema:<{ancho}}{t.d2_juez_media:>8.2f}   {dist}")
        if sanity:
            print(
                f"\n  El juez separa útil de inútil: "
                f"{'SÍ' if sanity['separa_util_de_inutil'] else 'NO'}."
            )
            print(
                f"  Detecta el signo invertido:    "
                f"{'SÍ' if sanity['detecta_signo_invertido'] else 'NO — punto ciego medido'}."
            )
            print("  Por eso el signo invertido lo verifica la dimensión 3, no el juez.")
        diag = diagnostico_juez(tarjetas)
        if diag and diag["delta"] is not None:
            print()
            print(
                f"  Poder discriminante sobre este eval set: nota media "
                f"{diag['media_correctas']:.2f} cuando la categoría era correcta contra "
                f"{diag['media_incorrectas']:.2f} cuando no lo era "
                f"(delta {diag['delta']:+.2f})."
            )
            print(
                f"  En {diag['n_ejemplos_indistinguibles']} de {diag['n_ejemplos']} ejemplos le da "
                f"la MISMA nota a una respuesta correcta y a una incorrecta."
            )

    print("\n" + "-" * 78)
    print("DIMENSIÓN 3 · DOMINIO (tasa de señal accionable)")
    print("-" * 78)
    print(
        f"{'sistema':<{ancho}}{'tasa':>8}{'falsa':>8}{'signo':>8}{'rumor':>8}"
        f"{'  adversar.':>12}{'normales':>10}"
    )
    for t in tarjetas:
        d = t.d3_detalle
        print(
            f"{t.sistema:<{ancho}}{t.d3_tasa_accionable:>8.2f}"
            f"{d['senal_falsa_alto_impacto']:>8}{d['signo_invertido']:>8}{d['rumor_como_hecho']:>8}"
            f"{t.adversariales['tasa_accionable']:>12.2f}{t.normales['tasa_accionable']:>10.2f}"
        )
    print("\n  falsa = señales de alto impacto inventadas · signo = impacto invertido")
    print("  rumor = un rumor presentado como hecho confirmado")
    print("  Las tres son las verificaciones duras; ninguna depende del juez.")


def main() -> int:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=here / "eval_set.json")
    parser.add_argument(
        "--lexicon", type=Path, default=here / ".." / "data_collection" / "lexicon.yaml"
    )
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--base-model", default="dccuchile/bert-base-spanish-wwm-cased")
    parser.add_argument("--sin-juez", action="store_true", help="omite la dimensión 2")
    parser.add_argument("--csv", type=Path, default=here / "scorecard_baseline.csv")
    parser.add_argument("--json", type=Path, default=here / "scorecard_baseline.json")
    args = parser.parse_args()

    eval_set = json.loads(args.eval_set.read_text(encoding="utf-8"))

    sistemas: list[tuple[str, Sistema]] = [
        ("mayoritaria", sistema_mayoritaria),
        ("lexico", SistemaLexico(args.lexicon)),
    ]
    if args.adapter:
        sistemas.append(
            ("lora_finetuned", SistemaLoRA(args.adapter, args.lexicon, args.base_model))
        )

    juez, sanity = None, None
    if not args.sin_juez:
        from judge import Judge

        juez = Judge()
        print(f"Juez: {juez.modelo_id} | rúbrica v{juez.version_rubrica} | device={juez.device}")
        print("Corriendo sanity check del juez...", flush=True)
        sanity = juez.sanity_check()

    tarjetas = []
    for nombre, sistema in sistemas:
        for aviso in getattr(sistema, "avisos", []):
            print(f"AVISO [{nombre}]: {aviso}\n")
        print(f"Evaluando {nombre}...", flush=True)
        tarjetas.append(harness(eval_set, sistema, nombre, juez))

    imprimir_scorecard(tarjetas, eval_set, sanity)

    with args.csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tarjetas[0].como_fila_csv().keys()))
        w.writeheader()
        for t in tarjetas:
            w.writerow(t.como_fila_csv())
    print(f"\nScorecard -> {args.csv}")

    payload = {
        "eval_set": args.eval_set.name,
        "n_ejemplos": len(eval_set),
        "juez": None
        if juez is None
        else {
            "modelo": juez.modelo_id,
            "rubrica_version": juez.version_rubrica,
            "sanity_check": sanity,
            "poder_discriminante": diagnostico_juez(tarjetas),
        },
        "sistemas": {
            t.sistema: {
                "d1": {"f1_macro": t.d1_f1_macro, "accuracy": t.d1_accuracy},
                "d2": {"media": t.d2_juez_media, "distribucion": t.d2_juez_distribucion},
                "d3": {
                    "tasa_accionable": t.d3_tasa_accionable,
                    "tasa_accionable_estricta": t.d3_tasa_accionable_estricta,
                    "detalle": t.d3_detalle,
                },
                "adversariales": t.adversariales,
                "normales": t.normales,
                "filas": t.filas,
            }
            for t in tarjetas
        },
    }
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detalle    -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
