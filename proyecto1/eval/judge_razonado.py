"""Juez de la dimensión 2, versión 2: puede razonar antes de puntuar y ve el texto.

Sale del feedback del M2: el juez v1 casi no separaba respuestas buenas de malas,
y el profesor sugirió dejarlo razonar antes de leer los logits. Con razonar=True:

  1. el modelo escribe un ANÁLISIS corto (greedy, determinista),
  2. se añade "Puntaje (1-5):" al final de ese análisis,
  3. se leen los logits de los tokens "1".."5" en esa posición, como en la v1.

El puntaje nunca se parsea de texto: solo cambia lo que el modelo "pensó" antes de
la posición donde se leen los logits. Con razonar=False es un juez de logits
directo, como la v1, pero con la rúbrica v2 (ve el texto de la respuesta).

La rúbrica es judge_rubric_v2.yaml: mismas anclas que la v1.

    python judge_razonado.py                                  # 1.5B, razonando
    python judge_razonado.py --modelo Qwen/Qwen2.5-3B-Instruct --4bit --sin-razonar
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

MODELO_JUEZ = "Qwen/Qwen2.5-1.5B-Instruct"
# Commits fijos en Hugging Face (el de 1.5B es el mismo que fija rag/config.py).
REVISIONES = {
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
}
NIVELES = [1, 2, 3, 4, 5]
MAX_TOKENS_ANALISIS = 110
INSTRUCCION_RAZONAR = (
    "Primero escribe un ANÁLISIS de dos o tres frases. Después, en una línea aparte, el puntaje."
)


@dataclass
class ScoreRazonado:
    nivel: int
    probs: dict[int, float]
    analisis: str

    @property
    def confianza(self) -> float:
        return self.probs[self.nivel]


def prompt_sistema(rubrica: dict, razonar: bool) -> str:
    lineas = [
        "Eres un evaluador de un sistema que responde a apostadores sobre noticias de futbol.",
        "",
        "Recibes una PREGUNTA (o un fragmento de prensa), el CRITERIO que dice cual es la",
        "respuesta correcta, y lo que respondio el sistema: una SEÑAL (categoria, impacto,",
        "equipo) y, si la hay, una RESPUESTA en texto. Puntua de 1 a 5.",
        "",
        "Compara: (1) del CRITERIO, la categoria y la direccion del impacto; (2) de la",
        "SEÑAL, lo mismo; (3) si hay texto, si afirma algo que el criterio contradice.",
        "",
        f"PRINCIPIO: {' '.join(rubrica['principio'].split())}",
        "",
        "ESCALA:",
    ]
    for ancla in sorted(rubrica["anclas"], key=lambda a: a["nivel"]):
        desc = " ".join(ancla["descripcion"].split())
        lineas.append(f"  {ancla['nivel']} = {ancla['nombre']}. {desc}")
    reglas = [
        r if (razonar or r != INSTRUCCION_RAZONAR) else "Responde solo con el puntaje."
        for r in rubrica["instrucciones_al_juez"]
    ]
    lineas += ["", "REGLAS:"] + [f"  - {r}" for r in reglas]
    return "\n".join(lineas)


def prompt_usuario(fragmento: str, criterio: str, senal: str) -> str:
    return (
        f"PREGUNTA O FRAGMENTO:\n{fragmento}\n\n"
        f"CRITERIO:\n{criterio}\n\n"
        f"LO QUE RESPONDIO EL SISTEMA:\n{senal}"
    )


def few_shot(rubrica: dict, razonar: bool) -> list[dict]:
    mensajes = []
    for ancla in sorted(rubrica["anclas"], key=lambda a: a["nivel"]):
        d = ancla["demo"]
        senal = f"{d['senal']}\nRESPUESTA: {d['respuesta']}"
        mensajes.append(
            {"role": "user", "content": prompt_usuario(d["fragmento"], d["criterio"], senal)}
        )
        puntaje = f"Puntaje (1-5): {ancla['nivel']}"
        mensajes.append(
            {
                "role": "assistant",
                "content": f"ANALISIS: {d['razonamiento']}\n{puntaje}" if razonar else puntaje,
            }
        )
    return mensajes


class JudgeRazonado:
    def __init__(
        self,
        rubrica_path: Path | None = None,
        modelo: str = MODELO_JUEZ,
        device: str | None = None,
        razonar: bool = True,
        cuantizar_4bit: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        here = Path(__file__).parent
        self.rubrica = yaml.safe_load(
            (rubrica_path or here / "judge_rubric_v2.yaml").read_text(encoding="utf-8")
        )
        self.version_rubrica = self.rubrica["version"]
        self.modelo_id = modelo
        self.razonar = razonar
        self.cuantizado = cuantizar_4bit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        rev = REVISIONES.get(modelo)
        self.tokenizer = AutoTokenizer.from_pretrained(modelo, revision=rev)
        if cuantizar_4bit:
            # Para que un 3B quepa en una GPU de 4 GB. En una T4 cabe en fp16 sin esto.
            from transformers import BitsAndBytesConfig

            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.model = AutoModelForCausalLM.from_pretrained(
                modelo, quantization_config=q, revision=rev, device_map=self.device
            ).eval()
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = (
                AutoModelForCausalLM.from_pretrained(modelo, dtype=dtype, revision=rev)
                .to(self.device)
                .eval()
            )
        self.sistema = prompt_sistema(self.rubrica, razonar)
        self.demos = few_shot(self.rubrica, razonar)
        self.ids_nivel = [
            self.tokenizer.encode(str(n), add_special_tokens=False)[0] for n in NIVELES
        ]

    @property
    def nombre(self) -> str:
        corto = self.modelo_id.split("/")[-1]
        return (
            f"{corto}{' 4bit' if self.cuantizado else ''} {'razona' if self.razonar else 'directo'}"
        )

    def score(self, fragmento: str, criterio: str, senal: str) -> ScoreRazonado:
        mensajes = [
            {"role": "system", "content": self.sistema},
            *self.demos,
            {"role": "user", "content": prompt_usuario(fragmento, criterio, senal)},
        ]
        base = self.tokenizer.apply_chat_template(
            mensajes, tokenize=False, add_generation_prompt=True
        )
        if not self.razonar:
            return self._puntuar(base, "")
        # 1 · el análisis, greedy. Se corta antes de que el modelo escriba el puntaje.
        entradas = self.tokenizer(base + "ANALISIS:", return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **entradas,
                max_new_tokens=MAX_TOKENS_ANALISIS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        analisis = self.tokenizer.decode(
            out[0, entradas["input_ids"].shape[1] :], skip_special_tokens=True
        )
        analisis = analisis.split("Puntaje")[0].strip()
        return self._puntuar(base + f"ANALISIS: {analisis}\n", analisis)

    def _puntuar(self, prefijo: str, analisis: str) -> ScoreRazonado:
        """Los logits de "1".."5" justo después de "Puntaje (1-5):"."""
        entradas = self.tokenizer(prefijo + "Puntaje (1-5): ", return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.model(**entradas).logits[0, -1]
        probs = self.torch.softmax(logits[self.ids_nivel].float(), dim=-1).tolist()
        nivel = NIVELES[max(range(5), key=lambda i: probs[i])]
        return ScoreRazonado(nivel=nivel, probs=dict(zip(NIVELES, probs)), analisis=analisis)

    # ------------------------------------------------------------------
    def sanity_check(self) -> dict:
        """El sanity check de la v1 (mismo caso, mismas cinco señales), dos veces:

        - con el criterio completo, que nombra la categoría y el impacto correctos;
        - con un criterio CIEGO, que describe qué hay que mirar pero no lo etiqueta.

        Si el juez solo acierta con el criterio completo, está comparando etiquetas
        que le soplamos, no leyendo el caso.
        """
        fragmento = (
            "El Getafe comunica una grave lesión en la rodilla derecha de Uche. "
            "El club confirma que el nigeriano se perderá toda la temporada"
        )
        criterios = {
            "completo": (
                "Debe leer el hecho que hay dentro de la declaración: el jugador se pierde toda "
                "la temporada, así que es una baja confirmada de impacto negativo alto para el "
                "Getafe."
            ),
            "ciego": (
                "Una buena respuesta tiene que reflejar lo que el club confirma sobre Uche y lo "
                "que eso significa para el Getafe en los próximos partidos."
            ),
        }
        casos = {
            "correcta": ("baja_confirmada | impacto: negativo_alto | equipo: Getafe", 5),
            "equipo_incorrecto": (
                "baja_confirmada | impacto: negativo_alto | equipo: Rayo Vallecano",
                4,
            ),
            "categoria_vecina": ("sancion_suspension | impacto: negativo_alto | equipo: Getafe", 3),
            "sin_senal": ("irrelevante | impacto: neutro | equipo: ninguno", 3),
            "signo_invertido": ("regreso_alta | impacto: positivo_alto | equipo: Getafe", 1),
        }
        salida = {"juez": self.nombre, "niveles_esperados": {k: v for k, (_, v) in casos.items()}}
        for nombre_crit, criterio in criterios.items():
            obtenidos, confianzas, analisis = {}, {}, {}
            for nombre, (senal, _) in casos.items():
                r = self.score(fragmento, criterio, senal)
                obtenidos[nombre], confianzas[nombre] = r.nivel, round(r.confianza, 3)
                analisis[nombre] = r.analisis
            salida[nombre_crit] = {
                "niveles_obtenidos": obtenidos,
                "confianza": confianzas,
                "analisis": analisis,
                "aciertos_exactos": sum(obtenidos[k] == v for k, (_, v) in casos.items()),
                "separa_util_de_inutil": obtenidos["correcta"] > obtenidos["sin_senal"],
                "detecta_signo_invertido": obtenidos["signo_invertido"] <= 2,
            }
        return salida


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", default=MODELO_JUEZ)
    parser.add_argument("--sin-razonar", action="store_true")
    parser.add_argument("--4bit", dest="cuantizar", action="store_true")
    args = parser.parse_args()
    juez = JudgeRazonado(
        modelo=args.modelo, razonar=not args.sin_razonar, cuantizar_4bit=args.cuantizar
    )
    print(f"Juez v2: {juez.nombre} | rúbrica v{juez.version_rubrica} | {juez.device}")
    res = juez.sanity_check()
    esperados = res["niveles_esperados"]
    for crit in ("completo", "ciego"):
        r = res[crit]
        print(f"\nCriterio {crit}: {r['aciertos_exactos']}/5 exactos")
        for caso, nivel in r["niveles_obtenidos"].items():
            print(
                f"  {caso:<18} -> {nivel} (esperado {esperados[caso]}) conf {r['confianza'][caso]:.2f}"
            )
            if r["analisis"][caso]:
                print(f"     {r['analisis'][caso][:150]}")
    out = Path(__file__).parent / "sanity_juez_v2.json"
    previos = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if "juez" in previos:  # formato antiguo, de una sola corrida
        previos = {}
    previos[juez.nombre] = res
    out.write_text(json.dumps(previos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
