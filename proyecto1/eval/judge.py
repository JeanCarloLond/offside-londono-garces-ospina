"""LLM-as-a-judge: dimensión 2 del harness.

Un LLM instruct pequeño y abierto (Qwen2.5-1.5B-Instruct por defecto) puntúa de
1 a 5 la señal que el sistema le entrega al usuario, usando las anclas de
`judge_rubric.yaml`.

Dos decisiones que conviene entender
------------------------------------
**El juez no ve la etiqueta gold.** Recibe el fragmento, el `criterio` de ese
ejemplo en prosa y la señal emitida. Si le diéramos la etiqueta correcta, el
juez se convertiría en un comparador de strings y no aportaría nada por encima
de la dimensión 1.

**El puntaje se lee de los logits, no del texto generado.** En vez de generar
texto y buscarle un número con una expresión regular —que falla cuando el
modelo contesta "Puntaje: 4/5" o se pone a explicar—, comparamos directamente
los logits de los tokens `1`..`5` en la primera posición de la respuesta y nos
quedamos con el mayor. Es determinista, no necesita `do_sample`, no puede
fallar el parseo y además deja una distribución de probabilidad sobre los cinco
niveles que sirve para ver cuán segura fue la decisión.

Uso:
    from judge import Judge
    juez = Judge()
    juez.score(fragmento, criterio, senal)   # -> ScoreJuez(nivel=4, probs=[...])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

MODELO_JUEZ = "Qwen/Qwen2.5-1.5B-Instruct"
NIVELES = [1, 2, 3, 4, 5]


@dataclass
class ScoreJuez:
    nivel: int
    probs: dict[int, float]

    @property
    def confianza(self) -> float:
        """Probabilidad del nivel elegido. Baja = el juez dudó entre anclas."""
        return self.probs[self.nivel]


def construir_prompt_sistema(rubrica: dict) -> str:
    """El prompt del sistema se genera DESDE el YAML, no se escribe a mano.

    Así es imposible que la rúbrica versionada y la que ve el juez se
    desincronicen: si alguien edita una ancla, el prompt cambia con ella.

    Las anclas van en orden ASCENDENTE (1 -> 5) a propósito. Con el orden
    descendente medimos que el juez se quedaba anclado en el 5 y puntuaba alto
    casi todo, incluidas señales con el signo invertido.
    """
    lineas = [
        "Eres un evaluador de un sistema que avisa a apostadores sobre noticias de futbol.",
        "",
        "Recibes un FRAGMENTO de prensa, el CRITERIO que dice cual es la respuesta",
        "correcta, y la SEÑAL que emitio el sistema. Puntua la señal de 1 a 5.",
        "",
        "Procedimiento obligatorio:",
        "  1. Del CRITERIO, anota la categoria correcta y la direccion del impacto",
        "     (negativo / neutro / positivo).",
        "  2. De la SEÑAL, anota la categoria emitida y su direccion.",
        "  3. Compara. La DIRECCION del impacto pesa mas que todo lo demas.",
        "",
        f"PRINCIPIO: {' '.join(rubrica['principio'].split())}",
        "",
        "ESCALA:",
    ]
    for ancla in sorted(rubrica["anclas"], key=lambda a: a["nivel"]):
        desc = " ".join(ancla["descripcion"].split())
        lineas.append(f"  {ancla['nivel']} = {ancla['nombre']}. {desc}")
    lineas.append("")
    lineas.append("REGLAS:")
    for regla in rubrica["instrucciones_al_juez"]:
        lineas.append(f"  - {regla}")
    return "\n".join(lineas)


def construir_few_shot(rubrica: dict) -> list[dict]:
    """Turnos de ejemplo, uno por nivel, tomados de los `demo` de la rúbrica.

    Un modelo de 1.5B no ejecuta el procedimiento solo con la descripción de las
    anclas: lo medimos y puntuaba 5 a señales con el signo invertido. Con un
    ejemplo resuelto por nivel sí lo hace.

    Los demos usan clubes FICTICIOS y no salen del eval set, así que el juez no
    ve ninguna de las respuestas que va a tener que puntuar.
    """
    mensajes = []
    for ancla in sorted(rubrica["anclas"], key=lambda a: a["nivel"]):
        demo = ancla.get("demo")
        if not demo:
            continue
        mensajes.append(
            {
                "role": "user",
                "content": construir_prompt_usuario(
                    demo["fragmento"], demo["criterio"], demo["senal"]
                ),
            }
        )
        mensajes.append({"role": "assistant", "content": str(ancla["nivel"])})
    return mensajes


def construir_prompt_usuario(fragmento: str, criterio: str, senal: str) -> str:
    return (
        f"FRAGMENTO DE PRENSA:\n{fragmento}\n\n"
        f"QUÉ HARÍA BUENA A LA RESPUESTA EN ESTE CASO:\n{criterio}\n\n"
        f"SEÑAL EMITIDA POR EL SISTEMA:\n{senal}\n\n"
        f"Puntaje (1-5):"
    )


class Judge:
    """Juez LLM. Carga el modelo una vez y puntúa señales de forma determinista."""

    def __init__(
        self,
        rubrica_path: Path | None = None,
        modelo: str = MODELO_JUEZ,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        here = Path(__file__).parent
        self.rubrica_path = rubrica_path or (here / "judge_rubric.yaml")
        self.rubrica = yaml.safe_load(self.rubrica_path.read_text(encoding="utf-8"))
        self.version_rubrica = self.rubrica["version"]
        self.modelo_id = modelo
        self.prompt_sistema = construir_prompt_sistema(self.rubrica)
        self.few_shot = construir_few_shot(self.rubrica)

        # Resolver el device ANTES de elegir el dtype: en CPU hay que forzar
        # float32, porque bfloat16 sin soporte de hardware va un orden de
        # magnitud más lento y aquí solo hacemos forward passes.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32 if self.device == "cpu" else "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        self.model = AutoModelForCausalLM.from_pretrained(modelo, dtype=dtype)
        self.model.eval()
        self.model.to(self.device)

        # Los cinco niveles deben ser un único token cada uno; si no, el truco
        # de los logits no aplica y hay que caer a generar y parsear.
        self.ids_nivel = []
        for n in NIVELES:
            ids = self.tokenizer.encode(str(n), add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(
                    f"El tokenizador de {modelo} parte '{n}' en {len(ids)} tokens; "
                    "la puntuación por logits necesita un token por nivel."
                )
            self.ids_nivel.append(ids[0])

    def score(self, fragmento: str, criterio: str, senal: str) -> ScoreJuez:
        mensajes = [
            {"role": "system", "content": self.prompt_sistema},
            *self.few_shot,
            {"role": "user", "content": construir_prompt_usuario(fragmento, criterio, senal)},
        ]
        texto = self.tokenizer.apply_chat_template(
            mensajes, tokenize=False, add_generation_prompt=True
        )
        entradas = self.tokenizer(texto, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.model(**entradas).logits[0, -1]
        logits_nivel = logits[self.ids_nivel]
        probs = self.torch.softmax(logits_nivel.float(), dim=-1).tolist()
        nivel = NIVELES[int(max(range(len(NIVELES)), key=lambda i: probs[i]))]
        return ScoreJuez(nivel=nivel, probs=dict(zip(NIVELES, probs)))

    # ----------------------------------------------------------------------
    def sanity_check(self) -> dict:
        """¿El juez distingue señales de calidad distinta, y dónde deja de hacerlo?

        Un juez que le pone el mismo número a todo no es evidencia de nada. Esto
        le pasa el MISMO fragmento con cinco señales de calidad conocida y
        comprueba dos cosas distintas:

        - `separa_util_de_inutil`: lo que el juez SÍ hace de forma fiable —
          poner la señal correcta por encima de una que no aporta nada.
        - `detecta_signo_invertido`: su punto ciego medido. Un juez de 1.5B no
          reconoce con fiabilidad que una señal tiene el signo al revés; en
          nuestras pruebas la puntúa 3 con una confianza de ~0.5, es decir,
          dudando. Por eso ese error —el más caro de nuestra rúbrica— NO se
          delega en el juez: la dimensión 3 del harness lo verifica de forma
          determinista.

        El scorecard guarda este resultado, para que nadie lea la nota del juez
        sin saber qué es capaz de ver.
        """
        fragmento = (
            "El Getafe comunica una grave lesión en la rodilla derecha de Uche. "
            "El club confirma que el nigeriano se perderá toda la temporada"
        )
        criterio = (
            "Debe leer el hecho que hay dentro de la declaración: el jugador se pierde toda la "
            "temporada, así que es una baja confirmada de impacto negativo alto para el Getafe."
        )
        casos = {
            "correcta": ("baja_confirmada | impacto: negativo_alto | equipo: Getafe", 5),
            "equipo_incorrecto": (
                "baja_confirmada | impacto: negativo_alto | equipo: Rayo Vallecano",
                4,
            ),
            "categoria_vecina": (
                "sancion_suspension | impacto: negativo_alto | equipo: Getafe",
                3,
            ),
            "sin_senal": ("irrelevante | impacto: neutro | equipo: ninguno", 3),
            "signo_invertido": ("regreso_alta | impacto: positivo_alto | equipo: Getafe", 1),
        }
        obtenidos, esperados, confianzas = {}, {}, {}
        for nombre, (senal, esperado) in casos.items():
            res = self.score(fragmento, criterio, senal)
            obtenidos[nombre] = res.nivel
            esperados[nombre] = esperado
            confianzas[nombre] = round(res.confianza, 3)

        return {
            "niveles_obtenidos": obtenidos,
            "niveles_esperados": esperados,
            "confianza": confianzas,
            "separa_util_de_inutil": obtenidos["correcta"] > obtenidos["sin_senal"],
            "detecta_signo_invertido": obtenidos["signo_invertido"] <= 2,
            "aciertos_exactos": sum(1 for k in casos if obtenidos[k] == esperados[k]),
            "total_casos": len(casos),
        }


def main() -> int:
    """Comprobación rápida del juez, sin correr el harness entero."""
    juez = Judge()
    print(f"Juez: {juez.modelo_id} | rúbrica v{juez.version_rubrica} | device={juez.device}")
    res = juez.sanity_check()
    print()
    print("SANITY CHECK — mismo fragmento, cinco señales de calidad conocida:")
    for caso in res["niveles_obtenidos"]:
        obt, esp, conf = (
            res["niveles_obtenidos"][caso],
            res["niveles_esperados"][caso],
            res["confianza"][caso],
        )
        marca = "OK" if obt == esp else f"esperado {esp}"
        print(f"  {caso:<20} -> {obt}  (conf {conf:.2f})  {marca}")
    print()
    print(f"  aciertos exactos:        {res['aciertos_exactos']}/{res['total_casos']}")
    print(f"  separa útil de inútil:   {'SÍ' if res['separa_util_de_inutil'] else 'NO'}")
    print(
        f"  detecta signo invertido: {'SÍ' if res['detecta_signo_invertido'] else 'NO (punto ciego conocido)'}"
    )
    return 0 if res["separa_util_de_inutil"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
