"""Construye el eval set de dominio de Offside.

Qué es
------
Ejemplos nuestros, etiquetados a mano, que miden lo que le importa a nuestro
usuario. No salen de ningún benchmark público, así que no pueden estar
contaminados: son noticias de agosto de 2026 y las etiquetas las pusimos
nosotros leyendo cada fragmento.

Cada ejemplo lleva los tres campos que pide M2:

    input      lo que ve el sistema (fragmento + fuente + fecha)
    esperado   la respuesta correcta: categoría, impacto y equipo afectado
    criterio   qué haría buena a la respuesta EN ESE CASO concreto

Cómo se eligieron
-----------------
NO son ejemplos al azar del corpus. Se eligieron a mano buscando **casos
difíciles**: donde el baseline se rompe, donde el impacto es matizado, donde
hay vocabulario disparador en un contexto que no aplica, o donde ni siquiera
está claro cuál es la respuesta. Un eval set de casos fáciles no informa nada.

Los marcados `adversarial` son los que el módulo pide explícitamente: casos
ambiguos, con trampa, fuera de dominio, o donde el sistema suele fallar. El
resto son casos normales que sirven de control: un eval set formado solo por
trampas tampoco sirve, porque no detectaría una regresión que rompa lo que hoy
funciona.

Procedencia
-----------
El texto NO se transcribe a mano: se extrae del corpus por `id`, así que es
byte a byte el que recolectamos, y cada ejemplo arrastra su enlace de origen.
La única excepción está marcada con `origen: "redactado_equipo"` y usa
entidades ficticias a propósito (ver OFF-18).

Uso:
    python build_eval_set.py [--corpus ../data/processed/weak_labeled.jsonl]
                             [--out eval_set.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# --------------------------------------------------------------------------
# Ejemplos curados a partir del corpus real, por `id`.
# --------------------------------------------------------------------------
CURADOS: list[dict] = [
    {
        "corpus_id": "024ca02de28ba07e",
        "esperado": {"category": "baja_confirmada", "impact": "negativo_alto", "team": "Getafe"},
        "criterio": (
            "Debe leer el hecho que hay DENTRO de la declaración: el club confirma que el "
            "jugador se pierde toda la temporada, así que es una baja confirmada de impacto "
            "negativo alto para el Getafe. Quedarse en 'esto es una declaración' es insuficiente: "
            "el usuario necesita saber que ese jugador no va a estar."
        ),
        "adversarial": True,
        "tipo_dificultad": "trampa: declaración que contiene un hecho",
        "por_que_dificil": (
            "Las comillas del titular disparan 'declaracion_contexto' y el léxico se detiene ahí, "
            "antes de leer 'se perderá toda la temporada'. Confunde el continente con el contenido."
        ),
    },
    {
        "corpus_id": "f538603932f7ce91",
        "esperado": {
            "category": "baja_confirmada",
            "impact": "negativo_alto",
            "team": "Real Oviedo",
        },
        "criterio": (
            "Una baja de dos a tres meses es la señal más accionable que existe en nuestro "
            "dominio. La respuesta debe marcarla como baja confirmada de impacto negativo alto "
            "para el Oviedo. Devolver 'irrelevante' aquí es el peor error posible del sistema."
        ),
        "adversarial": False,
        "tipo_dificultad": "falso negativo total del baseline",
        "por_que_dificil": (
            "'Baja sensible' solo está en el ajuste de sentimiento del léxico, no como patrón de "
            "categoría, así que la noticia más importante del corpus le resulta invisible."
        ),
    },
    {
        "corpus_id": "1c4483eeeea95edd",
        "esperado": {"category": "irrelevante", "impact": "neutro", "team": "Valencia"},
        "criterio": (
            "Hay una expulsión, pero en el Trofeo Naranja: un amistoso de pretemporada. Una roja "
            "en amistoso no arrastra sanción liguera, así que no cambia la disponibilidad para "
            "ningún partido apostable. La respuesta buena reconoce el contexto y NO emite señal."
        ),
        "adversarial": True,
        "tipo_dificultad": "trampa: vocabulario disparador fuera de contexto",
        "por_que_dificil": (
            "El léxico ve 'expulsado' y dispara sancion_suspension con impacto negativo alto. "
            "Para nuestro usuario esa señal es ruido puro. Forma par mínimo con OFF-11, que es "
            "casi el mismo texto pero en partido de liga y sí cuenta."
        ),
    },
    {
        "corpus_id": "c7e96a97e238a5e9",
        "esperado": {"category": "regreso_alta", "impact": "positivo_alto", "team": "Chelsea"},
        "criterio": (
            "La noticia es que el jugador VUELVE tras 615 días; la suspensión está en pasado y es "
            "el contexto, no el hecho. La respuesta debe tener signo POSITIVO. Devolver un impacto "
            "negativo aquí no es un error de matiz: empuja al usuario en la dirección contraria."
        ),
        "adversarial": True,
        "tipo_dificultad": "trampa: tiempo verbal, signo invertido",
        "por_que_dificil": (
            "El léxico ve 'suspendido' y devuelve sancion_suspension / negativo_alto, que es "
            "exactamente el signo opuesto a la respuesta correcta."
        ),
    },
    {
        "corpus_id": "f31bd83c4ff5d3f4",
        "esperado": {"category": "regreso_alta", "impact": "positivo_bajo", "team": "desconocido"},
        "criterio": (
            "Es un regreso, así que el signo es positivo. Pero la intensidad debe ser BAJA: el "
            "propio texto dice que tiene una contrarreloj para recuperar la forma. Reaparecer no "
            "es estar disponible, y prometer más de lo que hay perjudica al usuario."
        ),
        "adversarial": False,
        "tipo_dificultad": "regreso sin vocabulario de regreso",
        "por_que_dificil": (
            "'Reaparece once meses después' no coincide con ninguna de las fórmulas que el léxico "
            "busca, así que cae en irrelevante."
        ),
    },
    {
        "corpus_id": "200eadd0cede0fdb",
        "esperado": {"category": "regreso_alta", "impact": "positivo_bajo", "team": "FC Barcelona"},
        "criterio": (
            "La categoría es un regreso, pero adelantar la vuelta a los entrenamientos NO es "
            "estar disponible para jugar. La respuesta buena acierta la categoría y mantiene la "
            "intensidad baja."
        ),
        "adversarial": False,
        "tipo_dificultad": "impacto sobrestimado",
        "por_que_dificil": (
            "El léxico acierta la categoría pero le asigna positivo_alto por el valor por defecto "
            "de la clase. Es el caso que mide el matiz de intensidad."
        ),
    },
    {
        "corpus_id": "07ef5a01c92d183f",
        "esperado": {
            "category": "baja_confirmada",
            "impact": "negativo_alto",
            "team": "Las Palmas",
        },
        "criterio": (
            "El resumen pega dos noticias: un fichaje (irrelevante) y un portero operado (baja "
            "confirmada). La respuesta buena se queda con la SEGUNDA, que es la que afecta a un "
            "partido, y no con la primera que aparece."
        ),
        "adversarial": True,
        "tipo_dificultad": "trampa: fragmento compuesto",
        "por_que_dificil": (
            "El léxico acierta hoy solo porque le añadimos el patrón 'pasó por el quirófano' tras "
            "encontrar este error a mano. Sirve como test de regresión de ese parche."
        ),
    },
    {
        "corpus_id": "118d03958a0c1f46",
        "esperado": {"category": "irrelevante", "impact": "neutro", "team": "ninguno"},
        "criterio": (
            "Está cargado de vocabulario de sanción, pero es un cambio de normativa general de la "
            "UEFA: no afecta a ningún equipo concreto en ningún partido concreto. Sin equipo "
            "afectado no hay señal que darle al usuario."
        ),
        "adversarial": True,
        "tipo_dificultad": "trampa: distractor normativo",
        "por_que_dificil": "Amonestaciones y ciclos de amarillas sin que haya un partido afectado.",
    },
    {
        "corpus_id": "d6bd98772e5975c9",
        "esperado": {
            "category": "declaracion_contexto",
            "impact": "neutro",
            "team": "Sporting de Gijon",
        },
        "criterio": (
            "Aparece la palabra 'dudas', pero significa dudas DEFENSIVAS: un juicio táctico del "
            "entrenador, no una molestia de un jugador. La respuesta buena no la confunde con "
            "duda_fisica."
        ),
        "adversarial": True,
        "tipo_dificultad": "trampa: homonimia de 'dudas'",
        "por_que_dificil": (
            "El léxico acierta por el orden en que evalúa sus reglas, no porque entienda la "
            "diferencia. Un modelo debería acertarla por significado."
        ),
    },
    {
        "corpus_id": "50271335c0a843ea",
        "esperado": {
            "category": "declaracion_contexto",
            "impact": "neutro",
            "team": "Bayern Munich",
        },
        "criterio": (
            "'Insinúa' y 'podría' son lenguaje de rumor, pero la fuente es el propio jugador "
            "hablando de sí mismo: es una declaración. Y es sobre su retirada a fin de temporada, "
            "así que no afecta a ningún partido próximo y el impacto es neutro."
        ),
        "adversarial": True,
        "tipo_dificultad": "ambiguo: frontera declaración / rumor",
        "por_que_dificil": "El vocabulario apunta a rumor; la fuente y el horizonte temporal, no.",
    },
    # ---------------- Recolección en temporada (agosto 2026) ----------------
    {
        "corpus_id": "ce8358e668ef1260",
        "esperado": {
            "category": "sancion_suspension",
            "impact": "negativo_alto",
            "team": "Celta de Vigo",
        },
        "criterio": (
            "Aquí la roja SÍ cuenta: es un partido de liga, así que arrastra sanción para la "
            "jornada siguiente y el Celta pierde a ese jugador. La respuesta buena distingue este "
            "caso de OFF-03, donde el texto es casi idéntico pero la roja fue en un amistoso."
        ),
        "adversarial": True,
        "tipo_dificultad": "trampa: par mínimo con OFF-03",
        "por_que_dificil": (
            "OFF-03 y OFF-11 comparten casi todo el vocabulario (roja directa, expulsión) y tienen "
            "respuestas opuestas. Lo único que las separa es el contexto de competición, que no "
            "está dicho de forma explícita. El léxico se equivoca en las dos, cada una al revés."
        ),
    },
    {
        "corpus_id": "cfa48ba64563b824",
        "esperado": {"category": "baja_confirmada", "impact": "negativo_alto", "team": "Girona"},
        "criterio": (
            "Tres meses de baja es una señal de alto impacto para el Girona, y hay que emitirla "
            "aunque el resto del fragmento hable de una derrota. La respuesta buena separa el "
            "hecho accionable del relleno del resumen."
        ),
        "adversarial": False,
        "tipo_dificultad": "baja real en temporada que el baseline pierde",
        "por_que_dificil": (
            "'Estará de baja alrededor de tres meses' no coincide con ningún patrón del léxico, "
            "que devuelve irrelevante."
        ),
    },
    {
        "corpus_id": "92265aac586c7602",
        "esperado": {"category": "cambio_tactico", "impact": "neutro", "team": "FC Barcelona"},
        "criterio": (
            "Es información de alineación confirmada antes del partido: le dice al usuario quién "
            "juega. La categoría es cambio táctico y el impacto neutro, porque saber el once no "
            "favorece por sí mismo a ninguno de los dos equipos."
        ),
        "adversarial": False,
        "tipo_dificultad": "primer cambio_tactico real del corpus",
        "por_que_dificil": (
            "Toda la clase cambio_tactico estaba vacía en el corte de pretemporada. El léxico "
            "sigue devolviendo irrelevante porque el patrón que tiene es 'alineación probable', "
            "no 'alineaciones del partido'."
        ),
    },
    {
        "corpus_id": "4d8a6afbcdf30f52",
        "esperado": {"category": "duda_fisica", "impact": "negativo_bajo", "team": "Almeria"},
        "criterio": (
            "El jugador no viajó por molestias, así que es una duda física de impacto negativo "
            "bajo. Lo delicado es que el propio texto sugiere que la molestia alimenta rumores de "
            "salida: la respuesta buena reporta lo verificable (no viajó, molestias) sin "
            "convertirlo en una baja confirmada ni en un rumor de traspaso."
        ),
        "adversarial": True,
        "tipo_dificultad": "ambiguo: molestia real o pretexto",
        "por_que_dificil": (
            "El mismo fragmento admite dos lecturas —lesión leve o maniobra de mercado— y la "
            "diferencia importa para el usuario. Es el caso donde más fácil resulta sobre-afirmar."
        ),
    },
    {
        "corpus_id": "26dacbe218db7430",
        "esperado": {
            "category": "sancion_suspension",
            "impact": "negativo_alto",
            "team": "Millonarios",
        },
        "criterio": (
            "Roja directa en partido de Liga BetPlay: sanción para la fecha siguiente e impacto "
            "negativo alto para Millonarios. La respuesta debe ser la misma que daría para un caso "
            "equivalente en LaLiga; el sistema no puede rendir peor por estar leyendo prensa "
            "colombiana."
        ),
        "adversarial": True,
        "tipo_dificultad": "variante lingüística latinoamericana",
        "por_que_dificil": (
            "Todo el entrenamiento es prensa española. Este ejemplo existe para medir el sesgo de "
            "variante que documentamos en M1: léxico distinto ('se salió de casillas', 'fecha 7' "
            "en vez de 'jornada 7') para el mismo hecho."
        ),
    },
    {
        "corpus_id": "68d5d6c2119f26eb",
        "esperado": {"category": "irrelevante", "impact": "neutro", "team": "ninguno"},
        "criterio": (
            "Es tenis, no fútbol. La respuesta buena reconoce que está fuera del dominio y no "
            "emite ninguna señal. Inventar una categoría de fútbol aquí sería alucinar."
        ),
        "adversarial": True,
        "tipo_dificultad": "fuera de dominio",
        "por_que_dificil": (
            "Llega por el mismo feed de deportes que las noticias de fútbol, tiene nombres "
            "propios, competición y calendario. Todo se parece a nuestro dominio menos lo esencial."
        ),
    },
    {
        "corpus_id": "3867e6615f8b81e0",
        "esperado": {
            "category": "sancion_suspension",
            "impact": "negativo_bajo",
            "team": "Internacional de Bogota",
        },
        "criterio": (
            "Hay una sanción real y larga, así que la categoría es sanción. Pero el texto dice "
            "'integrante', no jugador titular: sin saber si afecta al once, el impacto debe ser "
            "BAJO. La respuesta buena no infla el impacto por lo llamativo de 'seis fechas'."
        ),
        "adversarial": True,
        "tipo_dificultad": "ambiguo: ¿a quién sanciona?",
        "por_que_dificil": (
            "La sanción es indiscutible pero el sujeto es vago. Es el caso que separa 'detectar el "
            "hecho' de 'estimar su impacto', que en nuestro dominio son dos preguntas distintas."
        ),
    },
    # ------------------------------ Controles ------------------------------
    # Casos FÁCILES a propósito, donde el baseline acierta y el texto no tiene
    # trampa. Sin ellos el eval set sería solo dificultad, y entonces no se
    # podría distinguir "el sistema es malo" de "los casos son imposibles":
    # estos tres dan el punto de referencia. También son la red de seguridad
    # ante regresiones — si un cambio futuro rompe ESTOS, algo va muy mal.
    {
        "corpus_id": "7c44fd3a047d7e56",
        "esperado": {"category": "baja_confirmada", "impact": "negativo_alto", "team": "Burgos"},
        "criterio": (
            "Un jugador operado de una rotura de menisco es una baja confirmada de impacto "
            "negativo alto para el Burgos. El texto lo dice sin ambigüedad: es el caso fácil."
        ),
        "adversarial": False,
        "tipo_dificultad": "control: baja explícita",
        "por_que_dificil": "No lo es. Es el control de que el sistema detecta lo evidente.",
    },
    {
        "corpus_id": "0e611bcca64b2549",
        "esperado": {"category": "irrelevante", "impact": "neutro", "team": "ninguno"},
        "criterio": (
            "Es un fichaje. No cambia la disponibilidad de nadie para un partido próximo, así que "
            "no hay señal que darle al usuario: la respuesta correcta es no emitir nada."
        ),
        "adversarial": False,
        "tipo_dificultad": "control: irrelevante explícito",
        "por_que_dificil": "No lo es. Es el control de que el sistema NO inventa señales.",
    },
    {
        "corpus_id": "9b1325849c3a3d1e",
        "esperado": {
            "category": "declaracion_contexto",
            "impact": "neutro",
            "team": "Manchester City",
        },
        "criterio": (
            "Es una declaración de un entrenador sin ningún hecho accionable dentro: no anuncia "
            "bajas, sanciones ni regresos. Categoría declaración e impacto neutro."
        ),
        "adversarial": False,
        "tipo_dificultad": "control: declaración sin hecho",
        "por_que_dificil": (
            "No lo es. Es el contraste de OFF-01, que también es una declaración pero SÍ contiene "
            "un hecho dentro."
        ),
    },
]

# --------------------------------------------------------------------------
# Ejemplo redactado por el equipo.
#
# El corpus no contiene ni un solo rumor no confirmado sobre DISPONIBILIDAD:
# los únicos rumores que llegan por RSS son de fichajes. Es una limitación real
# del formato (titular + entradilla), no un descuido — ese tipo de rumor vive
# en el cuerpo del artículo o en redes, que deliberadamente no scrapeamos.
#
# Sin ningún ejemplo de esta clase, el criterio "no presentar un rumor como un
# hecho" no se puede medir, y es uno de los que más pesa en nuestra rúbrica.
# Redactamos uno con entidades FICTICIAS a propósito: nunca atribuimos una
# lesión o sanción a una persona real que no la tuvo.
# --------------------------------------------------------------------------
REDACTADOS: list[dict] = [
    {
        "input": {
            "text": (
                "Según fuentes cercanas al vestuario, el capitán del Deportivo Sauce podría "
                "perderse el derbi del domingo por unas molestias que el club no ha confirmado. "
                "El cuerpo técnico no se ha pronunciado."
            ),
            "source": "redactado_equipo",
            "published_at": None,
        },
        "esperado": {
            "category": "rumor_no_confirmado",
            "impact": "negativo_bajo",
            "team": "Deportivo Sauce",
        },
        "criterio": (
            "Todo el fragmento está en condicional y sin fuente oficial ('según fuentes', "
            "'podría', 'el club no ha confirmado'). La respuesta buena lo marca como rumor y NO "
            "como baja confirmada: presentar un rumor como hecho propaga información falsa sobre "
            "una persona identificable, que es el error que nuestra rúbrica castiga más fuerte "
            "después de invertir el signo."
        ),
        "adversarial": True,
        "tipo_dificultad": "ambiguo: rumor con forma de baja",
        "por_que_dificil": (
            "Comparte casi todo el vocabulario con una baja confirmada ('se pierde el partido', "
            "'molestias') y solo lo separan los marcadores de incertidumbre."
        ),
        "procedencia": {
            "corpus_id": None,
            "link": None,
            "origen": "redactado_equipo",
            "nota": (
                "Entidades ficticias a propósito. El corpus real no contiene rumores de "
                "disponibilidad: los rumores que llegan por RSS son todos de fichajes."
            ),
        },
    },
]


def main() -> int:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=here / ".." / "data" / "processed" / "weak_labeled.jsonl"
    )
    parser.add_argument("--out", type=Path, default=here / "eval_set.json")
    args = parser.parse_args()

    corpus = {}
    with args.corpus.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                corpus[rec["id"]] = rec

    faltan = [c["corpus_id"] for c in CURADOS if c["corpus_id"] not in corpus]
    if faltan:
        raise SystemExit(f"Estos ids curados no están en el corpus: {faltan}")

    registros = []
    for i, cur in enumerate(CURADOS, start=1):
        src = corpus[cur["corpus_id"]]
        registros.append(
            {
                "eval_id": f"OFF-{i:02d}",
                "input": {
                    "text": src["text"],
                    "source": src["source"],
                    "published_at": src.get("published_at"),
                },
                "esperado": cur["esperado"],
                "criterio": cur["criterio"],
                "adversarial": cur["adversarial"],
                "tipo_dificultad": cur["tipo_dificultad"],
                "por_que_dificil": cur["por_que_dificil"],
                "procedencia": {
                    "corpus_id": cur["corpus_id"],
                    "link": src.get("link"),
                    "origen": "rss_real",
                },
            }
        )
    # Los redactados se numeran DESPUÉS de los curados. Tenerlo fijo en el
    # diccionario provocó un eval_id duplicado al añadir ejemplos nuevos, y el
    # duplicado no rompía nada de forma visible: simplemente hacía que dos
    # ejemplos se pisaran en cualquier análisis agrupado por eval_id.
    for j, red in enumerate(REDACTADOS, start=len(registros) + 1):
        registros.append({"eval_id": f"OFF-{j:02d}", **red})

    ids = [r["eval_id"] for r in registros]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"eval_id duplicado: {sorted({i for i in ids if ids.count(i) > 1})}")

    args.out.write_text(
        json.dumps(registros, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    n_adv = sum(1 for r in registros if r["adversarial"])
    cats = sorted({r["esperado"]["category"] for r in registros})
    print(f"{len(registros)} ejemplos escritos en {args.out}")
    print(
        f"  adversariales / borde: {n_adv} ({100 * n_adv / len(registros):.0f}%) — el mínimo es 20%"
    )
    print(f"  categorías ejercitadas: {len(cats)}/8 -> {', '.join(cats)}")
    print(f"  redactados por el equipo: {len(REDACTADOS)} (el resto es RSS real)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
