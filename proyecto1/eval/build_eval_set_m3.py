"""Construye el eval set del Módulo 3: preguntas de un apostador sobre el corpus.

    python build_eval_set_m3.py   ->   eval_set_m3.json

Por qué un eval set nuevo
-------------------------
El de M2 (`eval_set.json`) mide clasificación de fragmentos: el input ES la
noticia. Un RAG no tiene nada que recuperar ahí. El enunciado de M3 lo prevé:
si la tarea de M1 era de clasificación, se escriben preguntas nuevas sobre el
corpus. Eso hacemos, pero **sin cambiar la vara**: cada pregunta tiene la misma
respuesta estructurada que el harness de M2 ya sabe medir —categoría del hecho,
impacto y equipo— más la respuesta de referencia que necesita RAGAS.

    input     {"text": la pregunta}
    esperado  {"category", "impact", "team"}   <- lo que mide harness.py, intacto
    criterio  en prosa, para el juez de la dimensión 2 (igual que en M2)
    referencia + referencia_afirmaciones       <- lo que necesita RAGAS
    docs_relevantes                            <- ids del corpus: diagnóstico de recuperación
    en_corpus = False                          <- la respuesta esperada es la válvula

La válvula en el lenguaje del harness
-------------------------------------
Cuando la respuesta no está en el corpus, la señal correcta es "no hay señal":
`irrelevante | neutro | ninguno`. En la rúbrica del juez eso es el nivel 3 ("no
engaña"), y en la dimensión 3 una señal inventada de alto impacto tumba el
ejemplo. O sea: el harness de M2 ya castigaba inventar; no hubo que tocarlo.

Fecha de la consulta
--------------------
Todas las preguntas se hacen el 2026-10-14, el día del corte del corpus. El
sistema recibe esa fecha fija (`FECHA_CONSULTA` en rag/config.py), no la del
reloj, para que la tabla sea reproducible otro día.

Los ids de `docs_relevantes` se validan contra data/rag/corpus.jsonl al
construir: si alguien regenera el corpus y un documento desaparece, esto falla
en vez de producir un eval set que mide contra la nada.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import CATEGORIES  # noqa: E402

IMPACTOS = {"negativo_alto", "negativo_bajo", "neutro", "positivo_bajo", "positivo_alto"}
VALVULA = "No encuentro esa información en las noticias recolectadas."
SIN_SENAL = {"category": "irrelevante", "impact": "neutro", "team": "ninguno"}

CASOS = [
    # ------------------------------------------------------------------ sanciones
    {
        "eval_id": "M3-01",
        "pregunta": "¿Vinicius puede jugar el Clásico?",
        "esperado": {
            "category": "sancion_suspension",
            "impact": "negativo_alto",
            "team": "Real Madrid",
        },
        "criterio": (
            "La respuesta correcta es una sanción: Vinicius se pierde el Clásico porque le "
            "impusieron cuatro partidos de suspensión por su roja ante el Villarreal. Impacto "
            "negativo alto para el Real Madrid. Decir que juega, o que vuelve al once, invierte "
            "el signo."
        ),
        "referencia": (
            "No. Vinicius se pierde el Clásico: el Comité de Disciplina le impuso cuatro "
            "partidos de sanción por su roja ante el Villarreal."
        ),
        "referencia_afirmaciones": [
            "Vinicius no puede jugar el Clásico.",
            "Vinicius fue sancionado con cuatro partidos.",
            "La sanción es por la tarjeta roja ante el Villarreal.",
        ],
        "docs_relevantes": ["cba83938135d711f", "121aa9cb87ef0452", "a223284dd29ea88b"],
        "adversarial": True,
        "tipo_dificultad": "conflicto temporal",
        "por_que_dificil": (
            "Un día antes Marca publicó 'Vinicius vuelve al once de Mourinho' (para la Champions). "
            "Hay cuatro fragmentos sobre Vinicius compitiendo por el top-k."
        ),
    },
    {
        "eval_id": "M3-02",
        "pregunta": "¿Juega Vinicius contra la Roma en la Champions?",
        "esperado": {"category": "cambio_tactico", "impact": "neutro", "team": "Real Madrid"},
        "criterio": (
            "Par mínimo de M3-01. La sanción de cuatro partidos es de Liga; para la Champions "
            "ante la Roma, Marca da a Vinicius en el posible once. Es una noticia de alineación "
            "(cambio táctico) de impacto neutro. Responder que está sancionado para ese partido "
            "es una señal falsa de alto impacto."
        ),
        "referencia": (
            "Sí, previsiblemente: en el posible once del Roma-Real Madrid de Champions, "
            "Vinicius vuelve al once de Mourinho. Su sanción de cuatro partidos es de Liga."
        ),
        "referencia_afirmaciones": [
            "Vinicius aparece en el posible once del Real Madrid ante la Roma.",
            "El partido contra la Roma es de Champions League.",
            "La sanción de Vinicius es de cuatro partidos de Liga.",
        ],
        "docs_relevantes": ["6c0d96d821b0a34d", "121aa9cb87ef0452"],
        "adversarial": True,
        "tipo_dificultad": "par mínimo: misma persona, otra competición",
        "por_que_dificil": (
            "Los fragmentos de la sanción son más numerosos y más recientes; arrastran a decir "
            "que no juega. Hay que leer que la sanción es 'de Liga'."
        ),
    },
    {
        "eval_id": "M3-03",
        "pregunta": "¿Isaac Romero fue expulsado? ¿Con qué equipo juega?",
        "esperado": {
            "category": "sancion_suspension",
            "impact": "negativo_alto",
            "team": "Sevilla",
        },
        "criterio": (
            "Isaac Romero, delantero del Sevilla, fue expulsado por doble amarilla en un partido "
            "de Liga, lo que arrastra suspensión: sanción, impacto negativo alto para el Sevilla."
        ),
        "referencia": (
            "Sí. Isaac Romero, delantero del Sevilla, marcó el 2-3 y terminó expulsado tras ver "
            "dos amarillas en cuatro minutos."
        ),
        "referencia_afirmaciones": [
            "Isaac Romero fue expulsado.",
            "Isaac Romero juega en el Sevilla.",
            "Vio dos amarillas en cuatro minutos.",
        ],
        "docs_relevantes": ["f91e5278a392f72c", "ac139f7bbda09ef5"],
        "adversarial": False,
        "tipo_dificultad": "normal",
        "por_que_dificil": "Control: el nombre aparece literal en dos fragmentos.",
    },
    {
        "eval_id": "M3-04",
        "pregunta": "¿A quién le falta el Alavés en la banda derecha?",
        "esperado": {
            "category": "sancion_suspension",
            "impact": "negativo_alto",
            "team": "Alaves",
        },
        "criterio": (
            "Al Alavés le falta Ángel Pérez, indiscutible en la banda derecha, porque está "
            "sancionado. Sanción, impacto negativo alto para el Alavés."
        ),
        "referencia": (
            "A Ángel Pérez, que está sancionado; era indiscutible en la banda derecha y Quique "
            "debe buscarle recambio."
        ),
        "referencia_afirmaciones": [
            "Ángel Pérez está sancionado.",
            "Ángel Pérez juega en la banda derecha.",
            "El técnico debe buscarle un recambio.",
        ],
        "docs_relevantes": ["ed98fa9ca790f838"],
        "adversarial": True,
        "tipo_dificultad": "el club solo está en la URL",
        "por_que_dificil": (
            "El fragmento nunca dice 'Alavés' (solo 'Quique' y 'banda derecha'). El club está "
            "en la sección del enlace."
        ),
    },
    {
        "eval_id": "M3-05",
        "pregunta": "¿Quién fue expulsado en el Tottenham contra el Manchester United?",
        "esperado": {
            "category": "sancion_suspension",
            "impact": "negativo_alto",
            "team": "Tottenham",
        },
        "criterio": (
            "Bentancur vio la roja y el Tottenham acabó con diez: una expulsión en liga arrastra "
            "sanción. Impacto negativo alto para el Tottenham."
        ),
        "referencia": (
            "Rodrigo Bentancur: el Tottenham acabó con diez por su roja, en el partido que ganó "
            "al Manchester United con un gol de Solanke en el 92'."
        ),
        "referencia_afirmaciones": [
            "Bentancur fue expulsado con tarjeta roja.",
            "El Tottenham acabó con diez jugadores.",
        ],
        "docs_relevantes": ["400bd72e4fb13f07"],
        "adversarial": False,
        "tipo_dificultad": "normal",
        "por_que_dificil": "El dato está en la segunda frase de una crónica del United.",
    },
    # ------------------------------------------------------------------ bajas
    {
        "eval_id": "M3-06",
        "pregunta": "¿Cuenta el Sevilla con Guridi para jugar en el Bernabéu?",
        "esperado": {
            "category": "baja_confirmada",
            "impact": "negativo_alto",
            "team": "Sevilla",
        },
        "criterio": (
            "No: el Sevilla prepara la visita al Bernabéu sin Guridi, que tiene una lesión en el "
            "aductor. Baja confirmada, impacto negativo alto para el Sevilla."
        ),
        "referencia": (
            "No. El Sevilla prepara la visita al Bernabéu sin Guridi, que sufre una lesión en el "
            "aductor largo del muslo izquierdo."
        ),
        "referencia_afirmaciones": [
            "Guridi no estará en la visita del Sevilla al Bernabéu.",
            "Guridi sufre una lesión en el aductor largo del muslo izquierdo.",
        ],
        "docs_relevantes": ["3b31d5310b9bd6e4"],
        "adversarial": False,
        "tipo_dificultad": "normal",
        "por_que_dificil": "Control.",
    },
    {
        "eval_id": "M3-07",
        "pregunta": "¿Nico Williams estará con el Athletic en Mestalla?",
        "esperado": {
            "category": "baja_confirmada",
            "impact": "negativo_alto",
            "team": "Athletic Club",
        },
        "criterio": (
            "No: Nico seguirá de baja por lesión para el partido en Mestalla. Baja confirmada, "
            "impacto negativo alto para el Athletic."
        ),
        "referencia": "No. Según Marca, Nico seguirá de baja por lesión en el partido de Mestalla.",
        "referencia_afirmaciones": [
            "Nico no jugará en Mestalla.",
            "Nico está de baja por lesión.",
        ],
        "docs_relevantes": ["9c670dc48de7ddb2", "dd5fc7368339ee00"],
        "adversarial": True,
        "tipo_dificultad": "nombre parcial",
        "por_que_dificil": (
            "El corpus dice solo 'Nico'; el apellido de la pregunta no aparece en ningún fragmento."
        ),
    },
    {
        "eval_id": "M3-08",
        "pregunta": "¿Juega Marco Asensio con el Fenerbahçe contra el Aston Villa?",
        "esperado": {
            "category": "baja_confirmada",
            "impact": "negativo_alto",
            "team": "Fenerbahce",
        },
        "criterio": (
            "No: el Fenerbahçe informó de la baja de última hora de Asensio para el partido de "
            "Champions. Baja confirmada, impacto negativo alto para el Fenerbahçe."
        ),
        "referencia": (
            "No. Marco Asensio es baja de última hora para el partido de Champions ante el Aston "
            "Villa; lo informó el Fenerbahçe."
        ),
        "referencia_afirmaciones": [
            "Marco Asensio no jugará contra el Aston Villa.",
            "Es una baja de última hora.",
            "El partido es de Champions League.",
        ],
        "docs_relevantes": ["b5d8a4b3d13f63e4"],
        "adversarial": False,
        "tipo_dificultad": "normal",
        "por_que_dificil": "Control.",
    },
    {
        "eval_id": "M3-09",
        "pregunta": "¿Volverá Uche a jugar con el Getafe esta temporada?",
        "esperado": {
            "category": "baja_confirmada",
            "impact": "negativo_alto",
            "team": "Getafe",
        },
        "criterio": (
            "No: el Getafe confirmó una grave lesión de rodilla y Uche se pierde toda la "
            "temporada. Baja confirmada, impacto negativo alto. No hace falta calcular fechas."
        ),
        "referencia": (
            "No. El Getafe comunicó una grave lesión en la rodilla derecha de Uche, que se "
            "perderá toda la temporada."
        ),
        "referencia_afirmaciones": [
            "Uche sufre una grave lesión en la rodilla derecha.",
            "Uche se perderá toda la temporada.",
        ],
        "docs_relevantes": ["024ca02de28ba07e"],
        "adversarial": False,
        "tipo_dificultad": "normal; tentación de usar la herramienta sin necesidad",
        "por_que_dificil": "'Esta temporada' suena a pregunta de fechas, pero no hay nada que calcular.",
    },
    {
        "eval_id": "M3-10",
        "pregunta": "¿Qué problema tiene el Córdoba en el extremo izquierdo?",
        "esperado": {
            "category": "baja_confirmada",
            "impact": "negativo_alto",
            "team": "Cordoba",
        },
        "criterio": (
            "Kevin Medina, su extremo izquierdo, está lesionado para varias semanas. Baja "
            "confirmada, impacto negativo alto para el Córdoba."
        ),
        "referencia": (
            "Kevin Medina, su extremo izquierdo, está lesionado para varias semanas; es la última "
            "de una serie de lesiones en ese puesto."
        ),
        "referencia_afirmaciones": [
            "Kevin Medina está lesionado.",
            "La lesión es para varias semanas.",
            "Kevin Medina juega de extremo izquierdo en el Córdoba.",
        ],
        "docs_relevantes": ["3a1a705dc867b9e6"],
        "adversarial": False,
        "tipo_dificultad": "pregunta sin nombre propio",
        "por_que_dificil": "La pregunta no nombra al jugador: solo el puesto y el club.",
    },
    {
        "eval_id": "M3-11",
        "pregunta": "¿Quién será el portero de la Roma contra el Real Madrid?",
        "esperado": {"category": "baja_confirmada", "impact": "negativo_alto", "team": "Roma"},
        "criterio": (
            "El portero titular de la Roma, Svilar, está lesionado; juega Gollini. La señal es la "
            "baja de Svilar: baja confirmada, impacto negativo alto para la Roma."
        ),
        "referencia": "Pierluigi Gollini, por la lesión de Svilar.",
        "referencia_afirmaciones": [
            "Svilar está lesionado.",
            "Gollini será el portero de la Roma ante el Real Madrid.",
        ],
        "docs_relevantes": ["12c66095ff585dc6"],
        "adversarial": True,
        "tipo_dificultad": "titular metafórico",
        "por_que_dificil": (
            "La noticia se titula 'El rapero con guantes que quiere silenciar al Madrid': ni "
            "'portero' ni 'Roma' aparecen como tales en el titular."
        ),
    },
    {
        "eval_id": "M3-12",
        "pregunta": "¿Qué jugadores lesionados tiene el Celta?",
        "esperado": {"category": "baja_confirmada", "impact": "negativo_alto", "team": "Celta"},
        "criterio": (
            "Pablo Durán y Borja Iglesias están lesionados y fuera de la convocatoria. Baja "
            "confirmada, impacto negativo alto para el Celta."
        ),
        "referencia": (
            "Pablo Durán y Borja Iglesias, que se quedan fuera de la convocatoria del Celta por "
            "lesión."
        ),
        "referencia_afirmaciones": [
            "Pablo Durán está lesionado.",
            "Borja Iglesias está lesionado.",
            "Ninguno de los dos está en la convocatoria del Celta.",
        ],
        "docs_relevantes": ["3f2325dbfa69c70d"],
        "adversarial": False,
        "tipo_dificultad": "normal",
        "por_que_dificil": "Control.",
    },
    {
        "eval_id": "M3-13",
        "pregunta": "¿Qué le pasó a Ana María Guzmán con la Selección Colombia?",
        "esperado": {
            "category": "baja_confirmada",
            "impact": "negativo_alto",
            "team": "Seleccion Colombia",
        },
        "criterio": (
            "Sufrió una grave lesión en el amistoso contra Venezuela y será operada: baja "
            "confirmada, impacto negativo alto para la Selección Colombia."
        ),
        "referencia": (
            "Sufrió una grave lesión a los pocos segundos del amistoso contra Venezuela y será "
            "operada."
        ),
        "referencia_afirmaciones": [
            "Ana María Guzmán sufrió una grave lesión.",
            "La lesión fue en el amistoso contra Venezuela.",
            "Será operada.",
        ],
        "docs_relevantes": ["f325c610bd68f03a"],
        "adversarial": False,
        "tipo_dificultad": "variante latam (El Tiempo)",
        "por_que_dificil": "Fuente colombiana, minoritaria en el corpus (36 de 748).",
    },
    # ------------------------------------------------------------------ dudas / regresos
    {
        "eval_id": "M3-14",
        "pregunta": "¿Qué le pasa a Laporte en el Athletic?",
        "esperado": {
            "category": "duda_fisica",
            "impact": "negativo_bajo",
            "team": "Athletic Club",
        },
        "criterio": (
            "Laporte no viajó por una indisposición y sigue sin entrenar con el grupo: es una "
            "duda física, impacto negativo bajo. Inflarlo a baja confirmada de alto impacto es "
            "una señal falsa."
        ),
        "referencia": (
            "Laporte no viajó a Vallecas por una indisposición y sigue sin entrenar con el grupo."
        ),
        "referencia_afirmaciones": [
            "Laporte no viajó a Vallecas por una indisposición.",
            "Laporte sigue sin entrenar con el grupo.",
        ],
        "docs_relevantes": ["dd5fc7368339ee00", "9c670dc48de7ddb2"],
        "adversarial": False,
        "tipo_dificultad": "intensidad",
        "por_que_dificil": "Comparte fragmento con la baja de Nico: fácil contagiarse del 'de baja'.",
    },
    {
        "eval_id": "M3-15",
        "pregunta": "¿Es grave la lesión de Beitia?",
        "esperado": {
            "category": "duda_fisica",
            "impact": "negativo_bajo",
            "team": "Real Sociedad",
        },
        "criterio": (
            "No es grave: un esguince de tobillo; se temió algo peor y no lo fue. Duda física, "
            "impacto negativo bajo. Responder 'lesión grave' es una señal falsa de alto impacto."
        ),
        "referencia": (
            "No. Beitia sufre un esguince de tobillo izquierdo; se temió una lesión grave, pero "
            "no lo ha sido."
        ),
        "referencia_afirmaciones": [
            "Beitia sufre un esguince de tobillo izquierdo.",
            "La lesión no es grave.",
        ],
        "docs_relevantes": ["8bb40e4d298e1983"],
        "adversarial": True,
        "tipo_dificultad": "negación",
        "por_que_dificil": "El texto contiene 'lesión grave' negado: 'se temió... pero no ha sido así'.",
    },
    {
        "eval_id": "M3-16",
        "pregunta": "¿Ya está recuperado Sadiq en el Valencia?",
        "esperado": {"category": "duda_fisica", "impact": "negativo_bajo", "team": "Valencia"},
        "criterio": (
            "Todavía no: acelera su regreso pero está pendiente de pruebas que determinarán su "
            "evolución. Es una duda física (impacto negativo bajo), no un regreso confirmado."
        ),
        "referencia": (
            "Todavía no está confirmado: Sadiq acelera su regreso, pero está pendiente de nuevas "
            "pruebas este jueves que determinarán su evolución."
        ),
        "referencia_afirmaciones": [
            "Sadiq acelera su regreso.",
            "Sadiq está pendiente de nuevas pruebas.",
            "Su recuperación no está confirmada.",
        ],
        "docs_relevantes": ["4ad0d652a7c19a5f"],
        "adversarial": True,
        "tipo_dificultad": "regreso aparente",
        "por_que_dificil": "'Acelera su regreso' empuja a regreso_alta; no hay alta.",
    },
    {
        "eval_id": "M3-17",
        "pregunta": "¿Jens Cajuste ya puede jugar?",
        "esperado": {"category": "regreso_alta", "impact": "positivo_bajo", "team": "Malaga"},
        "criterio": (
            "Sí: Cajuste vuelve al grupo y podrá jugar tras un mes de lesión. Regreso, impacto "
            "positivo bajo para el Málaga."
        ),
        "referencia": "Sí. Jens Cajuste vuelve al grupo y podrá jugar tras un mes de lesión.",
        "referencia_afirmaciones": [
            "Cajuste vuelve al grupo.",
            "Cajuste podrá jugar.",
            "Estuvo un mes lesionado.",
        ],
        "docs_relevantes": ["25a07578dc93f4c8"],
        "adversarial": False,
        "tipo_dificultad": "normal",
        "por_que_dificil": "Control.",
    },
    {
        "eval_id": "M3-18",
        "pregunta": "¿Cómo está Musiala? ¿Puede volver a jugar con el Bayern?",
        "esperado": {
            "category": "regreso_alta",
            "impact": "positivo_bajo",
            "team": "Bayern Munich",
        },
        "criterio": (
            "Lo más reciente (8 de octubre): vuelve a ejercitarse tras el desgarro, aunque el "
            "Bayern no correrá riesgos. Regreso, impacto positivo bajo. Las noticias de agosto "
            "sobre su baja en la Supercopa están superadas."
        ),
        "referencia": (
            "Musiala vuelve a ejercitarse tras el desgarro que sufrió en la Nations League ante "
            "Países Bajos, aunque el Bayern no correrá riesgos con él."
        ),
        "referencia_afirmaciones": [
            "Musiala vuelve a ejercitarse.",
            "Musiala sufrió un desgarro en la Nations League ante Países Bajos.",
            "El Bayern no correrá riesgos con él.",
        ],
        "docs_relevantes": ["8969f60685def770"],
        "adversarial": True,
        "tipo_dificultad": "conflicto temporal",
        "por_que_dificil": (
            "Tres fragmentos de agosto dicen que Musiala es baja; el vigente es el de octubre."
        ),
    },
    # ------------------------------------------------------------------ rumor / declaración
    {
        "eval_id": "M3-19",
        "pregunta": "¿Erik Lira ya fichó por el Valencia?",
        "esperado": {
            "category": "rumor_no_confirmado",
            "impact": "neutro",
            "team": "Valencia",
        },
        "criterio": (
            "No hay fichaje: su agente se reunió con el director deportivo del Valencia y eso "
            "alimenta rumores. Es un rumor no confirmado; presentarlo como hecho es el error que "
            "más castiga la rúbrica."
        ),
        "referencia": (
            "No hay fichaje confirmado: el agente de Erik Lira se reunió con Braulio Vázquez en "
            "Paterna, lo que alimenta los rumores sobre su futuro."
        ),
        "referencia_afirmaciones": [
            "El fichaje de Erik Lira no está confirmado.",
            "Su agente se reunió con Braulio Vázquez en Paterna.",
            "La reunión alimenta rumores sobre su futuro.",
        ],
        "docs_relevantes": ["8615c6577f427a94"],
        "adversarial": True,
        "tipo_dificultad": "pregunta que presupone el hecho",
        "por_que_dificil": "La pregunta da el fichaje por hecho ('ya fichó').",
    },
    {
        "eval_id": "M3-20",
        "pregunta": "¿Qué dijo Arrasate en su vuelta a Osasuna?",
        "esperado": {
            "category": "declaracion_contexto",
            "impact": "neutro",
            "team": "Osasuna",
        },
        "criterio": (
            "Es una declaración del nuevo entrenador: siente que Osasuna lo necesita y el "
            "objetivo es la permanencia. Declaración/contexto, impacto neutro. 'Vuelta' no es un "
            "regreso de lesión."
        ),
        "referencia": (
            "Que siente que Osasuna lo necesita y que puede ayudar; el objetivo claro es la "
            "permanencia."
        ),
        "referencia_afirmaciones": [
            "Arrasate siente que Osasuna lo necesita.",
            "Arrasate cree que puede ayudar a Osasuna.",
            "El objetivo es la permanencia.",
        ],
        "docs_relevantes": ["62c4cd29be5f10d0"],
        "adversarial": True,
        "tipo_dificultad": "palabra trampa",
        "por_que_dificil": "'Vuelta'/'regreso' empujan a regreso_alta, pero es un técnico.",
    },
    # ------------------------------------------------------------------ fechas (herramienta)
    {
        "eval_id": "M3-21",
        "pregunta": "¿Estará Aaron Ochoa disponible para un partido el 1 de noviembre?",
        "esperado": {"category": "baja_confirmada", "impact": "negativo_alto", "team": "Malaga"},
        "criterio": (
            "La baja estimada es de unos tres meses desde el 27 de agosto: regreso hacia el 27 de "
            "noviembre, después del 1 de noviembre. Sigue siendo baja confirmada, impacto "
            "negativo alto."
        ),
        "referencia": (
            "No. Con una baja de alrededor de tres meses publicada el 27 de agosto de 2026, el "
            "regreso estimado es hacia el 27 de noviembre, después del 1 de noviembre."
        ),
        "referencia_afirmaciones": [
            "La baja de Aaron Ochoa es de alrededor de tres meses.",
            "La noticia se publicó el 27 de agosto de 2026.",
            "El regreso estimado es hacia el 27 de noviembre de 2026.",
            "El 1 de noviembre es anterior al regreso estimado.",
        ],
        "docs_relevantes": ["cfa48ba64563b824"],
        "necesita_herramienta": True,
        "adversarial": True,
        "tipo_dificultad": "aritmética de fechas",
        "por_que_dificil": (
            "Hay que sumar tres meses a la fecha de publicación y compararla. Además, el resumen "
            "RSS de ese fragmento habla del Girona por un error del feed: el club es el Málaga."
        ),
    },
    {
        "eval_id": "M3-22",
        "pregunta": "¿Podrá jugar Aaron Ochoa el 15 de diciembre?",
        "esperado": {"category": "duda_fisica", "impact": "negativo_bajo", "team": "Malaga"},
        "criterio": (
            "Par mínimo de M3-21. Con la estimación publicada (unos tres meses desde el 27 de "
            "agosto) ya debería estar de vuelta para el 15 de diciembre, pero no hay alta "
            "confirmada: duda física, impacto negativo bajo. Repetir 'está de baja' es una señal "
            "falsa de alto impacto."
        ),
        "referencia": (
            "Probablemente sí, pero sin confirmación: con la baja de unos tres meses publicada el "
            "27 de agosto de 2026, el regreso estimado es hacia el 27 de noviembre, antes del 15 "
            "de diciembre."
        ),
        "referencia_afirmaciones": [
            "La baja de Aaron Ochoa es de alrededor de tres meses.",
            "El regreso estimado es hacia el 27 de noviembre de 2026.",
            "El 15 de diciembre es posterior al regreso estimado.",
            "No hay un alta médica confirmada.",
        ],
        "docs_relevantes": ["cfa48ba64563b824"],
        "necesita_herramienta": True,
        "adversarial": True,
        "tipo_dificultad": "aritmética de fechas",
        "por_que_dificil": "La misma noticia que M3-21 lleva a la respuesta contraria.",
    },
    {
        "eval_id": "M3-23",
        "pregunta": "¿Puede contar el Oviedo con Chaira para el 29 de noviembre?",
        "esperado": {
            "category": "duda_fisica",
            "impact": "negativo_bajo",
            "team": "Real Oviedo",
        },
        "criterio": (
            "La baja era de entre dos y tres meses desde el 8 de agosto: regreso estimado entre el "
            "8 de octubre y el 8 de noviembre. Para el 29 de noviembre debería estar, pero sin "
            "confirmación: duda física, impacto negativo bajo."
        ),
        "referencia": (
            "Probablemente sí: la baja de Chaira, publicada el 8 de agosto de 2026, era de entre "
            "dos y tres meses, así que el regreso estimado es como tarde el 8 de noviembre, antes "
            "del 29 de noviembre. No hay alta confirmada."
        ),
        "referencia_afirmaciones": [
            "La baja de Chaira es de entre dos y tres meses.",
            "La noticia se publicó el 8 de agosto de 2026.",
            "El regreso estimado es como tarde el 8 de noviembre de 2026.",
            "El 29 de noviembre es posterior al regreso estimado.",
        ],
        "docs_relevantes": ["f538603932f7ce91"],
        "necesita_herramienta": True,
        "adversarial": True,
        "tipo_dificultad": "aritmética de fechas",
        "por_que_dificil": "Rango de duración (2-3 meses) y noticia de agosto.",
    },
    # ------------------------------------------------------------------ fuera del corpus
    {
        "eval_id": "M3-24",
        "pregunta": "¿Kylian Mbappé se pierde el Clásico?",
        "en_corpus": False,
        "criterio": (
            "El corpus no tiene ninguna noticia sobre la disponibilidad de Mbappé. La respuesta "
            "correcta es decir que no hay información y no emitir señal. Trasladarle la sanción "
            "de Vinicius es inventar una señal falsa de alto impacto."
        ),
        "adversarial": True,
        "tipo_dificultad": "fuera del corpus, casi-acierto",
        "por_que_dificil": (
            "'Se pierde el Clásico' es literal el titular de Vinicius: el retriever lo trae "
            "y el generador tiene la tentación de reusarlo."
        ),
    },
    {
        "eval_id": "M3-25",
        "pregunta": "¿Está lesionado Lamine Yamal?",
        "en_corpus": False,
        "criterio": (
            "El corpus menciona a Lamine pero nunca una lesión suya. La respuesta correcta es "
            "decir que no hay información y no emitir señal."
        ),
        "adversarial": True,
        "tipo_dificultad": "fuera del corpus, casi-acierto",
        "por_que_dificil": "Hay fragmentos con 'Lamine' (una roja que pudo ver, un rifirrafe).",
    },
    {
        "eval_id": "M3-26",
        "pregunta": "¿Qué cuota paga una victoria del Betis contra el Oporto?",
        "en_corpus": False,
        "criterio": (
            "El corpus no tiene cuotas (y Offside informa, no recomienda apostar). La respuesta "
            "correcta es decir que no hay información y no emitir señal."
        ),
        "adversarial": True,
        "tipo_dificultad": "fuera del corpus y del alcance",
        "por_que_dificil": "Hay un fragmento del Betis-Oporto (horario y TV) que atrae al retriever.",
    },
    {
        "eval_id": "M3-27",
        "pregunta": "¿Quién ganó el Balón de Oro 2025?",
        "en_corpus": False,
        "criterio": (
            "No está en el corpus. Aunque el modelo lo sepa de memoria, el sistema solo debe "
            "responder con las noticias recolectadas: la respuesta correcta es la válvula, sin "
            "señal."
        ),
        "adversarial": True,
        "tipo_dificultad": "fuera del corpus, conocimiento paramétrico",
        "por_que_dificil": "El generador puede saberlo de su preentrenamiento y 'responder bien' inventando fuente.",
    },
]


def construir(corpus_ids: set[str]) -> list[dict]:
    salida, errores = [], []
    for c in CASOS:
        en_corpus = c.get("en_corpus", True)
        esperado = c.get("esperado", SIN_SENAL) if en_corpus else dict(SIN_SENAL)
        if esperado["category"] not in CATEGORIES:
            errores.append(f"{c['eval_id']}: categoría {esperado['category']}")
        if esperado["impact"] not in IMPACTOS:
            errores.append(f"{c['eval_id']}: impacto {esperado['impact']}")
        for d in c.get("docs_relevantes", []):
            if d not in corpus_ids:
                errores.append(f"{c['eval_id']}: doc {d} no está en el corpus")
        if en_corpus and not c.get("docs_relevantes"):
            errores.append(f"{c['eval_id']}: en el corpus pero sin docs_relevantes")
        salida.append(
            {
                "eval_id": c["eval_id"],
                "input": {"text": c["pregunta"], "source": "usuario", "published_at": None},
                "esperado": esperado,
                "criterio": c["criterio"],
                "referencia": c.get("referencia", VALVULA),
                "referencia_afirmaciones": c.get("referencia_afirmaciones", []),
                "docs_relevantes": c.get("docs_relevantes", []),
                "en_corpus": en_corpus,
                "necesita_herramienta": c.get("necesita_herramienta", False),
                "adversarial": c["adversarial"],
                "tipo_dificultad": c["tipo_dificultad"],
                "por_que_dificil": c["por_que_dificil"],
            }
        )
    ids = [s["eval_id"] for s in salida]
    if len(ids) != len(set(ids)):
        errores.append("eval_id duplicado")
    if errores:
        raise SystemExit("eval set inválido:\n  " + "\n  ".join(errores))
    return salida


def main() -> int:
    corpus = HERE.parent / "data" / "rag" / "corpus.jsonl"
    corpus_ids = {json.loads(linea)["id"] for linea in corpus.open(encoding="utf-8")}
    casos = construir(corpus_ids)
    out = HERE / "eval_set_m3.json"
    out.write_text(json.dumps(casos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fuera = sum(not c["en_corpus"] for c in casos)
    adv = sum(c["adversarial"] for c in casos)
    cats = {c["esperado"]["category"] for c in casos}
    herr = sum(c["necesita_herramienta"] for c in casos)
    print(f"{len(casos)} casos | {fuera} fuera del corpus (válvula) | {adv} adversariales")
    print(f"{len(cats)}/8 categorías | {herr} necesitan la herramienta de fechas")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
