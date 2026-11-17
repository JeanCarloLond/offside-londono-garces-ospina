# Criterio de aceptación de la técnica de eficiencia

Escrito el 2026-11-16, **antes** de medir nada. La fecha del commit lo respalda.

**Técnica:** cuantizar a int4 (bitsandbytes NF4) el generador `Qwen/Qwen2.5-1.5B-Instruct`, que es
también el modelo de la válvula verificada. Es el componente que más tarda y más memoria ocupa en el
sistema del Módulo 3: casi toda la latencia de una consulta es generar con él.

**Qué aceptamos perder:**

> Aceptamos perder como máximo una pregunta del eval set (unos 0,04 de tasa accionable, la dimensión 3
> del harness) y **ningún** caso de la válvula, si la cuantización a int4 reduce a la mitad la memoria
> del generador o baja la latencia p50 al menos un 30 %.

**Cómo se mide:** el mismo eval set de 27 preguntas del M3, el mismo harness de M2, en la misma
máquina (una GTX 1650 Ti de 4 GB) y con el mismo prompt base. Antes = el sistema final del M3 en fp16;
después = el mismo sistema con el generador en int4. Se reportan las tres dimensiones del harness,
latencia p50 y p95, y memoria del generador. Sobre la versión int4 se repiten las pruebas que no miden
exactitud: la válvula (las 4 preguntas sin respuesta) y los casos adversariales.

Al final, en `final/reporte.md`, se dice si el criterio se cumplió.
