"""Registro estructurado de cada petición y envío a Weights & Biases.

Cada petición deja una línea JSON con la hora, la latencia, qué piezas se usaron (RAG,
herramienta, visión) y si se activó la válvula. Va a tres sitios:
  - la salida estándar, que es lo que muestra la pestaña "Logs" del Space;
  - un archivo local (registros.jsonl), que se pierde si el Space se reinicia;
  - W&B, si existe el secreto WANDB_API_KEY. Sin él, el sistema funciona igual.

Sin datos personales: no se guarda el texto de la pregunta ni la imagen. Solo su longitud,
si había imagen, y la señal que dio el sistema (categoría, impacto), que es pública.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

ARCHIVO = Path(os.getenv("OFFSIDE_REGISTROS", "registros.jsonl"))
PROYECTO_WANDB = os.getenv("WANDB_PROJECT", "offside")


class Observador:
    def __init__(self, version: str):
        self.version = version
        self.total = 0
        self.valvulas = 0
        self.herramienta = 0
        self.vision = 0
        self.errores = 0
        self._lock = threading.Lock()
        self.wandb = None
        if os.getenv("WANDB_API_KEY"):
            try:
                import wandb

                self.wandb = wandb.init(
                    project=PROYECTO_WANDB,
                    name=f"space-{datetime.now(UTC):%Y%m%d-%H%M}",
                    job_type="produccion",
                    config={"version": version},
                )
            except Exception as e:  # W&B nunca debe tumbar el sistema
                print(json.dumps({"evento": "wandb_no_disponible", "error": str(e)}), flush=True)

    def registrar(
        self,
        latencia_s: float,
        usa_rag: bool,
        usa_herramienta: bool,
        usa_vision: bool,
        valvula: bool,
        categoria: str,
        impacto: str,
        largo_pregunta: int,
        error: str | None = None,
        tiempos: dict | None = None,
    ) -> dict:
        with self._lock:
            self.total += 1
            self.valvulas += valvula
            self.herramienta += usa_herramienta
            self.vision += usa_vision
            self.errores += error is not None
            registro = {
                "id": uuid.uuid4().hex[:12],
                "hora_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "latencia_s": round(latencia_s, 3),
                "componentes": {
                    "rag": usa_rag,
                    "herramienta": usa_herramienta,
                    "vision": usa_vision,
                },
                "valvula": valvula,
                "categoria": categoria,
                "impacto": impacto,
                "largo_pregunta": largo_pregunta,
                "version": self.version,
                "error": error,
                "tiempos_s": tiempos or {},
            }
            linea = json.dumps(registro, ensure_ascii=False)
            print(linea, flush=True, file=sys.stdout)
            try:
                with ARCHIVO.open("a", encoding="utf-8") as f:
                    f.write(linea + "\n")
            except OSError:
                pass
            if self.wandb is not None:
                try:
                    self.wandb.log(
                        {
                            "latencia_s": latencia_s,
                            "valvula": int(valvula),
                            "usa_herramienta": int(usa_herramienta),
                            "usa_vision": int(usa_vision),
                            "error": int(error is not None),
                            # métricas de uso, acumuladas
                            "peticiones_total": self.total,
                            "tasa_valvula": self.valvulas / self.total,
                            "tasa_vision": self.vision / self.total,
                        }
                    )
                except Exception:
                    pass
            return registro

    def resumen(self) -> dict:
        return {
            "peticiones_total": self.total,
            "valvulas": self.valvulas,
            "con_herramienta": self.herramienta,
            "con_vision": self.vision,
            "errores": self.errores,
            "wandb_activo": self.wandb is not None,
        }
