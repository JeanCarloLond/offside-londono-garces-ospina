"""Componente visual: lee una captura y la convierte en texto para el RAG.

Patrón "visión como entrada" (sesión 11). El apostador recibe la información a
menudo como imagen: la captura de un tuit del club, de un parte médico o de una
alineación. El modelo de visión transcribe lo relevante y el RAG del M3 busca en el
corpus si alguna noticia lo respalda. La captura entra al generador marcada como
"sin verificar" (proyecto1/rag/sistemas.py): una imagen no es una fuente confirmada.

Modelo: Qwen/Qwen2.5-VL-3B-Instruct, que lee bien texto en español dentro de una
imagen. En el Space corre en bf16 sobre ZeroGPU; en una GPU de 4 GB, cuantizado a
4 bits (cuantizar=True).

Válvula visual: si la imagen no es de fútbol o no se lee, el modelo responde
SIN_INFORMACION y el sistema lo dice en vez de inventar.
"""

from __future__ import annotations

from dataclasses import dataclass

MODELO_VISION = "Qwen/Qwen2.5-VL-3B-Instruct"
SIN_INFO = "SIN_INFORMACION"
# Las capturas se reducen antes de entrar al modelo: el texto de una captura se lee
# bien a este tamaño, y cada píxel de más cuesta tiempo.
MAX_PIXELES = 1024 * 28 * 28

INSTRUCCION = (
    "La imagen debería ser una captura relacionada con fútbol: una noticia, un mensaje de un "
    "club, un parte médico o una alineación. Transcribe en español el texto relevante de la "
    "imagen y, en una frase, di qué hecho cuenta (qué jugador, qué club y qué pasa). No añadas "
    "nada que no esté en la imagen. Si la imagen no tiene que ver con fútbol o no se puede "
    f"leer, responde exactamente: {SIN_INFO}"
)


@dataclass
class Lectura:
    texto: str
    es_futbol: bool


class Vision:
    def __init__(
        self, modelo: str = MODELO_VISION, device: str | None = None, cuantizar: bool = False
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.modelo_id = modelo
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(modelo, max_pixels=MAX_PIXELES)
        if cuantizar:
            from transformers import BitsAndBytesConfig

            q = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                modelo, quantization_config=q, device_map=self.device
            )
        else:
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForImageTextToText.from_pretrained(modelo, dtype=dtype).to(
                self.device
            )
        self.model.eval()

    def leer(self, imagen) -> Lectura:
        """imagen: una PIL.Image. Devuelve el texto leído y si es de fútbol."""
        mensajes = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": INSTRUCCION}],
            }
        ]
        prompt = self.processor.apply_chat_template(mensajes, add_generation_prompt=True)
        entradas = self.processor(
            text=[prompt], images=[imagen.convert("RGB")], return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad():
            salida = self.model.generate(
                **entradas,
                max_new_tokens=160,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        texto = self.processor.batch_decode(
            salida[:, entradas["input_ids"].shape[1] :], skip_special_tokens=True
        )[0].strip()
        es_futbol = SIN_INFO not in texto.upper().replace(" ", "_") and len(texto) > 10
        return Lectura(texto=texto if es_futbol else "", es_futbol=es_futbol)
