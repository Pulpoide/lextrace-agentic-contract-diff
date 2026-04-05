"""Utilidades para procesamiento de imágenes de contratos via GPT-4o Vision."""

import base64
import logging
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


logger = logging.getLogger(__name__)


def encode_image_to_base64(image_path: str) -> str:
    """Lee un archivo de imagen y retorna su representación en base64.

    Args:
        image_path: Ruta al archivo de imagen.

    Returns:
        String base64 de la imagen.

    Raises:
        FileNotFoundError: Si la imagen no existe.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_contract_image(image_path: str, callbacks: list, api_key: str) -> str:
    """Extrae texto de una imagen de contrato usando GPT-4o Vision.

    Args:
        image_path: Ruta al archivo de imagen del contrato.
        callbacks: Lista de callbacks (e.g. Langfuse CallbackHandler)
                   que captura automáticamente tokens y metadata.

    Returns:
        Texto extraído de la imagen del contrato.
    """
    try:
        base64_image = encode_image_to_base64(image_path)

        # Detectar tipo MIME según extensión
        suffix = Path(image_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/png")

        llm = ChatOpenAI(
            model="gpt-4o",
            max_tokens=4096,
            temperature=0,
            api_key=api_key,
        )

        system_instructions = (
            "Eres un experto en transcripción de documentos legales. "
            "Tu única tarea es extraer TODO el texto visible de esta imagen. "
            "REGLAS ESTRICTAS:\n"
            "1. Mantén la estructura original usando formato Markdown (títulos con #, listas con -, etc.).\n"
            "2. Transcribe fielmente. No corrijas errores ortográficos del original.\n"
            "3. NO incluyas saludos, introducciones ni conclusiones. Devuelve ÚNICAMENTE el texto extraído."
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": system_instructions},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": "high",
                    },
                },
            ]
        )

        response = llm.invoke([message], config={"callbacks": callbacks})
        return response.content

    except Exception as e:
        logger.error(f"Error procesando la imagen {image_path}: {str(e)}")
        raise RuntimeError(f"Fallo en la extracción OCR: {str(e)}")
