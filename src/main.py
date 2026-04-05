"""LexTrace — Multi-Agent Contract Analysis System.

Orchestrator que coordina el pipeline completo:
1. Extracción de texto via GPT-4o Vision
2. Agente 1 (Cartógrafo): mapa de correspondencias
3. Agente 2 (Detective): detección de cambios
4. Validación Pydantic y output JSON
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler  # Langfuse v3+
except ImportError:
    from langfuse.callback import CallbackHandler  # Langfuse v2 fallback

    get_client = None

from src.models import ExtractedText
from src.utils.image_processor import parse_contract_image
from src.agents.contextualizer import ContextualizationAgent
from src.agents.extractor import ExtractionAgent


def run_pipeline(original_path: str, amendment_path: str) -> None:
    """Ejecuta el pipeline completo de análisis de contratos.

    Args:
        original_path: Ruta a la imagen del contrato original.
        amendment_path: Ruta a la imagen de la adenda/enmienda.
    """
    for img_path in [original_path, amendment_path]:
        if not Path(img_path).exists():
            print(f"[ERROR] Imagen no encontrada: {img_path}")
            sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY no encontrada en .env")
        sys.exit(1)

    langfuse_handler = CallbackHandler()
    callbacks = [langfuse_handler]

    try:
        print("LexTrace -- Iniciando analisis...")

        # ──────────────────────────────────────────────
        # Paso 1: Extraer texto de ambas imágenes
        # ──────────────────────────────────────────────
        print("\n[Paso 1] Extrayendo texto de las imagenes...")

        print(f"   -> Procesando contrato original: {original_path}")
        original_text = parse_contract_image(original_path, callbacks, api_key=api_key)
        original = ExtractedText(source="original", content=original_text)
        print(f"   [OK] Original: {len(original.content)} caracteres extraidos")

        print(f"   -> Procesando adenda: {amendment_path}")
        amendment_text = parse_contract_image(amendment_path, callbacks, api_key=api_key)
        amendment = ExtractedText(source="amendment", content=amendment_text)
        print(f"   [OK] Adenda: {len(amendment.content)} caracteres extraidos")

        # ──────────────────────────────────────────────
        # Paso 2: Agente 1 — Cartógrafo
        # ──────────────────────────────────────────────
        print("\n[Paso 2] Agente Cartografo -- Mapeando correspondencias...")
        cartographer = ContextualizationAgent(openai_api_key=api_key, callbacks=callbacks)
        mappings = cartographer.run(original.content, amendment.content)
        print(f"   [OK] {len(mappings)} secciones mapeadas")

        for m in mappings:
            if m.original_text and m.amended_text:
                status = "[MOD]"
            elif not m.original_text:
                status = "[NEW]"
            else:
                status = "[DEL]"
            print(f"      {status} {m.section_name}")

        # ──────────────────────────────────────────────
        # Paso 3: Agente 2 — Detective
        # ──────────────────────────────────────────────
        print("\n[Paso 3] Agente Detective -- Analizando cambios...")
        detective = ExtractionAgent(openai_api_key=api_key, callbacks=callbacks)
        result = detective.run(mappings)
        print("   [OK] Analisis completado")

        # ──────────────────────────────────────────────
        # Paso 4: Output JSON validado
        # ──────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("RESULTADO FINAL")
        print("=" * 60)
        print(result.model_dump_json(indent=2))
        print(
            "\n[OK] Pipeline completado. Revisa Langfuse para el trace."
        )

    except Exception as e:
        print(f"\n[ERROR CRITICO] durante el pipeline: {str(e)}")
        sys.exit(1)
    finally:
        if get_client is not None:
            get_client().flush()


def main():
    """Entry point CLI."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="LexTrace -- Multi-Agent Contract Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplo de uso:\n"
            "  python -m src.main data/original.png data/amendment.png\n"
        ),
    )
    parser.add_argument(
        "original",
        type=str,
        help="Ruta a la imagen del contrato original",
    )
    parser.add_argument(
        "amendment",
        type=str,
        help="Ruta a la imagen de la adenda/enmienda",
    )

    args = parser.parse_args()
    run_pipeline(args.original, args.amendment)


if __name__ == "__main__":
    main()
