"""Orquestador central del pipeline de análisis de contratos."""

from typing import Callable, List, Optional

from src.agents.contextualizer import ContextualizationAgent
from src.agents.extractor import ExtractionAgent
from src.models import ContractChangeOutput, SectionMapping


class PipelineOrchestrator:
    """Coordina la ejecución de los agentes del pipeline."""

    def __init__(self, api_key: str, callbacks: Optional[List] = None):
        self.api_key = api_key
        self.callbacks = callbacks or []

    def run_analysis(
        self,
        original_text: str,
        amendment_text: str,
        on_mapping_complete: Optional[Callable[[List[SectionMapping]], None]] = None,
    ) -> ContractChangeOutput:
        """
        Ejecuta el análisis comparativo entre el texto original y la adenda.

        Args:
            original_text: Texto completo del contrato original.
            amendment_text: Texto completo de la adenda.
            on_mapping_complete: Callback opcional ejecutado tras el paso 1 (Cartógrafo),
                                 útil para logging o actualizaciones de UI.

        Returns:
            ContractChangeOutput: Objeto validado con el análisis de cambios.
        """
        # Paso 1: Agente Cartógrafo
        cartographer = ContextualizationAgent(
            openai_api_key=self.api_key, callbacks=self.callbacks
        )
        section_mappings = cartographer.run(original_text, amendment_text)

        if on_mapping_complete:
            on_mapping_complete(section_mappings)

        # Paso 2: Agente Detective
        detective = ExtractionAgent(
            openai_api_key=self.api_key, callbacks=self.callbacks
        )
        result = detective.run(section_mappings)

        return result
