"""Orquestador central del pipeline de análisis de contratos."""

from typing import Callable, List, Optional

from langchain_openai import ChatOpenAI
from src.agents.contextualizer import ContextualizationAgent
from src.agents.extractor import ExtractionAgent
from src.models import ContractChangeOutput, SectionMapping


class PipelineOrchestrator:
    """Coordina la ejecución de los agentes del pipeline."""

    def __init__(self, api_key: str, callbacks: Optional[List] = None):
        self.api_key = api_key
        self.callbacks = callbacks or []
        self.llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o",
            temperature=0,
        )

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
            llm=self.llm, callbacks=self.callbacks
        )
        section_mappings = cartographer.run(original_text, amendment_text)

        if on_mapping_complete:
            on_mapping_complete(section_mappings)

        # Paso 2: Agente Detective
        detective = ExtractionAgent(
            llm=self.llm, callbacks=self.callbacks
        )
        result = detective.run(section_mappings)

        return result

    def run_cartographer(
        self, original_text: str, amendment_text: str
    ) -> list[SectionMapping]:
        """Execute Step 2: Cartographer agent for section mapping."""
        cartographer = ContextualizationAgent(
            llm=self.llm, callbacks=self.callbacks
        )
        return cartographer.run(original_text, amendment_text)

    def run_extractor(
        self, mappings: list[SectionMapping]
    ) -> ContractChangeOutput:
        """Execute Step 3: Detective agent for change extraction."""
        detective = ExtractionAgent(llm=self.llm, callbacks=self.callbacks)
        return detective.run(mappings)
