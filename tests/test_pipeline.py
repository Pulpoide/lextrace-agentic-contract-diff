from unittest.mock import MagicMock
import pytest

from src.pipeline import PipelineOrchestrator
from src.models import SectionMapping, ContractChangeOutput

def test_pipeline_orchestrator(mocker):
    # Mockear a los agentes
    mock_contextualizer_class = mocker.patch("src.pipeline.ContextualizationAgent")
    mock_extractor_class = mocker.patch("src.pipeline.ExtractionAgent")
    
    # Crear mocks de las instancias
    mock_contextualizer_instance = MagicMock()
    mock_extractor_instance = MagicMock()
    
    # Configurar los valores de retorno de run()
    mappings = [SectionMapping(section_name="General", original_text="A", amended_text="B")]
    mock_contextualizer_instance.run.return_value = mappings
    
    final_output = ContractChangeOutput(
        sections_changed=["General"],
        topics_touched=["Tópico 1"],
        summary_of_the_change="Cambió algo"
    )
    mock_extractor_instance.run.return_value = final_output
    
    # Asignar instancias a las clases mockeadas
    mock_contextualizer_class.return_value = mock_contextualizer_instance
    mock_extractor_class.return_value = mock_extractor_instance
    
    # Mockear el callback
    mock_callback = MagicMock()
    
    # Ejecutar pipeline
    orchestrator = PipelineOrchestrator(api_key="fake-key")
    result = orchestrator.run_analysis(
        original_text="Texto A",
        amendment_text="Texto B",
        on_mapping_complete=mock_callback
    )
    
    # Validaciones
    assert result == final_output
    
    # Verificar interacciones
    mock_contextualizer_instance.run.assert_called_once_with("Texto A", "Texto B")
    mock_extractor_instance.run.assert_called_once_with(mappings)
    mock_callback.assert_called_once_with(mappings)
