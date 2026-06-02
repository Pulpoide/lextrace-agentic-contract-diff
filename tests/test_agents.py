from unittest.mock import MagicMock
import pytest

from src.agents.contextualizer import ContextualizationAgent
from src.agents.extractor import ExtractionAgent
from src.models import SectionMapping, ContractChangeOutput

def test_contextualization_agent(mocker):
    # Setup del mock
    mock_chat = mocker.patch("src.agents.contextualizer.ChatOpenAI")
    
    # Crear la estructura de retorno para with_structured_output().invoke()
    mock_runnable = MagicMock()
    
    # La salida esperada es una instancia de algo con 'mappings' (SectionMappingList en el original, pero el agente retorna response.mappings)
    mock_response = MagicMock()
    mock_response.mappings = [
        SectionMapping(section_name="Sección 1", original_text="A", amended_text="B")
    ]
    mock_runnable.invoke.return_value = mock_response
    
    # Hacer que ChatOpenAI().with_structured_output() devuelva mock_runnable
    mock_instance = MagicMock()
    mock_instance.with_structured_output.return_value = mock_runnable
    mock_chat.return_value = mock_instance
    
    # Instanciar y ejecutar el agente
    agent = ContextualizationAgent()
    result = agent.run("Texto original A", "Texto adenda B")
    
    # Validaciones
    assert len(result) == 1
    assert result[0].section_name == "Sección 1"
    
    # Verificar llamadas
    mock_runnable.invoke.assert_called_once()


def test_extraction_agent(mocker):
    mock_chat = mocker.patch("src.agents.extractor.ChatOpenAI")
    
    mock_runnable = MagicMock()
    mock_response = ContractChangeOutput(
        sections_changed=["Sección 1"],
        topics_touched=["General"],
        summary_of_the_change="Se modificó A por B"
    )
    mock_runnable.invoke.return_value = mock_response
    
    mock_instance = MagicMock()
    mock_instance.with_structured_output.return_value = mock_runnable
    mock_chat.return_value = mock_instance
    
    agent = ExtractionAgent()
    mappings = [
        SectionMapping(section_name="Sección 1", original_text="A", amended_text="B")
    ]
    result = agent.run(mappings)
    
    assert isinstance(result, ContractChangeOutput)
    assert result.sections_changed == ["Sección 1"]
    
    mock_runnable.invoke.assert_called_once()
