from unittest.mock import MagicMock
import pytest
from langchain_core.language_models import BaseChatModel

from src.agents.contextualizer import ContextualizationAgent
from src.agents.extractor import ExtractionAgent
from src.models import SectionMapping, ContractChangeOutput

def test_contextualization_agent():
    # Mock del LLM inyectado
    mock_llm = MagicMock(spec=BaseChatModel)
    
    # Mock de la cadena retornada por with_structured_output()
    mock_chain = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    
    # Mock de la respuesta esperada
    mock_response = MagicMock()
    mock_response.mappings = [
        SectionMapping(section_name="Sección 1", original_text="A", amended_text="B")
    ]
    mock_chain.invoke.return_value = mock_response
    
    # Instanciar y ejecutar el agente
    agent = ContextualizationAgent(llm=mock_llm)
    result = agent.run("Texto original A", "Texto adenda B")
    
    # Validaciones
    assert len(result) == 1
    assert result[0].section_name == "Sección 1"
    
    # Verificar que with_structured_output fue llamado
    mock_llm.with_structured_output.assert_called_once()
    # Verificar que invoke fue llamado
    mock_chain.invoke.assert_called_once()


def test_extraction_agent():
    # Mock del LLM inyectado
    mock_llm = MagicMock(spec=BaseChatModel)
    
    # Mock de la cadena retornada por with_structured_output()
    mock_chain = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    
    # Mock de la respuesta esperada
    mock_response = ContractChangeOutput(
        sections_changed=["Sección 1"],
        topics_touched=["General"],
        summary_of_the_change="Se modificó A por B"
    )
    mock_chain.invoke.return_value = mock_response
    
    # Instanciar y ejecutar el agente
    agent = ExtractionAgent(llm=mock_llm)
    mappings = [
        SectionMapping(section_name="Sección 1", original_text="A", amended_text="B")
    ]
    result = agent.run(mappings)
    
    # Validaciones
    assert isinstance(result, ContractChangeOutput)
    assert result.sections_changed == ["Sección 1"]
    
    # Verificar que with_structured_output fue llamado
    mock_llm.with_structured_output.assert_called_once()
    # Verificar que invoke fue llamado
    mock_chain.invoke.assert_called_once()
