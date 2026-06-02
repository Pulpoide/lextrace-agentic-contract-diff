import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.utils.image_processor import encode_image_to_base64, parse_contract_image


@pytest.fixture
def sample_image_path():
    """Ruta a la imagen de prueba."""
    base_dir = Path(__file__).parent
    return str(base_dir / "fixtures" / "sample_ocr.png")


def test_encode_image_to_base64(sample_image_path):
    """Testea que la codificación base64 funcione correctamente."""
    encoded = encode_image_to_base64(sample_image_path)
    assert isinstance(encoded, str)
    assert len(encoded) > 0


def test_encode_image_not_found():
    """Testea que levante FileNotFoundError si la imagen no existe."""
    with pytest.raises(FileNotFoundError):
        encode_image_to_base64("ruta/falsa/imagen.png")


def test_parse_contract_image(mocker, sample_image_path):
    """Testea la extracción de texto mockeando ChatOpenAI."""
    # Mockear la clase ChatOpenAI
    mock_chat_openai_class = mocker.patch("src.utils.image_processor.ChatOpenAI")
    
    # Configurar el mock de la instancia
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Texto extraído simulado."
    mock_instance.invoke.return_value = mock_response
    
    mock_chat_openai_class.return_value = mock_instance

    callbacks = []
    api_key = "fake-key"

    result = parse_contract_image(sample_image_path, callbacks, api_key)

    assert result == "Texto extraído simulado."
    
    # Validar que invoke fue llamado exactamente 1 vez
    mock_instance.invoke.assert_called_once()
    
    # Inspeccionar los argumentos con los que fue llamado invoke
    args, kwargs = mock_instance.invoke.call_args
    
    messages = args[0]
    assert len(messages) == 1
    
    human_message = messages[0]
    content = human_message.content
    
    # content debe ser una lista con 2 elementos: texto de instrucciones e imagen url
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert "data:image/png;base64," in content[1]["image_url"]["url"]

