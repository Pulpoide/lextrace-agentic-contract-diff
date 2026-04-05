"""Pydantic v2 data models for the LexTrace contract analysis pipeline."""

from pydantic import BaseModel, Field


class ExtractedText(BaseModel):
    """Texto extraído de una imagen de contrato."""

    source: str = Field(description="Identificador de la fuente (e.g. 'original', 'amendment')")
    content: str = Field(description="Texto completo extraído de la imagen")


class SectionMapping(BaseModel):
    """Mapa de correspondencia entre una sección del original y la adenda.

    Producido por el Agente 1 (Cartógrafo). Solo alinea secciones,
    NO analiza cambios.
    """

    section_name: str = Field(description="Nombre o número de la sección")
    original_text: str = Field(description="Texto de la sección en el contrato original")
    amended_text: str = Field(description="Texto correspondiente en la adenda")



class ContractChangeOutput(BaseModel):
    """Esquema de salida final validado por Pydantic.

    Contiene el resumen estructurado de todos los cambios
    entre el contrato original y la adenda.
    """

    sections_changed: list[str] = Field(
        description="Lista de secciones que fueron modificadas"
    )
    topics_touched: list[str] = Field(
        description="Temas o conceptos afectados por los cambios"
    )
    summary_of_the_change: str = Field(
        description="Resumen general de los cambios en lenguaje natural"
    )
