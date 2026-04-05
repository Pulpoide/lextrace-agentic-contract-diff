"""Agente 1 — Cartógrafo 🗺️

Construye un mapa de correspondencias entre secciones del contrato
original y la adenda. NO analiza cambios, solo alinea secciones.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.models import SectionMapping
from pydantic import BaseModel, Field


SYSTEM_PROMPT = """\
Eres el "Cartógrafo" 🗺️, un experto en análisis estructural de documentos legales.

Tu ÚNICA tarea es construir un MAPA DE CORRESPONDENCIAS entre las secciones del \
contrato original y las secciones de la adenda/enmienda.

Reglas estrictas:
1. Identifica cada sección, cláusula o artículo en AMBOS documentos.
2. Para cada sección del original, encuentra su correspondiente en la adenda.
3. Si una sección existe solo en la adenda (nueva), el campo "original_text" debe estar vacío.
4. Si una sección del original fue eliminada en la adenda, el campo "amended_text" debe estar vacío.
5. NO analices ni describas los cambios. Solo alinea las secciones.

Responde EXCLUSIVAMENTE con un JSON array donde cada elemento tiene:
- "section_name": nombre o número de la sección
- "original_text": texto de esa sección en el original (o "" si no existe)
- "amended_text": texto de esa sección en la adenda (o "" si no existe)

Si el nombre de una sección cambia levemente (ej. 'Precios' a 'Tarifas') pero el propósito es el mismo, considératelas correspondientes y agrúpalas en el mismo objeto.

No incluyas explicaciones, solo responde con el JSON array."""


class SectionMappingList(BaseModel):
    mappings: list[SectionMapping] = Field(
        description="Lista de correspondencias entre secciones"
    )


class ContextualizationAgent:
    def __init__(self, openai_api_key: str | None = None, callbacks: list | None = None):
        self.callbacks = callbacks or []
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=openai_api_key,
        ).with_structured_output(SectionMappingList)

    def run(self, original_text: str, amended_text: str) -> list[SectionMapping]:
        human_content = (
            f"## CONTRATO ORIGINAL:\n{original_text}\n\n## ADENDA:\n{amended_text}"
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        response = self.llm.invoke(messages, config={"callbacks": self.callbacks})
        return response.mappings
