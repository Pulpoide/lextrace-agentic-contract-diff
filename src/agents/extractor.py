"""Agente 2 — Detective 🔍

Recibe el mapa de correspondencias del Cartógrafo y analiza cada par
de secciones para identificar cambios específicos. Produce el output
final validado por Pydantic: ContractChangeOutput.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.models import SectionMapping, ContractChangeOutput


SYSTEM_PROMPT = """\
Eres el "Detective" 🔍, un experto en auditoría legal y análisis de contratos.

Tu misión es identificar diferencias sustanciales entre un contrato original y su adenda. 
Recibirás pares de secciones ya mapeadas por el "Cartógrafo".

### INSTRUCCIONES DE ANÁLISIS:
1. **Pensamiento Analítico**: Para cada par, compara el significado legal. Ignora cambios de puntuación o formato que no alteren las obligaciones.
2. **Clasificación de Cambios**:
    - **Adición**: Cláusulas o frases nuevas que expanden obligaciones o derechos.
    - **Eliminación**: Texto removido que reduce o anula términos anteriores.
    - **Modificación**: Cambio en valores, fechas, nombres o condiciones de una cláusula existente.
3. **Vocabulario Controlado**: En 'topics_touched', intenta usar categorías estándar como: [Financiero, Plazos, Soporte Técnico, Responsabilidad Legal, Confidencialidad, Terminación, Propiedad Intelectual].

### REGLAS DE SALIDA:
- **sections_changed**: Solo incluye los nombres de las secciones donde hubo cambios REALES.
- **summary_of_the_change**: Redacta para un humano. Usa un tono profesional y directo. Ejemplo: "Se incrementó el honorario mensual de USD 8.000 a USD 9.500 y se redujo la frecuencia de reportes de mensual a quincenal".
"""


class ExtractionAgent:
    """Agente Detective: analiza pares de secciones y detecta cambios."""

    def __init__(self, openai_api_key: str | None = None, callbacks: list | None = None):
        """Inicializa el agente con modelo GPT-4o y structured output.

        Args:
            openai_api_key: API key de OpenAI. Si es None, se lee de OPENAI_API_KEY env var.
            callbacks: Lista de callbacks (e.g. Langfuse) para tracking.
        """
        self.callbacks = callbacks or []
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=4096,
            api_key=openai_api_key,
        ).with_structured_output(ContractChangeOutput)

    def run(self, mappings: list[SectionMapping]) -> ContractChangeOutput:
        buffer = [
            "Analiza los siguientes pares de secciones correspondientes entre el contrato original y la adenda:\n"
        ]

        for i, mapping in enumerate(mappings, 1):
            section_info = (
                f"### Par {i}: {mapping.section_name}\n"
                f"**Original:** {mapping.original_text or '(no existe)'}\n"
                f"**Adenda:** {mapping.amended_text or '(eliminada)'}\n"
                "---"
            )
            buffer.append(section_info)

        human_content = "\n\n".join(buffer)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        return self.llm.invoke(messages, config={"callbacks": self.callbacks})
