graph TD

    A[Imágenes: Contrato + Enmienda] --> B[parse_contract_image: GPT-4o Vision]
    B -->|Texto Plano| C[ContextualizationAgent: Agente 1]
    C -->|Mapa de Correspondencias| D[ExtractionAgent: Agente 2]
    D -->|Raw JSON| E[Validación Pydantic: ContractChangeOutput]
    E -->|JSON Final Estructurado| F[Output Usuario]
    subgraph Observabilidad (Langfuse)
    B -.-> L[Trace: contract-analysis]
    C -.-> L
    D -.-> L
    E -.-> L
    
end