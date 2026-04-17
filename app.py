"""
app.py
──────
LexTrace — Streamlit interface for multi-agent contract analysis.

Run with:
    streamlit run app.py
"""

from __future__ import annotations
from pathlib import Path
import os
import streamlit as st
import tempfile

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    # Langfuse v3+
    from langfuse import get_client, propagate_attributes
    from langfuse.langchain import CallbackHandler
except ImportError:
    # Langfuse v2 fallback
    from langfuse.callback import CallbackHandler  # type: ignore

    get_client = None
    propagate_attributes = None

from pydantic import ValidationError
from openai import RateLimitError, APITimeoutError

from src.agents.contextualizer import ContextualizationAgent
from src.agents.extractor import ExtractionAgent
from src.utils.image_processor import parse_contract_image
from src.models import ContractChangeOutput

# ---------------------------------------------------------------------------
# Page config  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LexTrace — Análisis de Contratos",
    page_icon="⚖️",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

MAX_HOJAS = 5

_DEFAULTS: dict = {
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "langfuse_public_key": os.getenv("LANGFUSE_PUBLIC_KEY", ""),
    "langfuse_secret_key": os.getenv("LANGFUSE_SECRET_KEY", ""),
    "langfuse_host": os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
    "analysis_result": None,
    "num_sheets_orig": 1,
    "num_sheets_add": 1,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Sidebar — API keys
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔑 Credenciales")
    st.caption("Guardadas en memoria de sesión. No se persisten en disco ni en `.env`.")

    st.session_state["openai_api_key"] = st.text_input(
        "OpenAI API Key",
        value=st.session_state["openai_api_key"],
        type="password",
        placeholder="sk-…",
    )
    st.session_state["langfuse_secret_key"] = st.text_input(
        "Langfuse Secret Key",
        value=st.session_state["langfuse_secret_key"],
        type="password",
        placeholder="sk-lf-…",
    )
    st.session_state["langfuse_public_key"] = st.text_input(
        "Langfuse Public Key",
        value=st.session_state["langfuse_public_key"],
        type="password",
        placeholder="pk-lf-…",
    )
    st.session_state["langfuse_host"] = st.text_input(
        "Langfuse Host",
        value=st.session_state["langfuse_host"],
        placeholder="https://us.cloud.langfuse.com",
    )

    st.divider()

    _keys_ready: bool = all(
        st.session_state[k]
        for k in ("openai_api_key", "langfuse_public_key", "langfuse_secret_key")
    )

    if _keys_ready:
        st.success("✅ Credenciales configuradas")
    else:
        st.warning("⚠️ Completá las tres keys para habilitar el análisis")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_langfuse_env() -> None:
    """Inyecta credenciales Langfuse como variables de entorno."""
    os.environ["LANGFUSE_PUBLIC_KEY"] = st.session_state["langfuse_public_key"]
    os.environ["LANGFUSE_SECRET_KEY"] = st.session_state["langfuse_secret_key"]
    os.environ["LANGFUSE_HOST"] = st.session_state["langfuse_host"]


def _build_callbacks() -> list:
    """Construye callbacks de Langfuse para tracking de LLM calls."""
    _set_langfuse_env()
    return [CallbackHandler()]


def _run_ocr(
    image_bytes: bytes,
    text_key: str,
    file_extension: str = ".png",
    label_for_toast: str = "",
) -> None:
    callbacks = _build_callbacks() if _keys_ready else None

    with st.spinner("Extrayendo texto con GPT-4o Vision…"):
        try:
            with tempfile.NamedTemporaryFile(
                suffix=file_extension, delete=False
            ) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name

            text = parse_contract_image(
                image_path=tmp_path,
                callbacks=callbacks,
                api_key=st.session_state["openai_api_key"],
            )
            st.session_state[text_key] = text
            st.toast(f"✅ Hoja {label_for_toast} extraída correctamente")
        except Exception as exc:
            st.error(f"Error en OCR (hoja {label_for_toast}): {exc}")
        finally:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
            # Flush de Langfuse base por las dudas
            if get_client is not None:
                get_client().flush()


def _build_document_text(prefix: str, num_sheets: int) -> str:
    """
    Concatenate non-empty sheet texts for a given slot range using a clear
    human-readable separator so agents can distinguish sheet boundaries.

    Output format:
        --- HOJA 1 ---

        <text>
    """
    parts: list[str] = []
    for idx in range(num_sheets):
        key = f"{prefix}_text_{idx}"
        text = st.session_state.get(key, "").strip()
        if text:
            parts.append(f"--- HOJA {idx + 1} ---\n\n{text}")
    return "\n\n".join(parts)


def _render_sheet_slot(label: str, slot_idx: int, prefix: str) -> None:
    """
    Render the full upload → preview → OCR → editable-textarea flow for a
    single sheet slot.
    """
    text_key = f"{prefix}_text_{slot_idx}"
    if text_key not in st.session_state:
        st.session_state[text_key] = ""

    current_text = st.session_state[text_key].strip()
    is_disabled = bool(current_text)

    # Usamos un H3 nativo para que se lea mucho más grande y claro
    st.markdown(f"### 📄 Hoja {slot_idx + 1}")

    uploaded = st.file_uploader(
        label=f"Cargar Hoja {slot_idx + 1} — {label}",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"uploader_{prefix}_{slot_idx}",
        label_visibility="collapsed",
        disabled=is_disabled,
    )

    if uploaded is not None:
        st.image(uploaded, width=150, caption=uploaded.name)

        ocr_disabled = not st.session_state["openai_api_key"]

        if st.button(
            "Extraer Texto",
            key=f"ocr_btn_{prefix}_{slot_idx}",
            disabled=ocr_disabled or is_disabled,
            use_container_width=True,
        ):
            ext = Path(uploaded.name).suffix.lower() or ".png"
            _run_ocr(
                uploaded.getvalue(),
                text_key,
                file_extension=ext,
                label_for_toast=f"{slot_idx + 1}",
            )

        if ocr_disabled:
            st.caption("⚠️ Configurá la OpenAI API Key para habilitar el OCR.")

    st.text_area(
        label="Texto extraído (editable)",
        key=text_key,
        height=220,
        placeholder=(
            'Hacé clic en "Extraer Texto" para poblar este campo '
            "automáticamente, o pegá el texto manualmente."
        ),
    )


# ---------------------------------------------------------------------------
# Main layout — 2-column upload area
# ---------------------------------------------------------------------------

st.title("⚖️ LexTrace")
st.markdown(
    "##### Ejecutá el análisis comparativo entre un contrato original y su adenda.",
    help="El sistema alinea los textos automáticamente usando LangChain y GPT-4o.",
)
st.divider()

col_original, col_addendum = st.columns(2, gap="large")

with col_original:
    st.subheader("📄 Contrato Original")
    st.divider()
    for idx in range(st.session_state["num_sheets_orig"]):
        _render_sheet_slot("Contrato Original", idx, "orig")
        if idx < st.session_state["num_sheets_orig"] - 1:
            st.divider()

    # Columnas internas para agrupar los botones
    btn_o1, btn_o2 = st.columns(2)
    with btn_o1:
        if st.session_state["num_sheets_orig"] < MAX_HOJAS:
            if st.button("Agregar Hoja", key="btn_add_orig", use_container_width=True):
                st.session_state["num_sheets_orig"] += 1
                st.rerun()
    with btn_o2:
        if st.session_state["num_sheets_orig"] > 1:
            if st.button("Quitar Hoja", key="btn_rem_orig", use_container_width=True):
                idx = st.session_state["num_sheets_orig"] - 1
                st.session_state.pop(f"orig_text_{idx}", None)
                st.session_state["num_sheets_orig"] -= 1
                st.rerun()

with col_addendum:
    st.subheader("📑 Adenda")
    st.divider()
    for idx in range(st.session_state["num_sheets_add"]):
        _render_sheet_slot("Adenda", idx, "add")
        if idx < st.session_state["num_sheets_add"] - 1:
            st.divider()

    btn_a1, btn_a2 = st.columns(2)
    with btn_a1:
        if st.session_state["num_sheets_add"] < MAX_HOJAS:
            if st.button("Agregar Hoja", key="btn_add_add", use_container_width=True):
                st.session_state["num_sheets_add"] += 1
                st.rerun()
    with btn_a2:
        if st.session_state["num_sheets_add"] > 1:
            if st.button("Quitar Hoja", key="btn_rem_add", use_container_width=True):
                idx = st.session_state["num_sheets_add"] - 1
                st.session_state.pop(f"add_text_{idx}", None)
                st.session_state["num_sheets_add"] -= 1
                st.rerun()

# ---------------------------------------------------------------------------
# Run pipeline button
# ---------------------------------------------------------------------------

st.divider()

_original_text = _build_document_text("orig", st.session_state["num_sheets_orig"])
_addendum_text = _build_document_text("add", st.session_state["num_sheets_add"])

_analysis_ready = bool(_original_text and _addendum_text and _keys_ready)

st.markdown("### 🚀 Ejecución del Análisis")

# Mensajes de validación limpios apilados antes del botón
if not _keys_ready:
    st.warning("⚠️ Completá las tres credenciales de API en el panel lateral.")
elif not _original_text:
    st.info("📄 Falta completar el texto del Contrato Original.")
elif not _addendum_text:
    st.info("📑 Falta completar el texto de la Adenda.")
else:
    st.success("✅ Todo listo para ejecutar el análisis comparativo.")

_run_analysis = st.button(
    "Ejecutar Análisis Legal",
    disabled=not _analysis_ready,
    type="primary",
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

if _run_analysis:
    _set_langfuse_env()

    with st.status("⚙️ Ejecutando pipeline de análisis…", expanded=True) as _status:
        _ctx = None  # Inicializamos por las dudas para evitar NameError en finally
        try:
            # ── Traza padre e integración explícita con Langfuse ─────────
            _trace_metadata = {
                "interface": "streamlit",
                "original_chars": str(len(_original_text)),
                "addendum_chars": str(len(_addendum_text)),
            }

            if propagate_attributes is not None:
                _ctx = propagate_attributes(
                    trace_name="lextrace-pipeline",
                    session_id="lextrace-streamlit",
                    user_id="streamlit-user",
                    metadata=_trace_metadata,
                )
                _ctx.__enter__()

            _callbacks = _build_callbacks()

            # ── Step 1: Contextualization Agent (Cartógrafo) ──────────────
            st.write("🗺️  **Agente 1 — Cartógrafo**: mapeando secciones…")
            cartographer = ContextualizationAgent(
                openai_api_key=st.session_state["openai_api_key"],
                callbacks=_callbacks,
            )
            section_mappings = cartographer.run(
                original_text=_original_text,
                amended_text=_addendum_text,
            )
            st.write(f"   ↳ {len(section_mappings)} sección(es) mapeada(s).")

            # ── Step 2: Extraction Agent (Detective) ──────────────────────
            st.write("🔎  **Agente 2 — Detective**: extrayendo cambios…")
            detective = ExtractionAgent(
                openai_api_key=st.session_state["openai_api_key"],
                callbacks=_callbacks,
            )
            result: ContractChangeOutput = detective.run(
                mappings=section_mappings,
            )

            st.session_state["analysis_result"] = result
            _status.update(label="✅ Análisis completado.", state="complete")

            # ── Flush Langfuse para asegurar envío de trazas ─────────────
            if get_client is not None:
                get_client().flush()
            if hasattr(_callbacks[0], "flush"):
                _callbacks[0].flush()

        except ValidationError as ve:
            _status.update(label="❌ Error de Validación de Datos", state="error")
            st.error(
                f"El modelo generó un formato incompatible con el schema esperado. "
                f"Detalles: {ve}"
            )
        except RateLimitError:
            _status.update(label="❌ Límite de API alcanzado", state="error")
            st.error(
                "Se alcanzó el límite de la API de OpenAI. "
                "Esperá un momento e intentá nuevamente."
            )
        except APITimeoutError:
            _status.update(label="❌ Timeout de API", state="error")
            st.error(
                "La conexión con OpenAI expiró. "
                "Verificá tu conexión a internet y reintentá la operación."
            )
        except Exception as _exc:
            _status.update(label="❌ Error durante el análisis.", state="error")
            st.exception(_exc)
        finally:
            if _ctx is not None:
                _ctx.__exit__(None, None, None)

# ---------------------------------------------------------------------------
# Results — three-level hierarchy aligned with ContractChangeOutput
# ---------------------------------------------------------------------------

if st.session_state["analysis_result"] is not None:
    result: ContractChangeOutput = st.session_state["analysis_result"]

    st.divider()
    st.header("📊 Resultados del Análisis")

    # ── Level 1: Summary ──────────────────────────────────────────────────
    st.subheader("📝 Resumen del Cambio")
    st.info(result.summary_of_the_change)

    st.divider()

    # ── Level 2: Topics touched — inline pill badges ───────────────────────
    st.subheader("🏷️ Temas Involucrados")

    if result.topics_touched:
        _badge_style = (
            "display:inline-block;"
            "background:#1B6E78;"
            "color:#ffffff;"
            "border-radius:999px;"
            "padding:0.25rem 0.85rem;"
            "margin:0.25rem 0.3rem 0.25rem 0;"
            "font-size:0.85rem;"
            "font-weight:500;"
            "letter-spacing:0.02em;"
        )
        _badges_html = " ".join(
            f'<span style="{_badge_style}">{topic}</span>'
            for topic in result.topics_touched
        )
        st.markdown(_badges_html, unsafe_allow_html=True)
    else:
        st.caption("Sin temas detectados.")

    st.divider()

    # ── Level 3: Sections changed — audit table ───────────────────────────
    st.subheader("📋 Secciones Modificadas")

    if result.sections_changed:
        st.table(
            [
                {"#": idx + 1, "Sección": section}
                for idx, section in enumerate(result.sections_changed)
            ]
        )
    else:
        st.caption("No se identificaron secciones con cambios.")

    st.divider()

    # ── Raw JSON — expandable ─────────────────────────────────────────────
    with st.expander("🗂️ Ver JSON completo del resultado", expanded=False):
        st.json(result.model_dump())
