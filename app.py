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
    # Langfuse v3+
    from langfuse import get_client, propagate_attributes
    from langfuse.langchain import CallbackHandler
except ImportError:
    # Langfuse v2 fallback
    from langfuse.callback import CallbackHandler

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
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

# sheet_text_0..2  → Contrato Original (hojas 1-3)
# sheet_text_3..5  → Adenda            (hojas 1-3)
_TEXT_KEYS = [f"sheet_text_{i}" for i in range(6)]

_DEFAULTS: dict = {
    "openai_api_key": "",
    "langfuse_public_key": "",
    "langfuse_secret_key": "",
    "langfuse_host": "https://us.cloud.langfuse.com",
    "analysis_result": None,
    **{k: "" for k in _TEXT_KEYS},
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
    st.session_state["langfuse_public_key"] = st.text_input(
        "Langfuse Public Key",
        value=st.session_state["langfuse_public_key"],
        type="password",
        placeholder="pk-lf-…",
    )
    st.session_state["langfuse_secret_key"] = st.text_input(
        "Langfuse Secret Key",
        value=st.session_state["langfuse_secret_key"],
        type="password",
        placeholder="sk-lf-…",
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


def _run_ocr(image_bytes: bytes, slot_index: int, file_extension: str = ".png") -> None:
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
            st.session_state[_TEXT_KEYS[slot_index]] = text
            st.toast(f"✅ Hoja {slot_index % 3 + 1} extraída correctamente")
        except Exception as exc:
            st.error(f"Error en OCR (slot {slot_index}): {exc}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def _build_document_text(slot_range: range) -> str:
    """
    Concatenate non-empty sheet texts for a given slot range using a clear
    human-readable separator so agents can distinguish sheet boundaries.

    Output format:
        --- HOJA 1 ---

        <text>

        --- HOJA 2 ---

        <text>
    """
    parts: list[str] = []
    for relative_idx, slot_idx in enumerate(slot_range, start=1):
        text = st.session_state[_TEXT_KEYS[slot_idx]].strip()
        if text:
            parts.append(f"--- HOJA {relative_idx} ---\n\n{text}")
    return "\n\n".join(parts)


def _render_sheet_slot(label: str, sheet_num: int, slot_index: int) -> None:
    """
    Render the full upload → preview → OCR → editable-textarea flow for a
    single sheet slot.  All state is scoped to *slot_index*.
    """
    st.markdown(f"**Hoja {sheet_num}**")

    uploaded = st.file_uploader(
        label=f"Cargar Hoja {sheet_num} — {label}",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"uploader_{slot_index}",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        st.image(uploaded, use_container_width=True, caption=uploaded.name)

        ocr_disabled = not st.session_state["openai_api_key"]

        if st.button(
            "🔍 Extraer Texto de esta Hoja",
            key=f"ocr_btn_{slot_index}",
            disabled=ocr_disabled,
            use_container_width=True,
        ):
            ext = Path(uploaded.name).suffix.lower() or ".png"
            _run_ocr(uploaded.getvalue(), slot_index, file_extension=ext)

        if ocr_disabled:
            st.caption("⚠️ Configurá la OpenAI API Key para habilitar el OCR.")

    # Streamlit owns this widget's value via its key.
    # Writing to st.session_state[key] from _run_ocr updates it on the next rerun
    # without conflict — never pass both `key=` and `value=` to the same widget.
    st.text_area(
        label="Texto extraído (editable)",
        key=_TEXT_KEYS[slot_index],
        height=220,
        placeholder=(
            'Hacé clic en "Extraer Texto de esta Hoja" para poblar este campo '
            "automáticamente, o pegá el texto manualmente."
        ),
    )


# ---------------------------------------------------------------------------
# Main layout — 2-column upload area
# ---------------------------------------------------------------------------

st.title("⚖️ LexTrace")
st.caption(
    "Cargá las hojas escaneadas, validá el OCR y ejecutá el análisis "
    "comparativo entre el contrato original y su adenda."
)
st.divider()

col_original, col_addendum = st.columns(2, gap="large")

with col_original:
    st.subheader("📄 Contrato Original")
    st.divider()
    for _sheet in range(1, 4):
        _render_sheet_slot("Contrato Original", _sheet, slot_index=_sheet - 1)
        if _sheet < 3:
            st.divider()

with col_addendum:
    st.subheader("📑 Adenda")
    st.divider()
    for _sheet in range(1, 4):
        _render_sheet_slot("Adenda", _sheet, slot_index=_sheet + 2)
        if _sheet < 3:
            st.divider()

# ---------------------------------------------------------------------------
# Run pipeline button
# ---------------------------------------------------------------------------

st.divider()

_original_text = _build_document_text(range(0, 3))
_addendum_text = _build_document_text(range(3, 6))

_analysis_ready = bool(_original_text and _addendum_text and _keys_ready)

btn_col, hint_col = st.columns([1, 3], gap="medium")

with btn_col:
    _run_analysis = st.button(
        "🚀 Ejecutar Análisis Legal",
        disabled=not _analysis_ready,
        type="primary",
        use_container_width=True,
    )

with hint_col:
    st.write("")  # vertical alignment nudge
    if not _original_text:
        st.info("Falta texto del Contrato Original.")
    elif not _addendum_text:
        st.info("Falta texto de la Adenda.")
    elif not _keys_ready:
        st.info("Completá las tres API keys en el sidebar.")
    else:
        st.success("Todo listo para ejecutar.")

# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

if _run_analysis:
    _set_langfuse_env()

    with st.status("⚙️ Ejecutando pipeline de análisis…", expanded=True) as _status:
        try:
            # ── Traza padre Langfuse con jerarquía de spans ──────────────
            _trace_metadata = {
                "langfuse_session_id": "lextrace-streamlit",
                "langfuse_tags": ["streamlit", "contract-analysis"],
            }

            if propagate_attributes is not None:
                _ctx = propagate_attributes(
                    trace_name="lextrace-pipeline",
                    session_id="lextrace-streamlit",
                    user_id="streamlit-user",
                    metadata={
                        "interface": "streamlit",
                        "original_chars": len(_original_text),
                        "addendum_chars": len(_addendum_text),
                    },
                )
                _ctx.__enter__()
            else:
                _ctx = None

            _callbacks = [CallbackHandler()]

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
