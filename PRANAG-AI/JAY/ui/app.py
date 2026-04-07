# ui/app.py
# Streamlit testing interface for the full pipeline.
# Run with: streamlit run ui/app.py

import sys, os, json, time, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

st.set_page_config(page_title="AgriAI Pipeline", page_icon="🌾", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌾 AgriAI")
    st.caption("Crop Specification Generator")
    st.divider()
    st.markdown("""
**Orchestrator**
- Prompt parser (LLM)
- LangGraph workflow
- Research integration
- Spec builder + validator

**Search Engine**
- Trait embeddings (384-dim)
- ChromaDB vector store
- Similarity search (<50ms)
- Data cleaning pipeline
    """)
    st.divider()
    top_k          = st.slider("Max traits",   3, 20, 10)
    max_papers     = st.slider("Max papers",   1, 10,  5)
    min_confidence = st.slider("Min confidence", 0.0, 1.0, 0.5)
    st.divider()
    st.caption("v1.0.0")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("PRANAG-AI ")
st.markdown("> Convert agricultural prompts into validated scientific JSON specifications")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🌾 Wheat · Jodhpur 48°C"):
        st.session_state["prompt"] = "wheat for Jodhpur at 48°C"
with col2:
    if st.button("🌾 Rice · Punjab drought"):
        st.session_state["prompt"] = "rice cultivation in Punjab under drought stress"
with col3:
    if st.button("🌽 Maize · Rajasthan heat"):
        st.session_state["prompt"] = "maize hybrid for Rajasthan high temperature tolerance"

user_prompt = st.text_area(
    "Agricultural prompt:",
    value=st.session_state.get("prompt", ""),
    placeholder="e.g., wheat for Jodhpur at 48°C",
    height=80
)
run = st.button("🚀 Generate spec.json", type="primary", use_container_width=True)

# ── Pipeline Run ──────────────────────────────────────────────────────────────
if run and user_prompt.strip():
    progress = st.progress(0)
    status   = st.empty()

    tab_spec, tab_traits, tab_research, tab_debug = st.tabs(
        ["📄 spec.json", "🧬 Traits", "📚 Research", "🔧 Debug"]
    )

    try:
        status.text("🧠 Step 1/5: Parsing prompt…")
        progress.progress(0.2)
        from JAY.orchestrator.prompt_parser import parse_prompt
        parsed = parse_prompt(user_prompt)

        status.text("🔍 Step 2/5: Searching vector database…")
        progress.progress(0.4)
        from JAY.search_engine.similarity_search import search_traits
        traits = search_traits(
            f"{parsed.crop_type} {parsed.location_raw} {' '.join(parsed.stress_hints)}",
            top_k=top_k
        )

        status.text("📚 Step 3/5: Fetching research papers…")
        progress.progress(0.6)
        from JAY.orchestrator.research_fetcher import fetch_research
        research = fetch_research(
            f"{parsed.crop_type} {' '.join(parsed.stress_hints) or 'general'} stress",
            max_results=max_papers
        )

        status.text("⚙️ Step 4/5: Building spec…")
        progress.progress(0.8)
        from JAY.orchestrator.spec_builder import build_spec
        spec_obj = build_spec(parsed, traits, research, user_prompt, str(uuid.uuid4()))

        status.text("✅ Step 5/5: Validating…")
        progress.progress(0.95)
        from JAY.orchestrator.output_validator import validate_spec, serialize_to_json
        validated = validate_spec(spec_obj.model_dump())
        spec_json = serialize_to_json(validated)

        progress.progress(1.0)
        status.text("✅ Done!")

        # Tab: spec.json
        with tab_spec:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Crop",     validated.get("crop_type", "—").title())
            c2.metric("Location", validated.get("location", {}).get("city", "—"))
            t = validated.get("conditions", {}).get("temperature_max")
            c3.metric("Max Temp", f"{t}°C" if t else "—")
            conf  = validated.get("confidence_score", 0)
            emoji = "🟢" if conf > 0.7 else "🟡" if conf > 0.5 else "🔴"
            c4.metric("Confidence", f"{emoji} {conf:.0%}")

            for w in validated.get("warnings", []):
                st.warning(w)

            st.subheader("spec.json")
            st.code(spec_json, language="json")
            st.download_button("⬇️ Download spec.json", spec_json,
                               file_name=f"spec_{validated.get('pipeline_id','')[:8]}.json",
                               mime="application/json")

        # Tab: Traits
        with tab_traits:
            st.subheader(f"🧬 Traits ({len(traits)})")
            if traits:
                for t in traits:
                    with st.expander(f"**{t.trait_name}** = {t.value} {t.unit}  [sim: {t.similarity_score:.3f}]"):
                        col_a, col_b = st.columns(2)
                        col_a.write(f"**ID:** `{t.trait_id}`  \n**Value:** {t.value} {t.unit}  \n**Confidence:** {t.confidence:.2%}")
                        col_b.write(f"**Similarity:** {t.similarity_score:.4f}  \n**Source:** {t.source_dataset}")
                        st.progress(t.similarity_score)
            else:
                st.info("No traits found. Populate the vector store first.")
                st.code("python search_engine/vector_store.py")

        # Tab: Research
        with tab_research:
            st.subheader(f"📚 Research ({len(research)})")
            for p in research:
                with st.expander(f"📄 {p.title} ({p.year}) — {p.relevance:.0%} relevant"):
                    st.markdown(f"**Finding:** {p.key_finding}")
                    if p.journal: st.caption(f"Journal: {p.journal}")
                    if p.doi:     st.caption(f"DOI: {p.doi}")
                    st.progress(p.relevance)

        # Tab: Debug
        with tab_debug:
            st.subheader("Parsed Prompt")
            st.json(parsed.model_dump())
            st.subheader("Pipeline Metadata")
            st.json({
                "pipeline_id":       validated.get("pipeline_id"),
                "generated_at":      validated.get("generated_at"),
                "traits_found":      len(traits),
                "papers_found":      len(research),
                "validation_passed": validated.get("validation_passed"),
            })

    except Exception as e:
        progress.empty()
        st.error(f"Pipeline error: {e}")
        with st.expander("Traceback"):
            import traceback
            st.code(traceback.format_exc())

elif run:
    st.warning("Please enter a prompt.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
col_l, col_r = st.columns(2)
with col_l:
    if st.button("📊 DB Status"):
        try:
            from JAY.search_engine.vector_store import get_stats
            st.json(get_stats())
        except Exception as e:
            st.error(str(e))
with col_r:
    if st.button("🌱 Load Sample Data (1,000 traits)"):
        with st.spinner("Populating…"):
            try:
                from JAY.search_engine.vector_store import populate_sample
                populate_sample(1000)
                st.success("✅ Sample data loaded!")
            except Exception as e:
                st.error(str(e))
