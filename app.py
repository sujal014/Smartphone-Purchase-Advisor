import os
import streamlit as st
import pandas as pd

from fuzzy_engine import build_fuzzy_system, score_phone
from phones_data import PHONES
from llm_agent import extract_requirements, explain_recommendation

st.set_page_config(page_title="Smartphone Purchase Advisor", page_icon="📱", layout="centered")

# ---- API key wiring: works both locally (env var) and on Streamlit
# Community Cloud (st.secrets) ----
if "GROQ_API_KEY" not in os.environ:
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

st.title("📱 Smartphone Purchase Advisor")
st.caption(
    "Tell me what you're looking for in plain English. An LLM (LangChain) will "
    "read your request, and a fuzzy logic engine will score and rank real phones "
    "against your priorities."
)

with st.expander("ℹ️ How this works"):
    st.markdown(
        """
1. **LangChain / LLM** reads your free-text request and extracts a budget and
   importance weights (1-10) for camera, performance, and battery.
2. A **fuzzy inference system** (membership functions → rule evaluation →
   centroid defuzzification) scores every phone in the catalog for how well
   it suits *your* weighted priorities and budget — not just a hard filter.
3. The **LLM** then explains the top pick and an alternative in plain language.
        """
    )

if not os.environ.get("GROQ_API_KEY"):
    st.warning(
        "No GROQ_API_KEY found. Set it as an environment variable locally, "
        "or add it under **Settings → Secrets** if this app is deployed on "
        "Streamlit Community Cloud.",
        icon="⚠️",
    )

query = st.text_area(
    "What are you looking for?",
    placeholder=(
        "e.g. I need a phone under 30000 rupees. I take a lot of photos for "
        "Instagram and my current phone's battery dies by evening, but I "
        "barely play games."
    ),
    height=100,
)

if st.button("🔍 Get Recommendation", type="primary", use_container_width=True):
    if not query.strip():
        st.error("Please describe what you're looking for first.")
    elif not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not configured — see the warning above.")
    else:
        with st.spinner("Reading your request..."):
            try:
                requirements = extract_requirements(query)
            except Exception as e:
                st.error(f"Couldn't parse your request: {e}")
                st.stop()

        st.subheader("What I understood")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Budget", f"₹{requirements.budget:,}")
        c2.metric("Camera priority", f"{requirements.camera_importance}/10")
        c3.metric("Performance priority", f"{requirements.performance_importance}/10")
        c4.metric("Battery priority", f"{requirements.battery_importance}/10")
        st.caption(requirements.summary)

        with st.spinner("Running the fuzzy inference engine over the catalog..."):
            system = build_fuzzy_system()
            scored = []
            for p in PHONES:
                r = score_phone(
                    system,
                    price=p["price"],
                    budget=requirements.budget,
                    camera_spec=p["camera"],
                    performance_spec=p["performance"],
                    battery_spec=p["battery"],
                    camera_importance=requirements.camera_importance,
                    performance_importance=requirements.performance_importance,
                    battery_importance=requirements.battery_importance,
                )
                scored.append({"name": p["name"], "price": p["price"], "score": r["suitability"], **r})

            scored.sort(key=lambda x: -x["score"])

        st.subheader("Ranked results")
        df = pd.DataFrame(scored)[["name", "price", "score"]].rename(
            columns={"name": "Phone", "price": "Price (₹)", "score": "Suitability score"}
        )
        st.dataframe(df, hide_index=True, use_container_width=True)

        with st.spinner("Writing your recommendation..."):
            try:
                explanation = explain_recommendation(query, requirements, scored)
            except Exception as e:
                st.error(f"Couldn't generate an explanation: {e}")
                st.stop()

        st.subheader("💬 My recommendation")
        st.markdown(explanation)

st.divider()
st.caption("Built with LangChain (LLM reasoning) + scikit-fuzzy (Mamdani fuzzy inference) + Streamlit.")
