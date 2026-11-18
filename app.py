"""TriageFlow — Streamlit interface.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os

import streamlit as st

from agent import JevClient, JevIntakeAgent, OfflineIntakeAgent, OpenAIIntakeAgent
from triage_engine import FIELDS, Outcome, evaluate, red_flag_fields_for
from workflow import TriageSession, handoff_summary

st.set_page_config(page_title="TriageFlow", page_icon="🩺", layout="wide")

GREETING = ("Hi — I'm the TriageFlow intake assistant. Describe the **fictional adult patient's** main "
            "symptom (headache, abdominal pain, or breathing/respiratory symptoms) and I'll ask follow-up "
            "questions one at a time.")

OUTCOME_STYLE = {
    Outcome.ER.value: ("🚨", "error"),
    Outcome.URGENT_CARE.value: ("⚠️", "warning"),
    Outcome.PRIMARY_CARE.value: ("🩺", "success"),
    Outcome.NEEDS_REVIEW.value: ("🔍", "info"),
}

SAMPLE_CASES = {
    "Headache — thunderclap": "A 45 year old with a headache that came on suddenly while lifting weights, worst headache of their life.",
    "Headache — mild, 2 days": "I've had a headache for two days.",
    "Abdominal — RLQ pain": "28 years old, stomach pain in the lower right side since yesterday, 6/10.",
    "Respiratory — cough": "34 yo with a cough for a week, no fever.",
}


def _secret(name: str) -> str | None:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets.toml present
        pass
    return os.environ.get(name)


MODE_JEV = "Jev decision model"
MODE_OPENAI = "OpenAI only"
MODE_OFFLINE = "Offline (keyword)"


def make_agent(mode: str):
    openai_key = _secret("OPENAI_API_KEY")
    language = (OpenAIIntakeAgent(api_key=openai_key, model=_secret("OPENAI_MODEL") or "gpt-4o-mini")
                if openai_key else OfflineIntakeAgent())
    if mode == MODE_JEV:
        return JevIntakeAgent(JevClient(_secret("OPENROUTER_API_KEY")), language)
    if mode == MODE_OPENAI:
        return language
    return OfflineIntakeAgent()


def new_case(mode: str) -> None:
    st.session_state.session = TriageSession(make_agent(mode))
    st.session_state.mode = mode


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
has_key = bool(_secret("OPENAI_API_KEY"))
has_jev = bool(_secret("OPENROUTER_API_KEY"))
with st.sidebar:
    st.title("🩺 TriageFlow")
    st.caption("AI-powered intake & rule-based triage prototype")
    st.warning("**Educational prototype.** Fictional adult cases only. Outputs demonstrate example "
               "software rules and are **not medical advice**. In a real emergency, call 911.")

    modes = ([MODE_JEV] if has_jev else []) + ([MODE_OPENAI] if has_key else []) + [MODE_OFFLINE]
    mode = st.radio("Fact extraction", modes,
                    help="Jev: yes/no facts come from decision-model probabilities and fixed thresholds. "
                         "OpenAI only: the LLM returns yes/no facts directly. Offline: keyword rules.")
    if not has_jev:
        st.caption("Add `OPENROUTER_API_KEY` to `.streamlit/secrets.toml` to enable the Jev decision model.")
    if not has_key:
        st.caption("No `OPENAI_API_KEY` found — free-text extraction uses the offline keyword agent.")

    if "session" not in st.session_state or st.session_state.get("mode") != mode:
        new_case(mode)

    if st.button("➕ Start new case", width="stretch"):
        new_case(mode)
        st.rerun()

    st.subheader("Sample scenarios")
    sample = None
    for label, text in SAMPLE_CASES.items():
        if st.button(label, width="stretch"):
            new_case(mode)
            sample = text

    st.divider()
    st.caption(f"Agent: **{st.session_state.session.agent.name}**")
    cost = st.session_state.session.state.get("decision_cost", 0.0)
    if cost:
        st.caption(f"Jev cost this case: ${cost:.6f}")

session: TriageSession = st.session_state.session
state = session.state

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
chat_col, case_col = st.columns([3, 2], gap="large")

with chat_col:
    st.header("Intake conversation")
    with st.chat_message("assistant"):
        st.markdown(GREETING)
    for m in state.get("messages", []):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    complete = state.get("status") == "complete"
    prompt = st.chat_input("Case complete — start a new case" if complete else "Describe the symptoms…",
                           disabled=complete)
    prompt = sample or prompt
    if prompt:
        with st.spinner("Thinking…"):
            session.send(prompt)
        st.rerun()

with case_col:
    facts = state.get("facts", {})
    contradictions = state.get("contradictions", [])
    result = state.get("result")

    if result:
        icon, kind = OUTCOME_STYLE[result["outcome"]]
        getattr(st, kind)(f"### {icon} {result['outcome']}")
        st.caption("Determined by the deterministic rule engine — the AI agent cannot set or change this.")
        for r in result["reasons"]:
            st.markdown(f"- {r}")
        if result["triggered_rules"]:
            st.markdown("**Rules triggered**")
            for r in result["triggered_rules"]:
                st.markdown(f"- `{r['id']}` {r['description']}")
        if result["missing_fields"]:
            st.markdown("**Information not established**")
            for f in result["missing_fields"]:
                st.markdown(f"- {FIELDS[f].label}")
        st.download_button("⬇️ Download handoff summary", handoff_summary(state),
                           file_name="triageflow_handoff.txt", width="stretch")
    else:
        st.subheader("Case in progress")
        live = evaluate(facts, contradictions)
        if facts.get("category"):
            screened = [f for f in red_flag_fields_for(facts["category"]) if f in facts]
            total = len(red_flag_fields_for(facts["category"]))
            st.progress(len(screened) / total if total else 0.0,
                        text=f"Emergency warning signs screened: {len(screened)}/{total}")
        if live.missing_fields:
            st.caption("Still needed: " + ", ".join(FIELDS[f].label for f in live.missing_fields))

    st.subheader("Structured facts")
    scores = state.get("fact_scores", {})
    if facts:
        rows = []
        for key, spec in FIELDS.items():
            if key in facts:
                v = facts[key]
                row = {"Finding": spec.label,
                       "Value": "✅ yes" if v is True else "❌ no" if v is False else str(v)}
                if scores:
                    row["Jev score"] = f"{scores[key]:.2f}" if key in scores else "—"
                rows.append(row)
        st.dataframe(rows, hide_index=True, width="stretch")
        if scores:
            st.caption("Jev score = p(yes) for yes/no findings (p(category) for the main complaint). "
                       "Thresholds: asked field yes ≥ 0.70, no ≤ 0.15; unasked finding yes ≥ 0.85.")
    else:
        st.caption("Nothing extracted yet.")

    if state.get("uncertain"):
        st.markdown("**Unclear answers (not accepted — re-asked)**")
        for u in state["uncertain"]:
            st.markdown(f"- {u}")

    if contradictions:
        st.markdown("**Contradictions**")
        for c in contradictions:
            st.markdown(f"- {c}")

    with st.expander("Debug: workflow state"):
        st.code(json.dumps({k: v for k, v in state.items() if k != "messages"}, indent=2, default=str),
                language="json")
