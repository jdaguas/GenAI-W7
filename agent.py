"""Intake agent for TriageFlow.

The agent is responsible for LANGUAGE ONLY:
  * extracting structured facts from free text, and
  * phrasing one follow-up question at a time.

It never decides a care setting. Anything it returns is validated against the
field catalog in ``triage_engine`` and unknown keys (e.g. an attempted
"outcome") are discarded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from triage_engine import CATEGORIES, FIELDS, OUT_OF_SCOPE_CATEGORY, sanitize_facts


@dataclass
class Extraction:
    facts: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    # Decision-model output (empty for agents that don't use one):
    scores: dict[str, float] = field(default_factory=dict)  # field -> probability behind an accepted fact
    uncertain: list[str] = field(default_factory=list)      # answers too unclear to accept
    cost: float = 0.0                                        # USD spent on decision calls this turn


class IntakeAgent(Protocol):
    name: str

    def extract(self, user_message: str, last_question: Optional[str],
                pending_field: Optional[str], facts: dict[str, Any]) -> Extraction: ...

    def phrase_question(self, field_key: str, facts: dict[str, Any],
                        history: list[dict[str, str]]) -> str: ...


# ---------------------------------------------------------------------------
# OpenAI-backed agent
# ---------------------------------------------------------------------------
def _field_catalog_text() -> str:
    lines = []
    for spec in FIELDS.values():
        lines.append(f'- "{spec.key}" ({spec.type}): {spec.description}')
    return "\n".join(lines)


EXTRACT_SYSTEM_PROMPT = f"""You are the intake assistant in TriageFlow, an EDUCATIONAL prototype that \
processes FICTIONAL adult patient cases. Your only job is to extract structured facts from the \
user's latest message. You do NOT decide urgency or care setting.

Field catalog (key, type, meaning):
{_field_catalog_text()}

Rules:
- "category" must be one of: {", ".join(CATEGORIES)}, {OUT_OF_SCOPE_CATEGORY}. Map synonyms \
(e.g. "stomach ache" -> abdominal, "cough" or "cold" -> respiratory, "migraine" -> headache).
- "last_question_asked" is the exact question the user is answering; it targets "pending_field".
- A clear yes/no answer to that question sets "pending_field" (e.g. "no", "nope", "none of those", \
"not really", "nothing like that" -> false). A question that lists several symptoms is answered \
false only if the user denies all of them, and true if they confirm any of them.
- Hedged or unknown answers ("I think so", "maybe", "not sure", "haven't checked", "I feel a bit \
warm") mean the field is UNKNOWN: omit it. Never turn uncertainty into false.
- Facts that rule a field out count as false: e.g. a male patient or post-menopausal patient \
-> "pregnancy_possible": false.
- Also extract any other field the message clearly states, even if it was not asked about.
- Report only what the latest message says. Do not repeat "facts_known_so_far".
- "severity" is an integer 0-10 (convert words: "mild" is not a number, omit it); "age" is an \
integer in years; "duration" is short text (e.g. "since Tuesday", "2 days").

Respond with ONLY a JSON object of the form:
{{"facts": {{"<field key>": <value>, ...}}, "notes": "<short note on anything ambiguous>"}}"""


ACK_SYSTEM_PROMPT = """You are the intake assistant in TriageFlow, an EDUCATIONAL prototype for \
FICTIONAL adult patient cases. Write a brief, warm acknowledgement of the user's latest message, \
which will be shown right before the next (pre-written) question.

Rules:
- At most 12 words. Do NOT ask a question. Do NOT use a question mark.
- Never give a diagnosis, reassurance about safety, or advice about where to seek care.
- Output only the acknowledgement text (e.g. "Thanks, that's helpful." or "Understood.")."""


class OpenAIIntakeAgent:
    name = "OpenAI"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", client: Any = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self.client = client
        self.model = model
        self._fallback = OfflineIntakeAgent()

    def extract(self, user_message, last_question, pending_field, facts) -> Extraction:
        context = {
            "pending_field": pending_field,
            "last_question_asked": last_question,
            "facts_known_so_far": facts,
            "latest_user_message": user_message,
        }
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                ],
            )
            data = json.loads(resp.choices[0].message.content or "{}")
        except Exception as exc:  # network / parsing problems -> offline fallback
            ext = self._fallback.extract(user_message, last_question, pending_field, facts)
            ext.notes = f"LLM extraction failed ({type(exc).__name__}); used keyword fallback."
            return ext
        raw = data.get("facts") if isinstance(data, dict) else None
        notes = data.get("notes", "") if isinstance(data, dict) else ""
        return Extraction(facts=sanitize_facts(raw if isinstance(raw, dict) else {}), notes=str(notes or ""))

    def phrase_question(self, field_key, facts, history) -> str:
        """Return a short LLM acknowledgement + the reviewed question, verbatim.

        The question text itself is never generated by the LLM, so the patient is
        always asked exactly what the rule engine needs (and what the extractor
        will map the answer onto).
        """
        question = FIELDS[field_key].question
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ACK_SYSTEM_PROMPT},
                    {"role": "user", "content": last_user},
                ],
            )
            ack = (resp.choices[0].message.content or "").strip().strip('"')
        except Exception:
            ack = ""
        if not ack or "?" in ack or len(ack.split()) > 15:
            return question
        return f"{ack} {question}"


# ---------------------------------------------------------------------------
# Offline keyword agent (no API key needed; used for demos and tests)
# ---------------------------------------------------------------------------
_YES = re.compile(r"^\s*(yes|yeah|yep|yup|correct|i do|it does|it is|they are|definitely|sure)\b", re.I)
_NO = re.compile(r"^\s*(no|nope|nah|none|not really|not at all|never|i don'?t|it doesn'?t|negative)\b", re.I)

_CATEGORY_KEYWORDS = {
    "headache": r"headache|migraine|head (hurts|pain|ache)",
    "abdominal": r"stomach|abdom|belly|tummy|cramp",
    "respiratory": r"cough|breath|wheez|chest cold|congest|sore throat|short of breath",
}

# Positive-mention patterns for boolean fields. Negation within the same
# clause ("no fever") flips the value to False.
_BOOL_KEYWORDS = {
    "confusion": r"confus|disorient|drowsy|hard to stay awake",
    "chest_pain": r"chest (pain|pressure|tight)",
    "fainting": r"faint|passed out|pass out|black(ed)? out",
    "sudden_worst_headache": r"worst headache|thunderclap|came on suddenly|sudden(ly)? (onset|started)",
    "neuro_deficit": r"weakness|numb|slurred|trouble speaking|facial droop|vision loss|can'?t see",
    "fever_stiff_neck": r"stiff neck",
    "head_injury": r"hit (my|his|her|their) head|head injury|blow to the head|fell and hit",
    "vomiting_blood": r"vomit(ing|ed)? blood|throwing up blood|coffee.?ground",
    "bloody_black_stool": r"black stool|bloody stool|blood in (my|the)? ?stool|tarry",
    "rigid_abdomen": r"rigid|board.?like|hard belly",
    "pregnancy_possible": r"pregnan",
    "right_lower_pain": r"lower right|right lower",
    "severe_breathing_difficulty": r"can'?t breathe|struggling to breathe|gasping",
    "blue_lips": r"blue lips|lips (are|look) blue|turning blue",
    "coughing_blood": r"cough(ing)? (up )?blood",
    "exertional_dyspnea": r"short of breath (when|walking|with)|winded",
    "wheezing": r"wheez",
    "fever": r"fever|temperature of 1\d\d",
    "persistent_vomiting": r"keep vomiting|keeps vomiting|can'?t keep (anything|fluids|water) down|vomiting a lot",
}

_NEGATION = re.compile(r"\b(no|not|without|denies|never|n't)\b[^.;,]*$", re.I)


class OfflineIntakeAgent:
    """Deterministic keyword-based stand-in for the LLM agent."""

    name = "Offline demo (keyword)"

    def extract(self, user_message, last_question, pending_field, facts) -> Extraction:
        text = user_message.strip()
        low = text.lower()
        out: dict[str, Any] = {}

        for cat, pat in _CATEGORY_KEYWORDS.items():
            if "category" not in facts and re.search(pat, low):
                out["category"] = cat
                break

        for key, pat in _BOOL_KEYWORDS.items():
            m = re.search(pat, low)
            if m:
                prefix = low[: m.start()]
                clause = re.split(r"[.;,!?]| but ", prefix)[-1]
                out[key] = not bool(_NEGATION.search(clause))

        m = re.search(r"\b(\d{1,3})\s*(years? old|y/?o|yo)\b|\bage(d)?\s*(\d{1,3})\b", low)
        if m:
            out["age"] = int(m.group(1) or m.group(4))
        m = re.search(r"\b(\d{1,2})\s*(/|out of)\s*10\b", low)
        if m:
            out["severity"] = int(m.group(1))
        m = re.search(r"\b(\d+|a|an|one|two|three|few|couple of)\s+(minute|hour|day|week|month)s?\b", low)
        if m:
            out["duration"] = m.group(0)
        elif re.search(r"\b(since )?(yesterday|this morning|last night|today)\b", low):
            out["duration"] = re.search(r"\b(since )?(yesterday|this morning|last night|today)\b", low).group(0)

        # Direct answers to the pending question.
        if pending_field and pending_field not in out:
            spec = FIELDS[pending_field]
            if spec.type == "bool":
                if _YES.search(text):
                    out[pending_field] = True
                elif _NO.search(text):
                    out[pending_field] = False
            elif spec.type == "int":
                m = re.search(r"\b(\d{1,3})\b", low)
                if m:
                    out[pending_field] = int(m.group(1))
            elif spec.type == "str" and text:
                out[pending_field] = text[:80]
            elif spec.type == "category":
                out[pending_field] = OUT_OF_SCOPE_CATEGORY

        return Extraction(facts=sanitize_facts(out), notes="")

    def phrase_question(self, field_key, facts, history) -> str:
        return FIELDS[field_key].question


# ---------------------------------------------------------------------------
# Jev decision-model agent (TypeSafe Jev via OpenRouter's Decisions API)
# ---------------------------------------------------------------------------
# Jev does not generate text: for each question it returns a probability.
# These thresholds turn that probability into a fact. They live in plain code
# so they can be reviewed and tested like the triage rules.
JEV_MODEL = "typesafe/jev-1.14"
JEV_URL = "https://openrouter.ai/api/alpha/decisions"

YES_THRESHOLD = 0.70             # answer to the question just asked: p(yes) >= this -> True
NO_THRESHOLD = 0.15              # answer to the question just asked: p(yes) <= this -> False
UNPROMPTED_YES_THRESHOLD = 0.85  # finding volunteered without being asked: can only become True
CATEGORY_MIN_PROB = 0.60         # complaint category accepted only above this probability
CLEAR_ANSWER_THRESHOLD = 0.50    # a "no" is accepted only if Jev rates the answer clear and complete

# p(yes) alone cannot tell "no" from "unsure": both are "not a confirmation" and
# score low (measured: "Maybe? Hard to tell" -> 0.12). So the asked field gets a
# second question, and only a clear, complete answer can rule a warning sign out.
_CLARITY_KEY = "_answer_is_clear"
_CLARITY_QUESTION = {
    "type": "noul",
    "instructions": "Is the patient's answer a clear, definite response that covers everything the question asked about?",
    "criteria": {
        "true": "A direct yes, or a definite no that covers every symptom listed in the question",
        "false": "Hedged, unsure, guessing, not yet checked, addresses only some of the listed symptoms, "
                 "or does not answer the question",
    },
}

_CATEGORY_CRITERIA = {
    "headache": "Head pain, headache, or migraine",
    "abdominal": "Stomach, belly, or abdominal pain or discomfort",
    "respiratory": "Cough, breathing problems, wheezing, or chest-cold symptoms",
    OUT_OF_SCOPE_CATEGORY: "Any other main complaint, or none described",
}


def decide_bool(p_yes: float, prompted: bool, p_clear: float = 1.0) -> Optional[bool]:
    """Map Jev's p(yes) to True / False / None (unknown).

    Only the field the user was actually asked about can become False, and only
    when the answer is also rated clear. A low score on a field nobody asked
    about just means "not mentioned", and the triage engine never treats "not
    mentioned" as "no". A "yes" is never blocked: escalating is the safe side.
    """
    if prompted:
        if p_yes >= YES_THRESHOLD:
            return True
        if p_yes <= NO_THRESHOLD and p_clear >= CLEAR_ANSWER_THRESHOLD:
            return False
        return None
    return True if p_yes >= UNPROMPTED_YES_THRESHOLD else None


def _noul_question(key: str) -> dict[str, Any]:
    spec = FIELDS[key]
    return {
        "type": "noul",
        "instructions": f"Based only on the patient's answer, does the patient report: {spec.description}",
        "criteria": {
            "true": f"The answer confirms this (or any part of it): {spec.description}",
            "false": "The answer clearly denies this. Hedged, unsure, or unrelated answers are not a denial.",
        },
    }


def _decision_state(user_message: str, last_question: Optional[str]) -> str:
    parts = ["Fictional patient intake (educational prototype)."]
    if last_question:
        parts.append(f"Question the intake assistant asked: {last_question}")
    parts.append(f"Patient's answer: {user_message}")
    return "\n".join(parts)


class JevClient:
    """Thin client for OpenRouter's Decisions API. Swap in a fake for tests."""

    def __init__(self, api_key: str, model: str = JEV_MODEL, url: str = JEV_URL, timeout: float = 30.0):
        self.api_key, self.model, self.url, self.timeout = api_key, model, url, timeout

    def decide(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        import httpx  # uses certifi, so it works with python.org builds on macOS

        resp = httpx.post(
            self.url,
            json={"model": self.model, "state": state, "questions": questions},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()


class JevIntakeAgent:
    """Jev decides yes/no facts and the category; a language agent does the rest.

    * Every boolean field and the complaint category come from Jev's
      probabilities via the thresholds above, in ONE request per turn.
    * The wrapped language agent (OpenAI or offline) still extracts free-form
      values (age, duration, severity) and writes the acknowledgement text.
    * If Jev fails, the language agent's extraction is used unchanged.
    """

    def __init__(self, client: Any, language_agent: Any):
        self.client = client
        self.language = language_agent
        self.name = f"Jev decision model + {language_agent.name}"

    def extract(self, user_message, last_question, pending_field, facts) -> Extraction:
        base = self.language.extract(user_message, last_question, pending_field, facts)

        # Fields already reported present are sticky in merge_facts, so skip them.
        bool_keys = [k for k, s in FIELDS.items() if s.type == "bool" and facts.get(k) is not True]
        questions: dict[str, Any] = {k: _noul_question(k) for k in bool_keys}
        if pending_field in bool_keys:
            questions[_CLARITY_KEY] = _CLARITY_QUESTION
        ask_category = "category" not in facts
        if ask_category:
            questions["category"] = {
                "type": "choice",
                "instructions": "What is the patient's main complaint?",
                "criteria": _CATEGORY_CRITERIA,
            }

        try:
            resp = self.client.decide(_decision_state(user_message, last_question), questions)
            answers = resp["answers"]
        except Exception as exc:  # network / API problems -> language-agent fallback
            return self._fallback_extraction(base, pending_field, exc)

        # Free-form values come from the language agent; yes/no values only from Jev.
        out = {k: v for k, v in base.facts.items() if FIELDS[k].type != "bool" and k != "category"}
        scores: dict[str, float] = {}
        uncertain: list[str] = []
        clarity = answers.get(_CLARITY_KEY) or {}
        p_clear = float(clarity["noul"]) if clarity.get("type") == "noul" else 0.0  # missing -> not clear

        for k in bool_keys:
            a = answers.get(k) or {}
            if a.get("type") != "noul":  # refusal or missing -> leave unknown
                continue
            p = float(a["noul"])
            prompted = k == pending_field
            value = decide_bool(p, prompted, p_clear if prompted else 1.0)
            if value is not None:
                out[k], scores[k] = value, p
            elif prompted:
                uncertain.append(f"{FIELDS[k].label}: answer unclear "
                                 f"(p(yes)={p:.2f}, p(clear)={p_clear:.2f}), not accepted")

        if ask_category:
            a = answers.get("category") or {}
            if a.get("type") == "choice":
                p = float(a.get("probabilities", {}).get(a.get("choice"), 0.0))
                if p >= CATEGORY_MIN_PROB:
                    out["category"], scores["category"] = a["choice"], p
                elif pending_field == "category":
                    uncertain.append(f"Main complaint: unclear (best guess {a.get('choice')}, p={p:.2f})")
        elif "category" in base.facts:
            out["category"] = base.facts["category"]

        cost = float((resp.get("usage") or {}).get("cost") or 0.0)
        return Extraction(facts=sanitize_facts(out), notes=base.notes, scores=scores,
                          uncertain=uncertain, cost=cost)

    def _fallback_extraction(self, base: Extraction, pending_field, exc: Exception) -> Extraction:
        """Use the language agent's facts, but never let it rule a warning sign out.

        Without Jev there is no threshold or clarity check, so a language-agent
        "yes" is kept (escalating is the safe side) while a "no" is dropped and
        the field stays unknown. Only Jev plus the clarity check can record "no".
        """
        facts = {k: v for k, v in base.facts.items() if not (FIELDS[k].type == "bool" and v is False)}
        dropped = [k for k in base.facts if k not in facts]
        uncertain = list(base.uncertain)
        if pending_field in dropped:
            uncertain.append(f"{FIELDS[pending_field].label}: \"no\" not accepted "
                             f"(Jev unavailable, can't confirm the answer is clear)")
        note = (f"Jev unavailable ({type(exc).__name__}); used {self.language.name} for yes/no fields, "
                f"accepting only \"yes\" answers.")
        if dropped:
            note += f" Not accepted as \"no\": {', '.join(dropped)}."
        return Extraction(facts=facts, notes=" ".join(filter(None, [base.notes, note])),
                          scores=base.scores, uncertain=uncertain, cost=base.cost)

    def phrase_question(self, field_key, facts, history) -> str:
        return self.language.phrase_question(field_key, facts, history)
