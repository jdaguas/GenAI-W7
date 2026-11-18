# TriageFlow

An AI-powered medical intake and triage prototype built using Streamlit and Langgraph.

A user describes a fictional patient's symptoms in a chat. An AI intake agent turns each message into structured facts and asks one follow-up question at a time. A separate, rule-based triage engine then routes the case to one of four outcomes:

| Outcome | Meaning |
| --- | --- |
| **Emergency Department** | A predefined emergency warning sign was reported. |
| **Urgent Care** | No emergency signs; the example rules call for prompt in-person evaluation. |
| **Primary Care** | Every emergency and urgent screen was explicitly answered "no". |
| 🔍 **Needs Review** | The rules cannot safely choose a care setting (missing, unclear, or contradictory information, or a case out of scope). |

> ⚠️ **Educational prototype only.** TriageFlow is for **fictional adult cases**. It is not a validated clinical decision-support tool, and its output is not medical advice. The rules are illustrative examples that have not been clinically reviewed. In a real emergency, call 911.

## What's new in this version: Jev decision model

This version builds on the previous TriageFlow assignment. It adds **[Jev](https://openrouter.ai/typesafe/jev-1.14)**, TypeSafe's decision model, called through OpenRouter's Decisions API. Jev doesn't generate text. For each question it returns a **probability**.

**What changed:** the yes/no warning-sign answers, and the main complaint, are now decided by Jev's probabilities plus fixed thresholds in code. Before, GPT returned them directly. GPT still pulls out age, duration and severity and writes the acknowledgements. The rule engine, the order of questions, and every care-setting decision are unchanged.

| Situation | Rule (in [agent.py](agent.py)) | Result |
| --- | --- | --- |
| Answer to the question just asked, p(yes) ≥ 0.70 | `YES_THRESHOLD` | Finding recorded as **present** |
| p(yes) ≤ 0.15 **and** Jev rates the answer clear and complete (p ≥ 0.50) | `NO_THRESHOLD`, `CLEAR_ANSWER_THRESHOLD` | Finding recorded as **absent** |
| Anything else | — | **Unknown**: the question is asked again and the answer is listed as unclear |
| Finding volunteered without being asked, p(yes) ≥ 0.85 | `UNPROMPTED_YES_THRESHOLD` | Recorded as **present** (can never become "absent") |
| Main complaint, p ≥ 0.60 | `CATEGORY_MIN_PROB` | Category accepted |

Why the clarity check exists: p(yes) alone can't tell "no" apart from "unsure," because neither answer confirms anything. In testing, "Maybe? Hard to tell" scored p(yes) = 0.12, close to a real "no" (0.02–0.06). A second Jev question asks whether the answer is clear and complete, and only then can a warning sign be ruled out.

All yes/no fields are scored in **one Jev request per turn**, about $0.00009 per turn. If Jev fails, the app falls back to GPT's extraction, but only GPT's "yes" answers are accepted. Its "no" answers are left unknown and asked again, since only Jev plus the clarity check can rule a warning sign out. During an outage a case can still go to the ED, but it can't reach Urgent Care or Primary Care. The app shows each finding's Jev score, the answers it found unclear, and the running Jev cost. The handoff summary includes the scores too.

**GPT vs. Jev on 16 labeled answers** (`python compare.py`):

| | Correct | Avg time | Wrong in the unsafe direction |
| --- | --- | --- | --- |
| GPT-4o-mini alone | 15/16 | 0.88 s | 1 ("speech fine, no weakness" recorded as *no* for the whole neuro screen, though numbness and vision were never answered) |
| Jev + thresholds | 15/16 | 0.25 s | 0 (its miss was re-asking "threw up once, nothing since") |

Setup: add `OPENROUTER_API_KEY` to `.streamlit/secrets.toml`, then pick **Jev decision model** in the sidebar.

## Key design principle: the LLM never makes the triage decision

The language model handles language only: it understands free text and writes a short acknowledgement before each question. Every care-setting decision comes from plain Python rules that can be read and tested.

- **Emergency signs are checked on every message**, not only at the end of intake. A reported warning sign ends intake right away and routes the case to the ED.
- **A question that was never answered is not treated as "no".** Every warning sign for the complaint has to be explicitly ruled out before the case can get Urgent Care or Primary Care.
- **Once a warning sign is reported, a later "no" doesn't clear it.** The conflict is recorded and the finding stays.
- **Anything the LLM returns is validated.** Unknown keys (such as an attempt to set `"outcome"`) and invalid values are dropped.
- **The follow-up questions are fixed, reviewed text.** The LLM can't change what is being asked, so each answer is always recorded against the right finding.

## Architecture

```text
Streamlit chat ──► LangGraph workflow (one run per user message)

   START ─► extract ─► safety_screen ──(warning sign)──────────► finalize ─► END
                            │
                            └─► assess ──(info missing)──► ask ─► END   (wait for next message)
                                   └──(decidable / budget spent)──► finalize ─► END
```

| Node | What it does | Uses the LLM? |
| --- | --- | --- |
| `extract` | Turns the latest message into validated facts and merges them into the case | Yes (extraction) |
| `safety_screen` | Checks for emergency warning signs | No |
| `assess` | Asks the rule engine what is still missing and picks the next question, safety questions first | No |
| `ask` | Shows one question: an LLM acknowledgement plus the fixed question text | Acknowledgement only |
| `finalize` | Produces the outcome | No |

State (facts, transcript, which questions were asked, contradictions) is saved between turns with a LangGraph checkpointer, keyed by a per-case thread ID. Each question is asked at most twice, and a case gets at most 26 questions.

## Project structure

```text
triageflow/
├── app.py                         # Streamlit interface
├── agent.py                       # OpenAI intake agent + offline keyword agent
├── workflow.py                    # LangGraph state, nodes, routing, handoff summary
├── triage_engine.py               # Field catalog and triage rules (no LLM)
├── requirements.txt
├── tests/
│   └── test_triage.py             # Synthetic scenario tests
└── .streamlit/
    ├── secrets.toml.example       # Template for your API key
    └── secrets.toml               # Your local key (git-ignored, never commit)
```

## Getting started

**Requirements:** Python 3.10+ and, optionally, an OpenAI API key.

```bash
cd triageflow
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Add your API key** (optional):

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini" 
```

You can also set `OPENAI_API_KEY` as an environment variable instead.

**Run the app:**

```bash
.venv/bin/streamlit run app.py
```

> Use the Streamlit inside `.venv`. Running a global `streamlit` (for example, from Anaconda) fails with `ModuleNotFoundError: No module named 'openai'` because the packages are installed only in the project environment.

The app falls back to an offline keyword agent without an OpenAI API key, so you can demo and test it without an OpenAI account. The sidebar shows which agent is active.

## Using the app

1. Describe a fictional adult patient's main symptom, for example *"I've had a headache for two days."* You can also click one of the sidebar's sample scenarios.
2. Answer the follow-up questions. You can mention several details in one message; anything already answered is skipped.
3. The right-hand panel shows the extracted facts, how many warning signs have been checked, and what is still needed.
4. When intake finishes, the panel shows the outcome, the rules that fired, any information not established, and a downloadable handoff summary.
5. Click Start new case to begin again.

**Why did my case go to Needs Review?** Usually a screening question got a vague answer ("not sure", "haven't checked") twice. That is intended: an unknown warning sign can't be ruled out. The result panel lists those items under "Information not established".

## Supported complaints and example rules

The prototype covers three adult complaint groups: headache, abdominal pain, and respiratory symptoms. Other complaints and patients under 18 go to Needs Review.

Rules are checked in this order: emergency → scope (complaint group and age) → contradictions → required information → urgent → primary care.

**Emergency Department** (applies to any complaint once reported)

| ID | Finding | Must be screened for |
| --- | --- | --- |
| ER-01 | New confusion or decreased alertness | all |
| ER-02 | Chest pain or pressure | abdominal, respiratory |
| ER-03 | Fainting or near-fainting | abdominal |
| ER-H1 | Sudden-onset / worst-ever headache | headache |
| ER-H2 | Focal neurological symptoms | headache |
| ER-H3 | Fever with stiff neck | headache |
| ER-H4 | Headache after head injury | headache |
| ER-A1 | Vomiting blood | abdominal |
| ER-A2 | Bloody or black stool | abdominal |
| ER-A3 | Rigid abdomen / pain worse with movement | abdominal |
| ER-R1 | Severe breathing difficulty at rest | respiratory |
| ER-R2 | Blue or gray lips/face | respiratory |
| ER-R3 | Coughing up blood | respiratory |

**Urgent Care** (only after all emergency screens are negative)

| ID | Finding | Applies to |
| --- | --- | --- |
| UC-01 | Fever ≥ 100.4°F / 38°C | all |
| UC-02 | Severity 7/10 or higher | all |
| UC-H1 / UC-A1 | Persistent vomiting | headache, abdominal |
| UC-A2 | Possible pregnancy | abdominal |
| UC-A3 | Right lower abdominal pain | abdominal |
| UC-R1 | Short of breath with light activity | respiratory |
| UC-R2 | Wheezing | respiratory |

**Primary Care** also requires age, duration, and severity, plus a "no" to every emergency and urgent screen for the complaint group.

All rules and question wording live in [`triage_engine.py`](triage_engine.py), as data, so they can be reviewed in one place.

## Testing

```bash
.venv/bin/python -m pytest tests
```

The tests use synthetic cases and the offline agent, so they don't need an API key. They check that:

- an emergency warning sign in each complaint group routes to the ED, including on the first message;
- an urgent finding routes to Urgent Care;
- incomplete, contradictory, out-of-scope, or under-18 cases never become Primary Care;
- an unscreened warning sign is not treated as "no";
- an adversarial agent that tries to set the outcome or clear a warning sign can't override the rules;
- questions aren't repeated endlessly and intake always ends;
- a full intake with all answers negative reaches Primary Care.

## Limitations and next steps

- The rules are **illustrative and not clinically validated**. They need review by qualified clinicians before any real use.
- Only three complaint groups are supported, and each warning sign is asked about one at a time (an abdominal case takes about 13 questions).
- Sessions live in memory only and are lost when the app restarts.
- There are no privacy or security controls. Never enter real patient information.

Possible improvements: group related warning signs into one question, retrieve guidelines with source references, save sessions between restarts, add a clinician review/approval step, and add more complaint groups after evaluation.
