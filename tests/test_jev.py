"""Tests for the Jev decision-model agent, using a fake Jev client (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import (  # noqa: E402
    CLEAR_ANSWER_THRESHOLD, NO_THRESHOLD, UNPROMPTED_YES_THRESHOLD, YES_THRESHOLD, JevIntakeAgent, OfflineIntakeAgent, decide_bool,
)
from triage_engine import Outcome  # noqa: E402
from workflow import TriageSession  # noqa: E402

NOT_MENTIONED = 0.02  # what Jev typically returns for a finding the answer doesn't mention


class FakeJev:
    """Returns preset p(yes) per field; everything else looks 'not mentioned'."""

    def __init__(self, scores=None, category=("headache", 0.95), fail=False, clear=0.95):
        self.scores = scores or {}
        self.clear = clear
        self.category = category
        self.fail = fail
        self.calls: list[dict] = []

    def decide(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        if self.fail:
            raise ConnectionError("network down")
        answers = {}
        for key, q in questions.items():
            if key == "_answer_is_clear":
                answers[key] = {"type": "noul", "noul": self.clear}
            elif q["type"] == "noul":
                answers[key] = {"type": "noul", "noul": self.scores.get(key, NOT_MENTIONED)}
            else:
                label, p = self.category
                answers[key] = {"type": "choice", "choice": label, "probabilities": {label: p}}
        return {"answers": answers, "usage": {"cost": 0.00002}}


def jev_session(fake: FakeJev) -> TriageSession:
    return TriageSession(JevIntakeAgent(fake, OfflineIntakeAgent()))


# ---------------------------------------------------------------------------
# Threshold mapping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p,prompted,expected", [
    (0.96, True, True),
    (YES_THRESHOLD, True, True),
    (0.50, True, None),          # hedged answer -> unknown, re-ask
    (NO_THRESHOLD, True, False),
    (0.01, True, False),
    (0.80, False, None),         # volunteered finding needs more evidence
    (UNPROMPTED_YES_THRESHOLD, False, True),
    (0.01, False, None),         # not mentioned is NOT a "no"
])
def test_decide_bool_thresholds(p, prompted, expected):
    assert decide_bool(p, prompted) is expected


def test_unclear_answer_cannot_become_no_but_can_become_yes():
    unclear = CLEAR_ANSWER_THRESHOLD - 0.3
    assert decide_bool(0.12, prompted=True, p_clear=unclear) is None   # "Maybe? Hard to tell."
    assert decide_bool(0.12, prompted=True, p_clear=0.9) is False
    assert decide_bool(0.90, prompted=True, p_clear=unclear) is True    # escalation is never blocked


def test_hedged_low_score_is_reasked_not_recorded_as_no():
    fake = FakeJev()
    s = jev_session(fake)
    s.send("I've had a headache for two days, 3/10, I'm 35 years old.")
    pending = s.state["pending_field"]
    fake.scores, fake.clear = {pending: 0.12}, 0.10
    state = s.send("maybe? hard to tell")
    assert pending not in state["facts"]
    assert state["pending_field"] == pending


# ---------------------------------------------------------------------------
# Agent behaviour inside the LangGraph workflow
# ---------------------------------------------------------------------------
def test_unmentioned_findings_are_not_recorded_as_no():
    fake = FakeJev()
    state = jev_session(fake).send("I've had a headache for two days, 3/10, I'm 35 years old.")
    bools = {k: v for k, v in state["facts"].items() if isinstance(v, bool)}
    assert bools == {}  # every red flag scored ~0.02, none may become False
    assert state["facts"]["category"] == "headache"
    assert state["status"] == "intake"


def test_confident_yes_to_asked_question_routes_to_er():
    fake = FakeJev()
    s = jev_session(fake)
    s.send("I've had a headache for two days, 3/10, I'm 35 years old.")
    pending = s.state["pending_field"]
    fake.scores = {pending: 0.96}
    state = s.send("hmm, sort of, it's hard to describe")  # offline keywords would miss this
    assert state["facts"][pending] is True
    assert state["fact_scores"][pending] == pytest.approx(0.96)
    assert state["result"]["outcome"] == Outcome.ER.value


def test_unclear_answer_is_reasked_not_guessed():
    fake = FakeJev()
    s = jev_session(fake)
    s.send("I've had a headache for two days, 3/10, I'm 35 years old.")
    pending = s.state["pending_field"]
    fake.scores = {pending: 0.55}
    state = s.send("maybe? I'm not sure")
    assert pending not in state["facts"]
    assert state["pending_field"] == pending  # asked again
    assert state["asked"][pending] == 2
    assert len(state["uncertain"]) == 1


def test_volunteered_red_flag_is_caught_immediately():
    fake = FakeJev(scores={"fainting": 0.93}, category=("abdominal", 0.9))
    state = jev_session(fake).send("Stomach pain since this morning and I passed out briefly")
    assert state["facts"]["fainting"] is True
    assert state["result"]["outcome"] == Outcome.ER.value


def test_low_confidence_category_is_not_accepted():
    fake = FakeJev(category=("respiratory", 0.40))
    state = jev_session(fake).send("I just feel off")
    assert "category" not in state["facts"]
    assert state["pending_field"] == "category"


def test_jev_failure_falls_back_to_language_agent():
    fake = FakeJev(fail=True)
    state = jev_session(fake).send("I have the worst headache of my life, it came on suddenly")
    assert state["result"]["outcome"] == Outcome.ER.value  # offline agent still caught it
    assert any("Jev unavailable" in n for n in state["agent_notes"])


def test_full_intake_with_clear_denials_reaches_primary_care():
    fake = FakeJev()
    s = jev_session(fake)
    state = s.send("I've had a headache for two days, 3/10, I'm 35 years old.")
    for _ in range(20):
        if state["status"] == "complete":
            break
        fake.scores = {state["pending_field"]: 0.01}  # a clear "no" to whatever was asked
        state = s.send("no, nothing like that")
    assert state["result"]["outcome"] == Outcome.PRIMARY_CARE.value
    assert state["decision_cost"] > 0


def test_reported_red_flag_is_not_rescored():
    fake = FakeJev(scores={"chest_pain": 0.01})
    agent = JevIntakeAgent(fake, OfflineIntakeAgent())
    ext = agent.extract("no", "Any fever?", "fever", {"category": "respiratory", "chest_pain": True})
    assert "chest_pain" not in fake.calls[-1]["questions"]  # a "yes" already on record is never re-asked
    assert "chest_pain" not in ext.facts
