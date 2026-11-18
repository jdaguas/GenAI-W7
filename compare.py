"""Compare yes/no fact extraction: OpenAI alone vs. the Jev decision model.

Each case is one follow-up question, a fictional patient answer, and the value
the triage engine SHOULD receive: True, False, or None (unclear -> re-ask).

Run:  python compare.py        (uses keys from .streamlit/secrets.toml)
"""

from __future__ import annotations

import re
import time
import tomllib
from pathlib import Path

from agent import JevClient, JevIntakeAgent, OfflineIntakeAgent, OpenAIIntakeAgent
from triage_engine import FIELDS

CASES = [
    # (field asked about, patient answer, expected value)
    ("chest_pain", "Yes, a crushing pressure in the middle of my chest.", True),
    ("chest_pain", "No chest pain at all.", False),
    ("chest_pain", "Just some tightness when I climb stairs.", True),
    ("chest_pain", "I don't think so, but I'm not totally sure.", None),
    ("fever", "I feel a bit warm but I haven't checked.", None),
    ("fever", "It was 101.5 this morning.", True),
    ("fever", "Thermometer says 98.6.", False),
    ("fainting", "Nope.", False),
    ("fainting", "Everything went gray and I had to sit down so I didn't fall.", True),
    ("neuro_deficit", "My left hand has felt kind of numb since this morning.", True),
    ("neuro_deficit", "None of those.", False),
    ("neuro_deficit", "My speech is fine, no weakness.", None),  # denies only part of the list
    ("wheezing", "My partner says there's a whistling sound when I breathe.", True),
    ("wheezing", "Maybe? Hard to tell.", None),
    ("persistent_vomiting", "I threw up once last night, nothing since.", False),
    ("persistent_vomiting", "I can't even keep water down.", True),
]
CATEGORY_FOR = {"chest_pain": "respiratory", "fever": "headache", "fainting": "abdominal",
                "neuro_deficit": "headache", "wheezing": "respiratory", "persistent_vomiting": "abdominal"}


def show(v):
    return {True: "yes", False: "no", None: "unclear"}[v]


def main() -> None:
    secrets = tomllib.loads(Path(".streamlit/secrets.toml").read_text())
    gpt = OpenAIIntakeAgent(secrets["OPENAI_API_KEY"], model=secrets.get("OPENAI_MODEL") or "gpt-4o-mini")
    jev = JevIntakeAgent(JevClient(secrets["OPENROUTER_API_KEY"]), OfflineIntakeAgent())

    totals = {"OpenAI": [0, 0.0], "Jev": [0, 0.0]}  # correct, seconds
    jev_cost = 0.0
    print(f"{'Field':<20} {'Answer':<58} {'Expected':<8} {'OpenAI':<8} {'Jev':<8} Jev scores")
    print("-" * 112)
    for field, answer, expected in CASES:
        facts = {"category": CATEGORY_FOR[field]}
        question = FIELDS[field].question
        row = {}
        for name, agent in (("OpenAI", gpt), ("Jev", jev)):
            t = time.perf_counter()
            ext = agent.extract(answer, question, field, facts)
            totals[name][1] += time.perf_counter() - t
            row[name] = ext.facts.get(field)
            totals[name][0] += row[name] is expected
            if name == "Jev":
                jev_cost += ext.cost
                p = ext.scores.get(field)
                unclear = next((u for u in ext.uncertain if u.startswith(FIELDS[field].label)), "")
                m = re.search(r"\((p\(yes\)=[\d.]+, p\(clear\)=[\d.]+)\)", unclear)
                p_txt = f"{p:.2f}" if p is not None else (m.group(1) if m else "")
        mark = lambda n: show(row[n]) + ("" if row[n] is expected else " ✗")
        print(f"{field:<20} {answer[:57]:<58} {show(expected):<8} {mark('OpenAI'):<8} {mark('Jev'):<8} {p_txt}")

    n = len(CASES)
    print("-" * 112)
    for name, (correct, secs) in totals.items():
        print(f"{name:<7} {correct}/{n} correct   avg {secs / n:.2f}s per answer")
    print(f"Jev total cost: ${jev_cost:.6f} for {n} turns (each turn scores every yes/no field at once)")


if __name__ == "__main__":
    main()
