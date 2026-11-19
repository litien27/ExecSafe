

from pathlib import Path
import json
import os
import sys
import pandas as pd

from dotenv import load_dotenv

# Locate the ExecSafe project folder
ROOT = Path(__file__).resolve().parent.parent

# Allow Python to import agent.py from the project root
sys.path.insert(0, str(ROOT))

# Load environment variables
load_dotenv(ROOT / ".env", override=True)

# Import the actual ExecSafe question-generation function
from agent import (
    STAGE_QUESTIONS,
    OPEN_ANSWER_CHOICES,
    generated_follow_up
)

RESULTS = ROOT / "evaluation/results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Four fictional elaborations per scenario stage
cases = [
    (1, "I would temporarily stop the affected lifting operations until the risks are reviewed."),
    (1, "I would meet the site management team before deciding what action to take."),
    (1, "I would introduce additional supervision while allowing unaffected activities to continue."),
    (1, "I would take a different approach, but I am not sure what is most appropriate yet."),

    (2, "I need the updated completion forecast before deciding whether to maintain my decision."),
    (2, "I want to understand how the accelerated programme affects worker fatigue."),
    (2, "I need the latest productivity figures and safety inspection records."),
    (2, "I cannot decide until the project team provides more information."),

    (3, "I would request an independent review because the two reports contradict each other."),
    (3, "I would revise the recovery programme and reduce concurrent high-risk activities."),
    (3, "I would consult both the Project Director and Head of WSH before deciding."),
    (3, "I would continue with the current arrangements but monitor the situation closely."),
]

# Match the actual open-answer choices in agent.py
choice_by_stage = {
    stage: choice
    for stage, choice in OPEN_ANSWER_CHOICES
}

output = RESULTS / "elicitation_questions_v1.csv"

# Prevent accidental replacement of collected results
if output.exists():
    raise SystemExit(
        "Elicitation results already exist. "
        "Preserve them before starting another run."
    )

rows = []

for index, (stage, elaboration) in enumerate(cases, start=1):
    scenario = STAGE_QUESTIONS[stage - 1]
    case_id = f"E{index:02d}"

    stage_record = {
        "stage": stage,
        "title": scenario["title"],
        "context": scenario["context"],
        "fixed_question": scenario["question"],
        "choice": choice_by_stage[stage],
        "elaboration": elaboration,
    }

    print(f"Generating question for {case_id}...", flush=True)

    try:
        result = generated_follow_up(stage_record)
        generated_question = result["next_question"]
        status = "success"
        error = ""
    except Exception as exc:
        generated_question = ""
        status = "error"
        error = str(exc)
        print(f"{case_id}: {error}", flush=True)

    rows.append({
        "case_id": case_id,
        "stage": stage,
        "scenario_title": scenario["title"],
        "scenario_context": scenario["context"],
        "selected_choice": choice_by_stage[stage],
        "participant_elaboration": elaboration,
        "generated_question": generated_question,
        "status": status,
        "error": error,
    })

    # Save progress after every case
    pd.DataFrame(rows).to_csv(output, index=False)

    print(f"{case_id}: {status}", flush=True)

print("\nGeneration complete.")
print("Saved to:", output)
print(
    "Successful questions:",
    sum(row["status"] == "success" for row in rows),
    "/",
    len(cases)
)
