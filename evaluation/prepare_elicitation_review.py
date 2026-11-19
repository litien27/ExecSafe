
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# Read generated questions
df = pd.read_csv(
    RESULTS / "elicitation_questions_v1.csv"
)

# Confirm all 12 questions were generated
assert len(df) == 12
assert (df["status"] == "success").all()

# Prepare reviewer worksheet
review = df[
    [
        "case_id",
        "stage",
        "scenario_context",
        "selected_choice",
        "participant_elaboration",
        "generated_question"
    ]
].copy()

# Add review columns
review["neutrality"] = ""
review["relevance"] = ""
review["clarity"] = ""
review["context_fidelity"] = ""
review["leading_flag"] = ""
review["review_notes"] = ""
review["reviewer"] = ""

# Save the worksheet
output = RESULTS / "elicitation_review_template.csv"

review.to_csv(output, index=False)

print("Reviewer worksheet saved to:", output)
print("Questions for review:", len(review))
