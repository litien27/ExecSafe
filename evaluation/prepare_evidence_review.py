
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

df = pd.read_csv(RESULTS / "test_predictions_v4.csv")

features = ["SC", "PC", "ES", "UA"]
rows = []

for _, response in df.iterrows():
    if response["status"] != "success":
        continue

    for feature in features:
        if int(response[f"{feature}_pred"]) == 1:
            rows.append({
                "response_id": response["response_id"],
                "response_text": response["response_text"],
                "feature": feature,
                "llm_label": 1,
                "human_reference": int(
                    response[f"{feature}_true"]
                ),
                "llm_evidence": response[f"{feature}_evidence"],
                "evidence_support": "",
                "reviewer": "",
                "review_notes": ""
            })

review = pd.DataFrame(rows)

output = RESULTS / "evidence_review_template.csv"
review.to_csv(output, index=False)

print("Positive predictions to review:", len(review))
print("Saved to:", output)
