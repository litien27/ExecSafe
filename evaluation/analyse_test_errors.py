
from pathlib import Path
import pandas as pd

# Locate saved test results
ROOT = Path(__file__).resolve().parent
file_path = ROOT / "results/test_predictions_v4.csv"

df = pd.read_csv(file_path)

features = ["SC", "PC", "ES", "UA"]
errors_list = []

print("\nEXECSAFE HELD-OUT TEST ERROR ANALYSIS")

for feature in features:
    errors = df[
        df[f"{feature}_true"] != df[f"{feature}_pred"]
    ]

    print(f"\n{feature}: {len(errors)} errors")

    for _, row in errors.iterrows():
        error_type = (
            "False Positive"
            if row[f"{feature}_pred"] == 1
            else "False Negative"
        )

        print("\nResponse ID:", row["response_id"])
        print("Response:", row["response_text"])
        print("Feature:", feature)
        print("Error type:", error_type)
        print("Human label:", row[f"{feature}_true"])
        print("LLM label:", row[f"{feature}_pred"])
        print("LLM evidence:", row[f"{feature}_evidence"])

        errors_list.append({
            "response_id": row["response_id"],
            "response_text": row["response_text"],
            "feature": feature,
            "error_type": error_type,
            "human_label": row[f"{feature}_true"],
            "llm_label": row[f"{feature}_pred"],
            "llm_evidence": row[f"{feature}_evidence"]
        })

# Save error analysis table
errors_df = pd.DataFrame(errors_list)

output_path = ROOT / "results/test_errors_v4.csv"
errors_df.to_csv(output_path, index=False)

print("\nTotal classification errors:", len(errors_df))
print("Error analysis saved to:", output_path)
