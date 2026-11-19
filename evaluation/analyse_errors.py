
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

file_path = (
    ROOT / "evaluation/results/development_predictions_v2.csv"
)

df = pd.read_csv(file_path)

features = ["SC", "PC", "ES", "UA"]

print("\nMISCLASSIFIED DEVELOPMENT RESPONSES")

for feature in features:
    errors = df[
        df[f"{feature}_true"] != df[f"{feature}_pred"]
    ]

    print(f"\n{feature}: {len(errors)} errors")

    for _, row in errors.iterrows():
        print("\nResponse ID:", row["response_id"])
        print("Response:", row["response_text"])
        print("Human:", row[f"{feature}_true"])
        print("LLM:", row[f"{feature}_pred"])
        print("Evidence:", row[f"{feature}_evidence"])
