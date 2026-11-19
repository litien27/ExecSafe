
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report

# Step 1: Find the results folder
results_dir = Path(__file__).resolve().parent / "results"

# Step 2: Read the saved V4 results
original = pd.read_csv(
    results_dir / "development_predictions_v4_backup.csv"
)

# Read the R013 retry result
retry = pd.read_csv(
    results_dir / "development_predictions_v4_R013.csv"
)

# Step 3: Keep successful responses only
original = original[original["status"] == "success"]
retry = retry[retry["status"] == "success"]

# Step 4: Combine both datasets
combined = pd.concat([original, retry], ignore_index=True)

# Check that all 16 responses are included
assert len(combined) == 16
assert combined["response_id"].is_unique

# Step 5: Save the complete results
output_path = results_dir / "development_predictions_v4_final.csv"

combined.to_csv(output_path, index=False)

print("Combined results saved to:", output_path)

# Step 6: Calculate precision, recall and F1
features = ["SC", "PC", "ES", "UA"]

y_true = combined[
    [f"{f}_true" for f in features]
].to_numpy(dtype=int)

y_pred = combined[
    [f"{f}_pred" for f in features]
].to_numpy(dtype=int)

print("\nPROMPT V4 DEVELOPMENT RESULTS")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=features,
        zero_division=0
    )
)

# Step 7: Check evidence fidelity
print("\nEVIDENCE FIDELITY")

print(combined["evidence_status"].value_counts())
