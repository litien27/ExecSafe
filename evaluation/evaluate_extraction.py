
from pathlib import Path
import json

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# 1. Find the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Load your API key from .env
load_dotenv(PROJECT_ROOT / ".env")

# 3. Connect to OpenAI
client = OpenAI()

# 4. Load the extraction prompt
prompt_path = PROJECT_ROOT / "evaluation/prompts/extraction_v4.txt"
prompt = prompt_path.read_text(encoding="utf-8")
print("\nPrompt file:", prompt_path.resolve())
print("Prompt length:", len(prompt))
print("Prompt preview:", prompt[:500])

# 5. Read your reviewed Excel workbook
file_path = (
    PROJECT_ROOT
    / "evaluation/data/ExecSafe_Coding_Benchmark_Review_Package_DoneCorrection.xlsx"
)

benchmark = pd.read_excel(
    file_path,
    sheet_name="Benchmark DRAFT"
)

adjudication = pd.read_excel(
    file_path,
    sheet_name="Adjudication"
)

# 6. Combine the responses with their final reference labels
df = benchmark[["response_id", "split", "response_text"]].merge(
    adjudication[
        ["response_id", "Final_SC", "Final_PC", "Final_ES", "Final_UA"]
    ],
    on="response_id",
    validate="one_to_one"
)

# 7. Select development responses only
development = df[
    df["split"].astype(str).str.strip().str.lower() == "development"
]

print("Development responses:", len(development))

assert len(development) == 16

# 8. Select the first development response

# Test all 16 development responses

from sklearn.metrics import classification_report

features = ["SC", "PC", "ES", "UA"]
results = []

for _, example in development[
    development["response_id"] == "R013"
].iterrows():

    response_id = example["response_id"]
    response_text = str(example["response_text"])

    print(f"\nProcessing {response_id}...", flush=True)

    try:
        # Send the response to OpenAI
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=prompt,
            input=(
                "Classify this participant response:\n\n"
                + response_text
            )
        )

        # Convert JSON output to Python dictionary
        print(
    f"{response_id} raw output:",
    repr(response.output_text),
    flush=True
)
        # Remove optional Markdown JSON formatting
        raw_text = response.output_text.strip()

        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines[-1].strip() == "```":
                raw_text = "\n".join(lines[1:-1]).strip()

        result = json.loads(raw_text)

        # Validate all four labels
        for feature in features:
            if type(result.get(feature)) is not int:
                raise ValueError(f"Invalid label for {feature}")

            if result[feature] not in (0, 1):
                raise ValueError(f"Label must be 0 or 1: {feature}")

        evidence = result.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("Missing evidence dictionary")

        # Check evidence fidelity separately from classification
        evidence_errors = []

        for feature in features:
            excerpt = evidence.get(feature)
            if result[feature] == 1:
                if not isinstance(excerpt, str) or not excerpt.strip():
                    evidence_errors.append(
                        f"{feature}: missing evidence"
                    )
                elif excerpt not in response_text:
                    evidence_errors.append(
                        f"{feature}: evidence is not an exact substring"
                    )
            elif excerpt is not None:
                evidence_errors.append(
                    f"{feature}: unexpected evidence for absent feature"
                )
        evidence_status = (
            "valid" if not evidence_errors else "invalid"
        )

        # Save reference labels, predictions, and evidence quality
        row = {
            "response_id": response_id,
            "response_text": response_text,
            "status": "success",
            "evidence_status": evidence_status,
            "evidence_errors": "; ".join(evidence_errors)
        }

        for feature in features:
            row[f"{feature}_true"] = int(
                example[f"Final_{feature}"]
            )
            row[f"{feature}_pred"] = result[feature]
            row[f"{feature}_evidence"] = evidence[feature]

        results.append(row)

        print(f"{response_id}: completed")

    except Exception as error:
        print(f"{response_id}: ERROR - {error}")

        results.append({
            "response_id": response_id,
            "response_text": response_text,
            "status": "error",
            "error": str(error)
        })

# Save all outputs
results_dir = PROJECT_ROOT / "evaluation/results"
results_dir.mkdir(parents=True, exist_ok=True)

output_path = results_dir / "development_predictions_v4_R013.csv"

results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)

# Check successful responses
successful = results_df[
    results_df["status"] == "success"
].copy()

print(
    f"\nSuccessful extractions: "
    f"{len(successful)}/{len(development)}"
)

# Calculate metrics only if all responses succeeded
if len(successful) == len(development):

    y_true = successful[
        [f"{f}_true" for f in features]
    ].to_numpy(dtype=int)

    y_pred = successful[
        [f"{f}_pred" for f in features]
    ].to_numpy(dtype=int)

    report = classification_report(
        y_true,
        y_pred,
        target_names=features,
        zero_division=0
    )

    print("\nDevelopment-set classification report:")
    print(report)

else:
    print(
        "\nSome responses failed. Fix the errors before "
        "calculating the full development-set metrics."
    )
