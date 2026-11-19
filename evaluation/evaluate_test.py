
from pathlib import Path
import json
import os
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation/results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Load API key
load_dotenv(ROOT / ".env", override=True)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Load frozen prompt
prompt = (
    ROOT / "evaluation/prompts/extraction_v4.txt"
).read_text(encoding="utf-8")

# Load dataset
workbook = (
    ROOT / "evaluation/data/"
    "ExecSafe_Coding_Benchmark_Review_Package_DoneCorrection.xlsx"
)

benchmark = pd.read_excel(
    workbook, sheet_name="Benchmark DRAFT"
)

adjudication = pd.read_excel(
    workbook, sheet_name="Adjudication"
)

df = benchmark[
    ["response_id", "split", "response_text"]
].merge(
    adjudication[
        ["response_id", "Final_SC", "Final_PC",
         "Final_ES", "Final_UA"]
    ],
    on="response_id",
    validate="one_to_one"
)

# Select held-out test responses only
test = df[
    df["split"].astype(str).str.strip().str.lower() == "test"
].copy()

assert len(test) == 32

features = ["SC", "PC", "ES", "UA"]

output_path = RESULTS / "test_predictions_v4.csv"

# Prevent accidental overwriting of final test results
if output_path.exists():
    raise SystemExit(
        "Test results already exist. Do not rerun or overwrite "
        "the held-out evaluation without documenting why."
    )

results = []

for _, example in test.iterrows():
    response_id = example["response_id"]
    response_text = str(example["response_text"])

    print(f"Processing {response_id}...", flush=True)

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=prompt,
            input=(
                "Classify this participant response:\n\n"
                + response_text
            )
        )

        raw = response.output_text.strip()

        # Remove Markdown fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines[-1].strip() == "```":
                raw = "\n".join(lines[1:-1]).strip()

        result = json.loads(raw)

        for feature in features:
            if type(result.get(feature)) is not int:
                raise ValueError(f"Invalid {feature} label")
            if result[feature] not in (0, 1):
                raise ValueError(f"Invalid {feature} value")

        evidence = result.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("Missing evidence dictionary")

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
                        f"{feature}: non-exact excerpt"
                    )
            elif excerpt is not None:
                evidence_errors.append(
                    f"{feature}: unexpected evidence"
                )

        row = {
            "response_id": response_id,
            "response_text": response_text,
            "status": "success",
            "evidence_status": (
                "valid" if not evidence_errors else "invalid"
            ),
            "evidence_errors": "; ".join(evidence_errors)
        }

        for feature in features:
            row[f"{feature}_true"] = int(
                example[f"Final_{feature}"]
            )
            row[f"{feature}_pred"] = result[feature]
            row[f"{feature}_evidence"] = evidence.get(feature)

    except AuthenticationError:
        print("Authentication failed. Stopping evaluation.")
        raise

    except Exception as error:
        row = {
            "response_id": response_id,
            "response_text": response_text,
            "status": "error",
            "error": str(error)
        }

    results.append(row)

    # Save progress after every response
    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"{response_id}: {row['status']}", flush=True)

print("\nFinal test run complete.")

results_df = pd.DataFrame(results)

successful = results_df[
    results_df["status"] == "success"
].copy()

print(
    f"Successful classifications: "
    f"{len(successful)}/{len(test)}"
)

# Calculate metrics for parseable outputs only
if not successful.empty:
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

    print("\nHELD-OUT TEST CLASSIFICATION REPORT")
    print(report)

    print("\nEVIDENCE FIDELITY")
    print(successful["evidence_status"].value_counts())

    # Save numerical classification report
    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=features,
            zero_division=0,
            output_dict=True
        )
    ).transpose()

    report_df.to_csv(
        RESULTS / "test_classification_report_v4.csv"
    )

    # Save confusion matrices
    for i, feature in enumerate(features):
        matrix = confusion_matrix(
            y_true[:, i],
            y_pred[:, i],
            labels=[0, 1]
        )

        print(f"\n{feature} confusion matrix:")
        print(matrix)

        pd.DataFrame(
            matrix,
            index=["Human_0", "Human_1"],
            columns=["LLM_0", "LLM_1"]
        ).to_csv(
            RESULTS / f"test_confusion_{feature}_v4.csv"
        )
