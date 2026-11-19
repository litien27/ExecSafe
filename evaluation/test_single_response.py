
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
prompt_path = PROJECT_ROOT / "evaluation/prompts/extraction_v1.txt"
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
example = development.iloc[0]

print("\nResponse ID:", example["response_id"])
print("Participant response:", example["response_text"])

# 9. Send the response to the LLM
response = client.responses.create(
    model="gpt-4.1-mini",
    instructions=prompt,
    input=(
        "Classify the following participant response:\n\n"
        + str(example["response_text"])
    )
)

# 10. Convert the model output into Python data

# Check the API response before parsing JSON
print("\nAPI response status:", response.status)
print("Incomplete details:", response.incomplete_details)
print("Raw model output:", repr(response.output_text))

if not response.output_text.strip():
    raise ValueError(
        "The model returned no text. Check the API response status."
    )

result = json.loads(response.output_text)


# 11. Display the LLM's classification
print("\nLLM extraction:")
print(json.dumps(result, indent=2))

# 12. Display the human reference coding
print("\nHuman reference coding:")

for feature in ["SC", "PC", "ES", "UA"]:
    print(
        feature,
        int(example[f"Final_{feature}"])
    )
