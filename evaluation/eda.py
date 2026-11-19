
from pathlib import Path
file_path = Path(
    "evaluation/data/ExecSafe_Coding_Benchmark_Review_Package_DoneCorrection.xlsx"
)
print("Workbook found:", file_path.exists())
import pandas as pd

# Read the worksheets
benchmark = pd.read_excel(file_path, sheet_name="Benchmark DRAFT")
adjudication = pd.read_excel(file_path, sheet_name="Adjudication")

# Check the number of rows and columns
print("Benchmark shape:", benchmark.shape)
print("Adjudication shape:", adjudication.shape)

# Display the column names
print("\nBenchmark columns:")
print(benchmark.columns.tolist())

print("\nAdjudication columns:")
print(adjudication.columns.tolist())
print(adjudication.columns.tolist())


# STEP 1: Combine benchmark and adjudication data

# Select additional information from the benchmark
info = benchmark[
    ["response_id", "response_type", "split", "word_count"]
]

# Merge with final adjudicated labels
df = adjudication.merge(
    info,
    on="response_id",
    how="left",
    validate="one_to_one"
)

# Rename the final labels
df = df.rename(columns={
    "Final_SC": "SC",
    "Final_PC": "PC",
    "Final_ES": "ES",
    "Final_UA": "UA"
})

features = ["SC", "PC", "ES", "UA"]

# Verify the dataset
assert len(df) == 48
assert df[features].notna().all().all()
assert df[features].isin([0, 1]).all().all()

print("\nDataset combined successfully!")
print("Final feature counts:")
print(df[features].sum())


# STEP 2: Generate feature distribution bar chart

import matplotlib.pyplot as plt

# Create results folder
results_dir = Path("evaluation/results")
results_dir.mkdir(parents=True, exist_ok=True)

# Count positive classifications
feature_counts = df[features].sum()

# Create chart
fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(feature_counts.index, feature_counts.values)

ax.set_title("Distribution of Observable Reasoning Features")
ax.set_xlabel("Reasoning Features")
ax.set_ylabel("Number of Positive Responses")
ax.set_ylim(0, 48)

# Add counts above each bar
for i, count in enumerate(feature_counts.values):
    ax.text(i, count + 1, str(int(count)), ha="center")

plt.tight_layout()

# Save chart
chart_path = results_dir / "feature_distribution.png"
fig.savefig(chart_path, dpi=300, bbox_inches="tight")

print("\nChart saved to:", chart_path.resolve())

plt.show()

# STEP 3: Feature Co-occurrence Heatmap

# Calculate how often two features appear together
cooccurrence = df[features].T.dot(df[features])

print("\nFeature co-occurrence matrix:")
print(cooccurrence)

# Create heatmap
fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(cooccurrence, cmap="Blues")

# Add feature names
ax.set_xticks(range(len(features)))
ax.set_yticks(range(len(features)))

ax.set_xticklabels(features)
ax.set_yticklabels(features)

# Display numbers inside each cell
for i in range(len(features)):
    for j in range(len(features)):
        value = int(cooccurrence.iloc[i, j])

        ax.text(
            j, i, str(value),
            ha="center",
            va="center",
            color="white" if value > 12 else "black"
        )

ax.set_title("Co-occurrence of Reasoning Features")

# Add colour scale
fig.colorbar(im, ax=ax, label="Number of Responses")

plt.tight_layout()

# Save heatmap
chart_path = results_dir / "feature_cooccurrence.png"
fig.savefig(chart_path, dpi=300, bbox_inches="tight")

print("\nHeatmap saved to:", chart_path.resolve())

plt.show()


# STEP 4: Human Inter-Rater Reliability

from sklearn.metrics import cohen_kappa_score

results = []

for feature in features:

    # Original independent coding by Reviewer A and B
    reviewer_a = df[f"A_{feature}"].astype(int)
    reviewer_b = df[f"B_{feature}"].astype(int)

    # Calculate percentage agreement
    agreement = (reviewer_a == reviewer_b).mean() * 100

    # Calculate Cohen's kappa
    kappa = cohen_kappa_score(reviewer_a, reviewer_b)

    results.append({
        "Feature": feature,
        "Agreement (%)": round(agreement, 1),
        "Cohen's Kappa": round(kappa, 3)
    })

# Convert results into a table
reliability = pd.DataFrame(results)

print("\nHuman Inter-Rater Reliability:")
print(reliability.to_string(index=False))

# Save results as CSV
reliability.to_csv(
    results_dir / "human_reliability.csv",
    index=False
)

# STEP 5: Plot Cohen's Kappa

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    reliability["Feature"],
    reliability["Cohen's Kappa"]
)

ax.set_title("Inter-Rater Reliability of Human Coding")
ax.set_xlabel("Reasoning Feature")
ax.set_ylabel("Cohen's Kappa")
ax.set_ylim(0, 1)

# Add kappa values above bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{height:.3f}",
        ha="center"
    )

plt.tight_layout()

# Save chart
chart_path = results_dir / "human_kappa.png"
fig.savefig(chart_path, dpi=300, bbox_inches="tight")

print("\nKappa chart saved to:", chart_path.resolve())

plt.show()

# STEP 5: Plot Cohen's Kappa

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    reliability["Feature"],
    reliability["Cohen's Kappa"]
)

ax.set_title("Inter-Rater Reliability of Human Coding")
ax.set_xlabel("Reasoning Feature")
ax.set_ylabel("Cohen's Kappa")
ax.set_ylim(0, 1)

# Add kappa values above bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{height:.3f}",
        ha="center"
    )

plt.tight_layout()

# Save chart
chart_path = results_dir / "human_kappa.png"
fig.savefig(chart_path, dpi=300, bbox_inches="tight")

print("\nKappa chart saved to:", chart_path.resolve())

plt.show()

# STEP 5: Plot Cohen's Kappa

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    reliability["Feature"],
    reliability["Cohen's Kappa"]
)

ax.set_title("Inter-Rater Reliability of Human Coding")
ax.set_xlabel("Reasoning Feature")
ax.set_ylabel("Cohen's Kappa")
ax.set_ylim(0, 1)

# Add kappa values above bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{height:.3f}",
        ha="center"
    )

plt.tight_layout()

# Save chart
chart_path = results_dir / "human_kappa.png"
fig.savefig(chart_path, dpi=300, bbox_inches="tight")

print("\nKappa chart saved to:", chart_path.resolve())

plt.show()

# STEP 6: Response Length by Scenario Stage

# Recalculate word count directly from the response text
df["word_count"] = df["response_text"].str.split().str.len()

# Print descriptive statistics
print("\nResponse Length by Stage:")
print(
    df.groupby("stage")["word_count"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

# Prepare data for boxplot
stage_data = [
    df.loc[df["stage"] == stage, "word_count"]
    for stage in sorted(df["stage"].unique())
]

# Create boxplot
fig, ax = plt.subplots(figsize=(8, 5))

ax.boxplot(stage_data)

ax.set_title("Response Length by Scenario Stage")
ax.set_xlabel("Scenario Stage")
ax.set_ylabel("Number of Words")

ax.set_xticks([1, 2, 3])
ax.set_xticklabels([
    "Stage 1: Safety",
    "Stage 2: Production",
    "Stage 3: Conflicting Info"
])

plt.tight_layout()

# Save chart
chart_path = results_dir / "response_length.png"
fig.savefig(chart_path, dpi=300, bbox_inches="tight")

print("\nResponse length chart saved to:", chart_path.resolve())

plt.show()

# STEP 7: Response Difficulty Distribution

# Count responses by type
type_counts = df["response_type"].value_counts()

print("\nResponse Type Distribution:")
print(type_counts)

# Create chart
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(type_counts.index, type_counts.values)

ax.set_title("Distribution of Synthetic Response Types")
ax.set_xlabel("Response Type")
ax.set_ylabel("Number of Responses")

# Add counts above bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.2,
        str(int(height)),
        ha="center"
    )

plt.xticks(rotation=25, ha="right")
plt.tight_layout()

# Save chart
chart_path = results_dir / "response_types.png"
fig.savefig(chart_path, dpi=300, bbox_inches="tight")

print("\nResponse type chart saved to:", chart_path.resolve())

plt.show()

# STEP 7: Response Difficulty Distribution

# Count responses by type
type_counts = df["response_type"].value_counts()

print("\nResponse Type Distribution:")
print(type_counts)

# Create chart
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(type_counts.index, type_counts.values)

ax.set_title("Distribution of Synthetic Response Types")
ax.set_xlabel("Response Type")
ax.set_ylabel("Number of Responses")

# Add counts above bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.2,
        str(int(height)),
        ha="center"
    )

plt.xticks(rotation=25, ha="right")
plt.tight_layout()

# Save chart
chart_path = results_dir / "response_types.png"
fig.savefig(chart_path, dpi=300, bbox_inches="tight")

print("\nResponse type chart saved to:", chart_path.resolve())

plt.show()

# STEP 7: Response Difficulty Distribution

# Count responses by type
type_counts = df["response_type"].value_counts()
print("\nAll response types:")
print(df["response_type"].unique())

print("\nResponse type counts:")
print(type_counts)

print("\nResponse Type Distribution:")
print(type_counts)

# Create chart
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(type_counts.index, type_counts.values)

ax.set_title("Distribution of Synthetic Response Types")
ax.set_xlabel("Response Type")
ax.set_ylabel("Number of Responses")

# Add counts above bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.2,
        str(int(height)),
        ha="center"
    )

plt.xticks(rotation=25, ha="right")
plt.tight_layout()

# Save chart
chart_path = results_dir / "response_types.png"
fig.savefig(chart_path, dpi=300, bbox_inches="tight")

print("\nResponse type chart saved to:", chart_path.resolve())

plt.show()
