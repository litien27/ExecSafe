
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# Read your final held-out test predictions
df = pd.read_csv(
    RESULTS / "test_predictions_v4.csv"
)

features = ["SC", "PC", "ES", "UA"]

# Check that all 32 classifications succeeded
assert len(df) == 32
assert (df["status"] == "success").all()

for feature in features:

    y_true = df[f"{feature}_true"].astype(int)
    y_pred = df[f"{feature}_pred"].astype(int)

    fig, ax = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["Absent (0)", "Present (1)"],
        cmap="Blues",
        values_format="d",
        ax=ax
    )

    ax.set_title(f"{feature} - Held-out Test (n=32)")
    ax.set_xlabel("LLM Prediction")
    ax.set_ylabel("Human Reference")

    fig.tight_layout()

    fig.savefig(
        RESULTS / f"confusion_matrix_{feature}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"{feature} confusion matrix saved!")

print("\nAll four charts generated.")
