import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"
PLOTS_DIR = EXPERIMENTS_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def find_column(rows, candidates):
    if not rows:
        return None

    for candidate in candidates:
        if candidate in rows[0]:
            return candidate

    return None


def save_experiment_plot(
    filename,
    x_candidates,
    title,
    x_label,
    output_filename,
):
    path = EXPERIMENTS_DIR / filename

    if not path.exists():
        print(f"Skipping missing file: {filename}")
        return

    rows = read_csv(path)

    x_key = find_column(rows, x_candidates)
    y_key = find_column(
        rows,
        ["validation_loss", "val_loss"],
    )

    if x_key is None or y_key is None:
        print(f"Skipping {filename}: required columns not found.")
        return

    points = []

    for row in rows:
        try:
            x = float(row[x_key])
            y = float(row[y_key])

            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))

        except (ValueError, TypeError, KeyError):
            continue

    if not points:
        print(f"Skipping {filename}: no valid data.")
        return

    points.sort(key=lambda point: point[0])

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("Validation loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = PLOTS_DIR / output_filename
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# ARCHITECTURE EXPERIMENTS
# ============================================================

save_experiment_plot(
    "layer_scaling.csv",
    ["num_layers", "n_layer", "layers"],
    "Layer Scaling",
    "Number of Transformer layers",
    "layer_scaling.png",
)

save_experiment_plot(
    "embedding_scaling.csv",
    ["embedding_dim", "n_embd"],
    "Embedding Dimension Scaling",
    "Embedding dimension",
    "embedding_scaling.png",
)

save_experiment_plot(
    "context_scaling.csv",
    ["context_length", "block_size"],
    "Context Length Scaling",
    "Context length",
    "context_scaling.png",
)

save_experiment_plot(
    "attention_head_scaling.csv",
    ["num_heads", "heads"],
    "Attention Head Scaling",
    "Number of attention heads",
    "attention_head_scaling.png",
)


# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

save_experiment_plot(
    "learning_rate_scaling.csv",
    ["learning_rate", "max_lr"],
    "Learning Rate Scaling",
    "Maximum learning rate",
    "learning_rate_scaling.png",
)

save_experiment_plot(
    "extended_lr_scaling.csv",
    ["learning_rate", "max_lr"],
    "Extended Learning Rate Search",
    "Maximum learning rate",
    "extended_lr_scaling.png",
)

save_experiment_plot(
    "dropout_scaling.csv",
    ["dropout", "dropout_rate"],
    "Dropout Regularization",
    "Dropout rate",
    "dropout_scaling.png",
)


# ============================================================
# TRAINING-BUDGET PLOT
# ============================================================

budget_path = EXPERIMENTS_DIR / "training_budget_scaling.csv"

if budget_path.exists():

    rows = read_csv(budget_path)

    if rows:
        plt.figure(figsize=(8, 5))

        plotted = False

        for column, label in [
            ("train_loss", "Training loss"),
            ("validation_loss", "Validation loss"),
        ]:

            valid_points = []

            for row in rows:
                try:
                    x = float(row["training_steps"])
                    y = float(row[column])

                    if math.isfinite(x) and math.isfinite(y):
                        valid_points.append((x, y))

                except (ValueError, TypeError, KeyError):
                    continue

            if valid_points:
                valid_points.sort(key=lambda point: point[0])

                plt.plot(
                    [point[0] for point in valid_points],
                    [point[1] for point in valid_points],
                    marker="o",
                    label=label,
                )

                plotted = True

        if plotted:
            plt.xlabel("Training steps")
            plt.ylabel("Loss")
            plt.title("Training-Budget Scaling")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            output_path = PLOTS_DIR / "training_budget_scaling.png"
            plt.savefig(output_path, dpi=150)
            print(f"Saved: {output_path}")

        plt.close()

print("\nExperiment plotting finished.")
print(f"Check the plots folder: {PLOTS_DIR}")