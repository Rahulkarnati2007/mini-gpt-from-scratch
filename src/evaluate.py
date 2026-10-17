import math
from pathlib import Path

import torch

from dataset import (
    vocab_size,
    BATCH_SIZE,
    BLOCK_SIZE,
    val_data
)

from model import GPTLanguageModel


# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = BASE_DIR / "checkpoints"

BEST_MODEL_PATH = CHECKPOINT_DIR / "model.pth"
FINAL_MODEL_PATH = CHECKPOINT_DIR / "final_model.pth"


# ==================================================
# Model configuration
# ==================================================

n_embd = 32
num_heads = 4
n_layer = 4

EVAL_ITERS = 100

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================
# Create model
# ==================================================

def create_model():

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        block_size=BLOCK_SIZE,
        num_heads=num_heads,
        n_layer=n_layer
    )

    return model.to(device)


# ==================================================
# Create fixed validation batches
# ==================================================

def create_fixed_batches():

    generator = torch.Generator()
    generator.manual_seed(12345)

    batches = []

    max_start = (
        len(val_data)
        - BLOCK_SIZE
        - 1
    )

    for _ in range(EVAL_ITERS):

        positions = torch.randint(
            0,
            max_start + 1,
            (BATCH_SIZE,),
            generator=generator
        )

        X = torch.stack(
            [
                val_data[
                    pos:pos + BLOCK_SIZE
                ]
                for pos in positions
            ]
        )

        Y = torch.stack(
            [
                val_data[
                    pos + 1:
                    pos + BLOCK_SIZE + 1
                ]
                for pos in positions
            ]
        )

        batches.append((X, Y))

    return batches


# ==================================================
# Evaluate one checkpoint
# ==================================================

@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path,
    batches
):

    model = create_model()

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    total_loss = 0.0

    for X, Y in batches:

        X = X.to(device)
        Y = Y.to(device)

        _, loss = model(X, Y)

        total_loss += loss.item()

    mean_loss = (
        total_loss / len(batches)
    )

    perplexity = math.exp(
        mean_loss
    )

    return mean_loss, perplexity


# ==================================================
# Create fixed evaluation set
# ==================================================

print("Creating fixed validation batches...")

fixed_batches = create_fixed_batches()


# ==================================================
# Evaluate checkpoints
# ==================================================

print("\n" + "=" * 60)
print("MiniGPT Fixed Evaluation")
print("=" * 60)

results = []


for name, path in [
    ("Best checkpoint", BEST_MODEL_PATH),
    ("Final checkpoint", FINAL_MODEL_PATH)
]:

    if not path.exists():

        print(
            f"\nSkipping {name}:"
            f" {path} not found."
        )

        continue

    loss, perplexity = evaluate_checkpoint(
        path,
        fixed_batches
    )

    results.append(
        (name, loss, perplexity)
    )

    print(
        f"\n{name}"
    )

    print(
        f"Validation loss : {loss:.4f}"
    )

    print(
        f"Perplexity      : {perplexity:.2f}"
    )


# ==================================================
# Compare
# ==================================================

if len(results) == 2:

    best_name, best_loss, best_ppl = results[0]
    final_name, final_loss, final_ppl = results[1]

    print("\n" + "-" * 60)

    if best_loss < final_loss:

        print(
            "Best checkpoint performs better "
            "on the fixed validation set."
        )

    elif final_loss < best_loss:

        print(
            "Final checkpoint performs better "
            "on the fixed validation set."
        )

    else:

        print(
            "Both checkpoints have identical "
            "validation loss."
        )

print("=" * 60)