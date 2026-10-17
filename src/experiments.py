import math
import csv
from pathlib import Path

import torch

from dataset import (
    train_data,
    val_data,
    vocab_size,
    BATCH_SIZE,
)
from model import GPTLanguageModel


# ============================================================
# CONFIGURATION
# ============================================================

DROPOUT_RATES = [0.0, 0.1, 0.2]

N_EMBD = 32
NUM_HEADS = 4
N_LAYER = 4
BLOCK_SIZE = 128

MAX_ITERS = 2000

MAX_LR = 1.6e-3
MIN_LR = 3e-5
WARMUP_ITERS = 200

EVAL_INTERVAL = 500
EVAL_ITERS = 50

WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = (
    BASE_DIR / "experiments" / "dropout_scaling.csv"
)


# ============================================================
# BATCH GENERATION
# ============================================================

def get_batch(data, batch_size, block_size, generator):

    max_start = len(data) - block_size - 1

    ix = torch.randint(
        0,
        max_start,
        (batch_size,),
        generator=generator
    )

    x = torch.stack([
        data[i:i + block_size] for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1] for i in ix
    ])

    return x, y


def create_fixed_batches(data, seed):

    generator = torch.Generator()
    generator.manual_seed(seed)

    return [
        get_batch(
            data,
            BATCH_SIZE,
            BLOCK_SIZE,
            generator
        )
        for _ in range(EVAL_ITERS)
    ]


# ============================================================
# LEARNING RATE SCHEDULE
# ============================================================

def get_learning_rate(step):

    if step < WARMUP_ITERS:
        return MAX_LR * (step + 1) / WARMUP_ITERS

    if step >= MAX_ITERS:
        return MIN_LR

    decay_ratio = (
        (step - WARMUP_ITERS)
        / (MAX_ITERS - WARMUP_ITERS)
    )

    coefficient = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )

    return MIN_LR + coefficient * (MAX_LR - MIN_LR)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, fixed_batches):

    model.eval()

    total_loss = 0.0

    for x, y in fixed_batches:

        _, loss = model(x, y)

        if not torch.isfinite(loss).item():
            model.train()
            return float("nan")

        total_loss += loss.item()

    model.train()

    return total_loss / len(fixed_batches)


# ============================================================
# FIXED TRAINING AND VALIDATION DATA
# ============================================================

fixed_train_batches = create_fixed_batches(
    train_data,
    seed=54321
)

fixed_val_batches = create_fixed_batches(
    val_data,
    seed=12345
)

all_results = []
final_results = []


# ============================================================
# RUN EACH DROPOUT CONFIGURATION
# ============================================================

for dropout_rate in DROPOUT_RATES:

    print("\n" + "=" * 65)
    print(f"DROPOUT RATE: {dropout_rate}")
    print("=" * 65)

    # Use the same initial weights for each experiment.
    torch.manual_seed(42)

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        block_size=BLOCK_SIZE,
        num_heads=NUM_HEADS,
        n_layer=N_LAYER,
        dropout=dropout_rate
    )

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Parameters: {parameter_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MAX_LR,
        weight_decay=WEIGHT_DECAY
    )

    # Same training batch positions in every experiment.
    batch_generator = torch.Generator()
    batch_generator.manual_seed(42)

    run_results = []

    # --------------------------------------------------------
    # INITIAL EVALUATION
    # --------------------------------------------------------

    train_loss = evaluate(model, fixed_train_batches)
    val_loss = evaluate(model, fixed_val_batches)

    initial_result = {
        "dropout": dropout_rate,
        "steps": 0,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "perplexity": math.exp(val_loss)
    }

    run_results.append(initial_result)
    all_results.append(initial_result)

    print(
        f"Steps    0 | "
        f"Train {train_loss:.4f} | "
        f"Val {val_loss:.4f} | "
        f"PPL {math.exp(val_loss):.2f}"
    )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for step in range(MAX_ITERS):

        learning_rate = get_learning_rate(step)

        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        x, y = get_batch(
            train_data,
            BATCH_SIZE,
            BLOCK_SIZE,
            batch_generator
        )

        optimizer.zero_grad(set_to_none=True)

        _, loss = model(x, y)

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite training loss at step {step + 1}, "
                f"dropout={dropout_rate}"
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM
        )

        if not torch.isfinite(grad_norm).item():
            raise RuntimeError(
                f"Non-finite gradient at step {step + 1}, "
                f"dropout={dropout_rate}"
            )

        optimizer.step()

        steps_completed = step + 1

        if (
            steps_completed % EVAL_INTERVAL == 0
            or steps_completed == MAX_ITERS
        ):

            train_loss = evaluate(
                model,
                fixed_train_batches
            )

            val_loss = evaluate(
                model,
                fixed_val_batches
            )

            if not math.isfinite(val_loss):
                raise RuntimeError(
                    f"Non-finite validation loss at step "
                    f"{steps_completed}, dropout={dropout_rate}"
                )

            perplexity = math.exp(val_loss)

            result = {
                "dropout": dropout_rate,
                "steps": steps_completed,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "perplexity": perplexity
            }

            run_results.append(result)
            all_results.append(result)

            print(
                f"Steps {steps_completed:4d} | "
                f"LR {learning_rate:.6f} | "
                f"Train {train_loss:.4f} | "
                f"Val {val_loss:.4f} | "
                f"PPL {perplexity:.2f}"
            )

    # --------------------------------------------------------
    # RECORD FINAL RESULT
    # --------------------------------------------------------

    final_result = run_results[-1]

    final_results.append({
        "dropout": dropout_rate,
        "parameters": parameter_count,
        "train_loss": final_result["train_loss"],
        "validation_loss": final_result["validation_loss"],
        "perplexity": final_result["perplexity"]
    })


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(RESULTS_PATH, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "dropout",
            "steps",
            "train_loss",
            "validation_loss",
            "perplexity"
        ]
    )

    writer.writeheader()
    writer.writerows(all_results)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DROPOUT REGULARIZATION RESULTS")
print("=" * 70)

for result in final_results:

    print(
        f"Dropout {result['dropout']:.1f} | "
        f"Train {result['train_loss']:.4f} | "
        f"Val {result['validation_loss']:.4f} | "
        f"PPL {result['perplexity']:.2f}"
    )

best_result = min(
    final_results,
    key=lambda result: result["validation_loss"]
)

print("\nBest final validation result:")
print(f"Dropout: {best_result['dropout']:.1f}")
print(f"Validation loss: {best_result['validation_loss']:.4f}")
print(f"Perplexity: {best_result['perplexity']:.2f}")

print(f"\nResults saved to: {RESULTS_PATH}")