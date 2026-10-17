import math
from pathlib import Path

import torch

from dataset import (
    train_data,
    val_data,
    vocab_size,
    BATCH_SIZE,
    BLOCK_SIZE,
)
from model import GPTLanguageModel


# ============================================================
# CONFIGURATION
# ============================================================

N_EMBD = 32
NUM_HEADS = 4
N_LAYER = 4
DROPOUT = 0.0

MAX_STEPS = 4000

MAX_LR = 1.6e-3
MIN_LR = 3e-5
WARMUP_STEPS = 200

EVAL_INTERVAL = 500
EVAL_ITERS = 50

WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

BEST_PATH = CHECKPOINT_DIR / "model.pth"
FINAL_PATH = CHECKPOINT_DIR / "final_model.pth"
TRAINING_PATH = CHECKPOINT_DIR / "final_training_checkpoint.pth"


# ============================================================
# BATCH GENERATION
# ============================================================

def get_batch(data, batch_size, block_size, generator):

    max_start = len(data) - block_size - 1

    ix = torch.randint(
        0,
        max_start,
        (batch_size,),
        generator=generator,
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
            generator,
        )
        for _ in range(EVAL_ITERS)
    ]


# ============================================================
# LEARNING RATE SCHEDULE
# ============================================================

def get_learning_rate(step):

    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS

    if step >= MAX_STEPS:
        return MIN_LR

    progress = (
        (step - WARMUP_STEPS)
        / (MAX_STEPS - WARMUP_STEPS)
    )

    coefficient = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )

    return MIN_LR + coefficient * (MAX_LR - MIN_LR)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, batches):

    model.eval()
    total_loss = 0.0

    for x, y in batches:

        _, loss = model(x, y)

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                "Non-finite loss encountered during evaluation."
            )

        total_loss += loss.item()

    model.train()

    return total_loss / len(batches)


# ============================================================
# INITIALIZE MODEL
# ============================================================

torch.manual_seed(42)

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=N_EMBD,
    block_size=BLOCK_SIZE,
    num_heads=NUM_HEADS,
    n_layer=N_LAYER,
    dropout=DROPOUT,
)

parameter_count = sum(
    p.numel() for p in model.parameters()
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=MAX_LR,
    weight_decay=WEIGHT_DECAY,
)

training_generator = torch.Generator()
training_generator.manual_seed(42)

fixed_train_batches = create_fixed_batches(
    train_data,
    seed=54321,
)

fixed_val_batches = create_fixed_batches(
    val_data,
    seed=12345,
)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

best_val_loss = float("inf")
history = []

print("\n" + "=" * 65)
print("FINAL MINI-GPT TRAINING")
print("=" * 65)
print(f"Vocabulary size : {vocab_size}")
print(f"Parameters      : {parameter_count:,}")
print(f"Embedding dim   : {N_EMBD}")
print(f"Layers          : {N_LAYER}")
print(f"Heads           : {NUM_HEADS}")
print(f"Context length  : {BLOCK_SIZE}")
print(f"Dropout         : {DROPOUT}")
print(f"Training steps  : {MAX_STEPS}")
print(f"Maximum LR      : {MAX_LR}")


# ============================================================
# TRAINING LOOP
# ============================================================

for step in range(MAX_STEPS):

    learning_rate = get_learning_rate(step)

    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    x, y = get_batch(
        train_data,
        BATCH_SIZE,
        BLOCK_SIZE,
        training_generator,
    )

    optimizer.zero_grad(set_to_none=True)

    _, loss = model(x, y)

    if not torch.isfinite(loss).item():
        raise RuntimeError(
            f"Training loss became non-finite at step {step + 1}."
        )

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        MAX_GRAD_NORM,
    )

    if not torch.isfinite(grad_norm).item():
        raise RuntimeError(
            f"Gradient norm became non-finite at step {step + 1}."
        )

    optimizer.step()

    steps_completed = step + 1

    # --------------------------------------------------------
    # EVALUATION AND BEST CHECKPOINT
    # --------------------------------------------------------

    if (
        steps_completed % EVAL_INTERVAL == 0
        or steps_completed == MAX_STEPS
    ):

        train_loss = evaluate(
            model,
            fixed_train_batches,
        )

        val_loss = evaluate(
            model,
            fixed_val_batches,
        )

        perplexity = math.exp(val_loss)

        print(
            f"Step {steps_completed:4d} | "
            f"LR {learning_rate:.6f} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"PPL {perplexity:.2f}"
        )

        history.append({
            "step": steps_completed,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "perplexity": perplexity,
        })

        # Save only when validation performance improves.
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                BEST_PATH,
            )

            print(
                f"  Best checkpoint saved "
                f"(val loss: {best_val_loss:.4f})"
            )


# ============================================================
# SAVE FINAL MODEL AND TRAINING METADATA
# ============================================================

torch.save(
    model.state_dict(),
    FINAL_PATH,
)

torch.save(
    {
        "step": MAX_STEPS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_loss": best_val_loss,
        "config": {
            "vocab_size": vocab_size,
            "n_embd": N_EMBD,
            "num_heads": NUM_HEADS,
            "n_layer": N_LAYER,
            "block_size": BLOCK_SIZE,
            "dropout": DROPOUT,
            "max_lr": MAX_LR,
            "min_lr": MIN_LR,
            "max_steps": MAX_STEPS,
        },
        "history": history,
    },
    TRAINING_PATH,
)

print("\n" + "=" * 65)
print("FINAL TRAINING COMPLETED")
print("=" * 65)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best checkpoint     : {BEST_PATH}")
print(f"Final checkpoint    : {FINAL_PATH}")
print(f"Training metadata   : {TRAINING_PATH}")