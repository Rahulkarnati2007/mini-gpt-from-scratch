import math
from pathlib import Path

import torch

from dataset import (
    vocab_size,
    BATCH_SIZE,
    BLOCK_SIZE,
    train_data,
    val_data,
    get_batch
)

from model import GPTLanguageModel


# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = BASE_DIR / "checkpoints"

BEST_MODEL_PATH = CHECKPOINT_DIR / "model.pth"
FINAL_MODEL_PATH = CHECKPOINT_DIR / "final_model.pth"
TRAINING_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "training_checkpoint.pth"
)

CHECKPOINT_DIR.mkdir(
    exist_ok=True
)


# ==================================================
# Configuration
# ==================================================

n_embd = 32
num_heads = 4
n_layer = 4

# Training
max_iters = 3000

learning_rate = 3e-4
min_learning_rate = 3e-5

# Warmup
warmup_iters = 200

# Evaluation
eval_interval = 100
eval_iters = 100

# Gradient clipping
max_grad_norm = 1.0

# Resume training?
resume_training = False


# ==================================================
# Device
# ==================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ==================================================
# Reproducibility
# ==================================================

torch.manual_seed(1337)


# ==================================================
# Create model
# ==================================================

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=BLOCK_SIZE,
    num_heads=num_heads,
    n_layer=n_layer
)

model = model.to(device)


# ==================================================
# Parameter count
# ==================================================

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("Vocabulary size:", vocab_size)
print("Model parameters:", f"{num_parameters:,}")
print("Training tokens:", f"{len(train_data):,}")
print("Validation tokens:", f"{len(val_data):,}")
print("Batch size:", BATCH_SIZE)
print("Block size:", BLOCK_SIZE)


# ==================================================
# Optimizer
# ==================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.01
)


# ==================================================
# Learning-rate schedule
# ==================================================

def get_learning_rate(iteration):
    """
    Linear warmup followed by cosine decay.
    """

    # Warmup
    if iteration < warmup_iters:

        return learning_rate * (
            (iteration + 1)
            / warmup_iters
        )

    # Cosine decay
    decay_ratio = (
        (iteration - warmup_iters)
        / (max_iters - warmup_iters)
    )

    decay_ratio = min(
        max(decay_ratio, 0.0),
        1.0
    )

    cosine_coeff = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * decay_ratio
            )
        )
    )

    return (
        min_learning_rate
        + cosine_coeff
        * (
            learning_rate
            - min_learning_rate
        )
    )


# ==================================================
# Validation loss
# ==================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    losses = {}

    # Make evaluation repeatable
    evaluation_generator = torch.Generator()

    evaluation_generator.manual_seed(12345)

    for split, split_data in [
        ("train", train_data),
        ("val", val_data)
    ]:

        loss_values = torch.zeros(
            eval_iters
        )

        for k in range(eval_iters):

            if len(split_data) <= BLOCK_SIZE:
                raise ValueError(
                    "Dataset is too small for "
                    "the selected block size."
                )

            max_start = (
                len(split_data)
                - BLOCK_SIZE
                - 1
            )

            positions = torch.randint(
                0,
                max_start + 1,
                (BATCH_SIZE,),
                generator=evaluation_generator
            )

            X = torch.stack(
                [
                    split_data[
                        pos:pos + BLOCK_SIZE
                    ]
                    for pos in positions
                ]
            )

            Y = torch.stack(
                [
                    split_data[
                        pos + 1:
                        pos + BLOCK_SIZE + 1
                    ]
                    for pos in positions
                ]
            )

            X = X.to(device)
            Y = Y.to(device)

            _, loss = model(
                X,
                Y
            )

            loss_values[k] = loss.item()

        losses[split] = (
            loss_values.mean().item()
        )

    model.train()

    return losses


# ==================================================
# Resume training
# ==================================================

start_iteration = 0
best_val_loss = float("inf")

if (
    resume_training
    and TRAINING_CHECKPOINT_PATH.exists()
):

    print("\nLoading training checkpoint...")

    checkpoint = torch.load(
        TRAINING_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_iteration = (
        checkpoint["iteration"] + 1
    )

    best_val_loss = checkpoint[
        "best_val_loss"
    ]

    print(
        "Resuming from iteration:",
        start_iteration
    )

    print(
        "Best validation loss:",
        best_val_loss
    )


# ==================================================
# Training
# ==================================================

print("\nStarting training...\n")


for iteration in range(
    start_iteration,
    max_iters
):

    # --------------------------------------------------
    # Update learning rate
    # --------------------------------------------------

    current_learning_rate = (
        get_learning_rate(iteration)
    )

    for param_group in optimizer.param_groups:

        param_group["lr"] = (
            current_learning_rate
        )


    # --------------------------------------------------
    # Get training batch
    # --------------------------------------------------

    X, Y = get_batch(
        train_data,
        BATCH_SIZE,
        BLOCK_SIZE
    )

    X = X.to(device)
    Y = Y.to(device)


    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    _, loss = model(
        X,
        Y
    )


    # --------------------------------------------------
    # Clear old gradients
    # --------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )


    # --------------------------------------------------
    # Backpropagation
    # --------------------------------------------------

    loss.backward()


    # --------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------

    grad_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm
        )
    )


    # --------------------------------------------------
    # Update parameters
    # --------------------------------------------------

    optimizer.step()


    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    if (
        iteration == start_iteration
        or iteration % eval_interval == 0
        or iteration == max_iters - 1
    ):

        losses = estimate_loss()

        train_loss = losses["train"]
        val_loss = losses["val"]

        print(
            f"Step {iteration:4d} | "
            f"LR {current_learning_rate:.6f} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"Grad {grad_norm:.4f}"
        )


        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )

            print(
                "  → Saved new best model"
            )


        # --------------------------------------------------
        # Save resumable training checkpoint
        # --------------------------------------------------

        torch.save(
            {
                "iteration": iteration,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "best_val_loss":
                    best_val_loss,
            },
            TRAINING_CHECKPOINT_PATH
        )


# ==================================================
# Save final model
# ==================================================

torch.save(
    model.state_dict(),
    FINAL_MODEL_PATH
)


print("\nTraining complete!")

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    "Best model:",
    BEST_MODEL_PATH
)

print(
    "Final model:",
    FINAL_MODEL_PATH
)

print(
    "Training checkpoint:",
    TRAINING_CHECKPOINT_PATH
)