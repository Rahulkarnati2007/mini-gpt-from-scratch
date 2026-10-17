import torch

from dataset import (
    tokenizer,
    vocab_size,
    BLOCK_SIZE
)

from model import GPTLanguageModel


# ==================================================
# Configuration
# ==================================================

n_embd = 32
num_heads = 4
n_layer = 4

max_new_tokens = 100

temperature = 0.8
top_k = 20
top_p = 0.9

device = "cuda" if torch.cuda.is_available() else "cpu"


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
# Load trained weights
# ==================================================

model.load_state_dict(
    torch.load(
        "checkpoints/model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()


# ==================================================
# Top-p filtering
# ==================================================

def apply_top_p(logits, top_p):
    """
    Keep the smallest set of tokens whose
    cumulative probability reaches top_p.
    """

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    sorted_probs, sorted_indices = torch.sort(
        probabilities,
        descending=True
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1
    )

    sorted_indices_to_remove = (
        cumulative_probs > top_p
    )

    # Keep the first token that crosses top_p
    sorted_indices_to_remove[..., 1:] = (
        sorted_indices_to_remove[..., :-1].clone()
    )

    sorted_indices_to_remove[..., 0] = False

    # Convert sorted mask back to vocabulary order
    indices_to_remove = torch.zeros_like(
        logits,
        dtype=torch.bool
    )

    indices_to_remove.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove
    )

    logits = logits.masked_fill(
        indices_to_remove,
        float("-inf")
    )

    return logits


# ==================================================
# Text generation
# ==================================================

@torch.no_grad()
def generate(
    idx,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None
):
    """
    Autoregressively generate new token IDs.
    """

    for _ in range(max_new_tokens):

        # Keep only the context the model supports
        idx_cond = idx[:, -BLOCK_SIZE:]

        # Forward pass
        logits, _ = model(idx_cond)

        # Only use logits from the final position
        logits = logits[:, -1, :]

        # --------------------------------------------------
        # Temperature
        # --------------------------------------------------

        if temperature <= 0:
            raise ValueError(
                "Temperature must be greater than 0."
            )

        logits = logits / temperature

        # --------------------------------------------------
        # Top-k filtering
        # --------------------------------------------------

        if top_k is not None:

            k = min(
                top_k,
                logits.size(-1)
            )

            values, _ = torch.topk(
                logits,
                k
            )

            min_value = values[:, [-1]]

            logits = torch.where(
                logits < min_value,
                torch.full_like(
                    logits,
                    float("-inf")
                ),
                logits
            )

        # --------------------------------------------------
        # Top-p filtering
        # --------------------------------------------------

        if top_p is not None:

            if not 0 < top_p <= 1:
                raise ValueError(
                    "top_p must be between 0 and 1."
                )

            logits = apply_top_p(
                logits,
                top_p
            )

        # --------------------------------------------------
        # Convert to probabilities
        # --------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        # --------------------------------------------------
        # Sample next token
        # --------------------------------------------------

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # --------------------------------------------------
        # Append token
        # --------------------------------------------------

        idx = torch.cat(
            (idx, next_token),
            dim=1
        )

    return idx


# ==================================================
# Prompt
# ==================================================

prompt = "ROMEO:"


# ==================================================
# Encode prompt using BPE tokenizer
# ==================================================

encoding = tokenizer.encode(prompt)

context = torch.tensor(
    [encoding.ids],
    dtype=torch.long,
    device=device
)

print("Prompt:", prompt)
print("Prompt token IDs:", encoding.ids)


# ==================================================
# Generate
# ==================================================

generated = generate(
    context,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p
)


# ==================================================
# Decode generated tokens
# ==================================================

generated_ids = generated[0].tolist()

generated_text = tokenizer.decode(
    generated_ids
)


# ==================================================
# Output
# ==================================================

print("\nGeneration settings:")
print("Temperature:", temperature)
print("Top-k:", top_k)
print("Top-p:", top_p)

print("\nGenerated text:")
print(generated_text)