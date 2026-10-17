# import torch
# from pathlib import Path


# # ==================================================
# # Configuration
# # ==================================================

# BATCH_SIZE = 32
# BLOCK_SIZE = 128

# TRAIN_RATIO = 0.90

# DATA_PATH = (
#     Path(__file__).resolve().parent.parent
#     / "data"
#     / "input.txt"
# )


# # ==================================================
# # Load dataset
# # ==================================================

# if not DATA_PATH.exists():
#     raise FileNotFoundError(
#         f"Dataset not found at: {DATA_PATH}\n"
#         "Make sure data/input.txt exists."
#     )

# text = DATA_PATH.read_text(
#     encoding="utf-8"
# )

# print("Dataset loaded.")
# print("Characters:", len(text))


# # ==================================================
# # Create vocabulary
# # ==================================================

# chars = sorted(
#     list(set(text))
# )

# vocab_size = len(chars)

# # Character -> integer
# stoi = {
#     ch: i
#     for i, ch in enumerate(chars)
# }

# # Integer -> character
# itos = {
#     i: ch
#     for ch, i in stoi.items()
# }


# print("Vocabulary size:", vocab_size)


# # ==================================================
# # Encode entire dataset
# # ==================================================

# data = torch.tensor(
#     [stoi[ch] for ch in text],
#     dtype=torch.long
# )


# # ==================================================
# # Train / validation split
# # ==================================================

# split_index = int(
#     TRAIN_RATIO * len(data)
# )

# train_data = data[:split_index]

# val_data = data[split_index:]


# print("\nDataset split:")
# print("Training tokens  :", len(train_data))
# print("Validation tokens:", len(val_data))


# # ==================================================
# # Batch generator
# # ==================================================

# def get_batch(
#     data,
#     batch_size=BATCH_SIZE,
#     block_size=BLOCK_SIZE
# ):
#     """
#     Create a random batch of input/target sequences.

#     x contains input tokens.
#     y contains the same sequence shifted by one token.
#     """

#     if len(data) <= block_size:
#         raise ValueError(
#             "Dataset is too small for the selected block_size."
#         )

#     # Valid starting positions
#     max_start = len(data) - block_size - 1

#     ix = torch.randint(
#         0,
#         max_start + 1,
#         (batch_size,)
#     )

#     # Input sequences
#     x = torch.stack(
#         [
#             data[i:i + block_size]
#             for i in ix
#         ]
#     )

#     # Target sequences
#     y = torch.stack(
#         [
#             data[i + 1:i + block_size + 1]
#             for i in ix
#         ]
#     )

#     return x, y

# if __name__ == "__main__":

#     x, y = get_batch(
#         train_data,
#         BATCH_SIZE,
#         BLOCK_SIZE
#     )

#     print("\nBatch test:")
#     print("x shape:", x.shape)
#     print("y shape:", y.shape)

#     print("\nFirst input:")
#     print(
#         "".join(
#             itos[i.item()]
#             for i in x[0]
#         )
#     )

#     print("\nFirst target:")
#     print(
#         "".join(
#             itos[i.item()]
#             for i in y[0]
#         )
#     )
































import torch
from pathlib import Path
from tokenizers import Tokenizer


# ==================================================
# Configuration
# ==================================================

BATCH_SIZE = 32
BLOCK_SIZE = 128

TRAIN_RATIO = 0.90

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "input.txt"
TOKENIZER_PATH = BASE_DIR / "data" / "tokenizer.json"


# ==================================================
# Check files
# ==================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(
        f"Tokenizer not found:\n{TOKENIZER_PATH}\n"
        "Run tokenizer.py first."
    )


# ==================================================
# Load tokenizer
# ==================================================

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)

vocab_size = tokenizer.get_vocab_size()

print("Tokenizer loaded.")
print("Vocabulary size:", vocab_size)


# ==================================================
# Load text
# ==================================================

text = DATA_PATH.read_text(
    encoding="utf-8"
)

print("Characters in dataset:", len(text))


# ==================================================
# Encode entire dataset
# ==================================================

encoding = tokenizer.encode(text)

data = torch.tensor(
    encoding.ids,
    dtype=torch.long
)

print("Tokens in dataset:", len(data))


# ==================================================
# Train / validation split
# ==================================================

split_index = int(
    TRAIN_RATIO * len(data)
)

train_data = data[:split_index]

val_data = data[split_index:]

print("\nDataset split:")
print("Training tokens  :", len(train_data))
print("Validation tokens:", len(val_data))


# ==================================================
# Batch generator
# ==================================================

def get_batch(
    data,
    batch_size=BATCH_SIZE,
    block_size=BLOCK_SIZE
):
    """
    Create a random batch for next-token prediction.

    x:
        Input token sequences.

    y:
        Same sequences shifted by one token.
    """

    if len(data) <= block_size:
        raise ValueError(
            "Dataset is too small for the selected block_size."
        )

    # Highest valid starting position
    max_start = len(data) - block_size - 1

    ix = torch.randint(
        0,
        max_start + 1,
        (batch_size,)
    )

    # Input sequences
    x = torch.stack(
        [
            data[i:i + block_size]
            for i in ix
        ]
    )

    # Target sequences
    y = torch.stack(
        [
            data[i + 1:i + block_size + 1]
            for i in ix
        ]
    )

    return x, y


# ==================================================
# Decode helper
# ==================================================

def decode(token_ids):
    """
    Convert token IDs back into text.
    """

    return tokenizer.decode(
        token_ids.tolist()
    )


# ==================================================
# Test when running this file directly
# ==================================================

if __name__ == "__main__":

    x, y = get_batch(
        train_data,
        BATCH_SIZE,
        BLOCK_SIZE
    )

    print("\nBatch test:")
    print("x shape:", x.shape)
    print("y shape:", y.shape)

    print("\nFirst input:")
    print(
        decode(x[0])
    )

    print("\nFirst target:")
    print(
        decode(y[0])
    )