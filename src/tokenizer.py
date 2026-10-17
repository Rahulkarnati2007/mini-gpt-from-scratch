from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import models
from tokenizers import trainers
from tokenizers import pre_tokenizers
from tokenizers import decoders


# ==================================================
# Configuration
# ==================================================

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "input.txt"
)

TOKENIZER_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "tokenizer.json"
)

VOCAB_SIZE = 512


# ==================================================
# Check dataset
# ==================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )


# ==================================================
# Create BPE tokenizer
# ==================================================

tokenizer = Tokenizer(
    models.BPE(
        unk_token="<unk>"
    )
)


# ==================================================
# Byte-level pre-tokenization
# ==================================================

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False
)


# ==================================================
# Byte-level decoding
# ==================================================

tokenizer.decoder = decoders.ByteLevel()


# ==================================================
# BPE trainer
# ==================================================

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<unk>"]
)


# ==================================================
# Train tokenizer
# ==================================================

print("Training BPE tokenizer...")

tokenizer.train(
    files=[str(DATA_PATH)],
    trainer=trainer
)


# ==================================================
# Save tokenizer
# ==================================================

tokenizer.save(
    str(TOKENIZER_PATH)
)

print("\nTokenizer saved to:")
print(TOKENIZER_PATH)


# ==================================================
# Display vocabulary information
# ==================================================

actual_vocab_size = tokenizer.get_vocab_size()

print("\nTokenizer information:")
print("Requested vocabulary size:", VOCAB_SIZE)
print("Actual vocabulary size   :", actual_vocab_size)


# ==================================================
# Test encoding
# ==================================================

test_text = "hello world"

encoding = tokenizer.encode(
    test_text
)

print("\nTest text:")
print(test_text)

print("\nToken IDs:")
print(encoding.ids)

print("\nTokens:")
print(encoding.tokens)

print("\nDecoded text:")
print(tokenizer.decode(encoding.ids))