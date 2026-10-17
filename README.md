
# Mini-GPT: Building a GPT-Style Language Model from Scratch

## Overview

Mini-GPT is an educational language-modeling project built in Python and PyTorch. It implements the key components of a GPT-style Transformer and trains the model on the Tiny Shakespeare text dataset.

The project explores tokenization, causal self-attention, Transformer architecture, autoregressive text generation, and controlled experiments on model training and scaling.

## Final Model Configuration

| Parameter | Value |
|---|---:|
| Dataset | Tiny Shakespeare |
| Tokenizer | Byte Pair Encoding (BPE) |
| Vocabulary size | 512 |
| Model parameters | 87,872 |
| Embedding dimension | 32 |
| Transformer layers | 4 |
| Attention heads | 4 |
| Context length | 128 tokens |
| Dropout | 0.0 |
| Training steps | 4,000 |
| Optimizer | AdamW |
| Maximum learning rate | 0.0016 |

## Architecture

The model contains:

1. Token embeddings and learned positional embeddings.
2. Causal multi-head self-attention.
3. Feed-forward neural networks with GELU activation.
4. Layer normalization and residual connections.
5. A linear output layer for next-token prediction.

Training minimizes cross-entropy loss. The implementation includes learning-rate warmup and cosine decay, gradient clipping, checkpoint saving, and validation.

## Text Generation

The model generates text autoregressively, predicting one token at a time.

Supported sampling controls include:

- Temperature
- Top-k sampling
- Top-p (nucleus) sampling

The trained model generates Shakespeare-like dialogue and character names, although its output is not consistently coherent.

## Final Evaluation

| Metric | Result |
|---|---:|
| Validation loss | 3.4627 |
| Perplexity | 31.90 |

These measurements were obtained using the project's fixed validation evaluation procedure.

## Experiments

The project investigated:

- Transformer layer scaling
- Embedding dimension scaling
- Context length scaling
- Attention-head scaling
- Learning-rate sensitivity
- Extended learning-rate search
- Training-budget scaling
- Dropout regularization

The experiments showed that embedding dimension and learning-rate selection had substantial effects on measured validation performance. Longer training also improved validation loss in the tested run. Changing the number of attention heads had little effect in the small model, and dropout did not improve the final validation result under the tested configuration.

Results from different experiments should be interpreted within their respective training budgets and learning-rate schedules.

## Project Structure

```text
mini-gpt/
├── data/
│   ├── input.txt
│   └── tokenizer.json
├── checkpoints/
│   ├── model.pth
│   ├── final_model.pth
│   └── final_training_checkpoint.pth
├── experiments/
│   ├── plots/
│   └── experiment result CSV files
├── src/
│   ├── dataset.py
│   ├── tokenizer.py
│   ├── model.py
│   ├── train.py
│   ├── final_train.py
│   ├── generate.py
│   ├── evaluate.py
│   ├── experiments.py
│   └── plot_results.py
└── README.md
```

## Environment

The project was developed with Python 3.13, PyTorch, Hugging Face Tokenizers, and Matplotlib in a virtual environment.

For Windows, run commands from the project root using:

```powershell
.venv\Scripts\python.exe
```

Install the required packages in the virtual environment if needed:

```powershell
.venv\Scripts\python.exe -m pip install tokenizers matplotlib
```

The project uses a CPU-compatible PyTorch installation.

## Running the Project

Train the final model:

```powershell
.venv\Scripts\python.exe src/final_train.py
```

Generate text:

```powershell
.venv\Scripts\python.exe src/generate.py
```

Evaluate checkpoints:

```powershell
.venv\Scripts\python.exe src/evaluate.py
```

Generate experiment plots:

```powershell
.venv\Scripts\python.exe src/plot_results.py
```

Training the model again may take considerable time on a CPU. The saved checkpoints can be used for evaluation and generation without repeating training.

## Limitations

- The model is intentionally small and is not expected to match the capabilities of large pretrained language models.
- Training and evaluation results depend on the dataset, random seed, optimization schedule, and experimental budget.
- A lower perplexity does not necessarily guarantee more coherent or factually reliable generated text.
- Architecture and hyperparameter conclusions are specific to the configurations tested.

## Learning Outcomes

This project provided practical experience with Transformer architecture, attention mechanisms, tokenization, next-token prediction, model optimization, validation, text generation, and experimental analysis.
