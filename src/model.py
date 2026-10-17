# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # ==================================================
# # Single Attention Head
# # ==================================================

# class Head(nn.Module):

#     def __init__(self, n_embd, head_size, block_size):
#         super().__init__()

#         self.query = nn.Linear(
#             n_embd,
#             head_size,
#             bias=False
#         )

#         self.key = nn.Linear(
#             n_embd,
#             head_size,
#             bias=False
#         )

#         self.value = nn.Linear(
#             n_embd,
#             head_size,
#             bias=False
#         )

#         # Causal mask
#         self.register_buffer(
#             "tril",
#             torch.tril(
#                 torch.ones(block_size, block_size)
#             )
#         )

#     def forward(self, x):

#         B, T, C = x.shape

#         # Create Q, K, V
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)

#         # Attention scores
#         scores = Q @ K.transpose(-2, -1)

#         # Scale
#         scores = scores / (Q.size(-1) ** 0.5)

#         # Causal masking
#         scores = scores.masked_fill(
#             self.tril[:T, :T] == 0,
#             float("-inf")
#         )

#         # Softmax
#         weights = F.softmax(
#             scores,
#             dim=-1
#         )

#         # Weighted sum of values
#         out = weights @ V

#         return out


# # ==================================================
# # Multi-Head Attention
# # ==================================================

# class MultiHeadAttention(nn.Module):

#     def __init__(
#         self,
#         n_embd,
#         num_heads,
#         block_size
#     ):
#         super().__init__()

#         head_size = n_embd // num_heads

#         self.heads = nn.ModuleList(
#             [
#                 Head(
#                     n_embd,
#                     head_size,
#                     block_size
#                 )
#                 for _ in range(num_heads)
#             ]
#         )

#         # Mix information from all heads
#         self.proj = nn.Linear(
#             n_embd,
#             n_embd
#         )

#     def forward(self, x):

#         head_outputs = [
#             head(x)
#             for head in self.heads
#         ]

#         # Concatenate all heads
#         out = torch.cat(
#             head_outputs,
#             dim=-1
#         )

#         # Projection
#         out = self.proj(out)

#         return out


# # ==================================================
# # Feed-Forward Network
# # ==================================================

# class FeedForward(nn.Module):

#     def __init__(self, n_embd):
#         super().__init__()

#         self.network = nn.Sequential(

#             nn.Linear(
#                 n_embd,
#                 4 * n_embd
#             ),

#             nn.GELU(),

#             nn.Linear(
#                 4 * n_embd,
#                 n_embd
#             )
#         )

#     def forward(self, x):

#         return self.network(x)


# # ==================================================
# # Transformer Block
# # ==================================================

# class TransformerBlock(nn.Module):

#     def __init__(
#         self,
#         n_embd,
#         num_heads,
#         block_size
#     ):
#         super().__init__()

#         self.attention = MultiHeadAttention(
#             n_embd,
#             num_heads,
#             block_size
#         )

#         self.feed_forward = FeedForward(
#             n_embd
#         )

#         self.ln1 = nn.LayerNorm(
#             n_embd
#         )

#         self.ln2 = nn.LayerNorm(
#             n_embd
#         )

#     def forward(self, x):

#         # Pre-LN attention + residual
#         x = x + self.attention(
#             self.ln1(x)
#         )

#         # Pre-LN FFN + residual
#         x = x + self.feed_forward(
#             self.ln2(x)
#         )

#         return x


# # ==================================================
# # GPT Language Model
# # ==================================================

# class GPTLanguageModel(nn.Module):

#     def __init__(
#         self,
#         vocab_size,
#         n_embd,
#         block_size,
#         num_heads,
#         n_layer
#     ):
#         super().__init__()

#         # Token embeddings
#         self.token_embedding_table = nn.Embedding(
#             vocab_size,
#             n_embd
#         )

#         # Position embeddings
#         self.position_embedding_table = nn.Embedding(
#             block_size,
#             n_embd
#         )

#         # Transformer blocks
#         self.blocks = nn.Sequential(
#             *[
#                 TransformerBlock(
#                     n_embd,
#                     num_heads,
#                     block_size
#                 )
#                 for _ in range(n_layer)
#             ]
#         )

#         # Final LayerNorm
#         self.ln_f = nn.LayerNorm(
#             n_embd
#         )

#         # Language model head
#         self.lm_head = nn.Linear(
#             n_embd,
#             vocab_size
#         )

#     def forward(
#         self,
#         idx,
#         targets=None
#     ):

#         B, T = idx.shape

#         # Token embeddings
#         token_embeddings = self.token_embedding_table(idx)

#         # Position indices
#         positions = torch.arange(
#             T,
#             device=idx.device
#         )

#         # Position embeddings
#         position_embeddings = self.position_embedding_table(
#             positions
#         )

#         # Token + position information
#         x = token_embeddings + position_embeddings

#         # Transformer blocks
#         x = self.blocks(x)

#         # Final normalization
#         x = self.ln_f(x)

#         # Next-token logits
#         logits = self.lm_head(x)

#         loss = None

#         if targets is not None:

#             B, T, C = logits.shape

#             logits_for_loss = logits.view(
#                 B * T,
#                 C
#             )

#             targets_for_loss = targets.view(
#                 B * T
#             )

#             loss = F.cross_entropy(
#                 logits_for_loss,
#                 targets_for_loss
#             )

#         return logits, loss












import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()

        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (Q.size(-1) ** 0.5)

        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)

        # Drop some attention weights during training.
        weights = self.attention_dropout(weights)

        return weights @ V


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_embd,
        num_heads,
        block_size,
        dropout=0.0
    ):
        super().__init__()

        head_size = n_embd // num_heads

        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(n_embd, n_embd)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x):

        head_outputs = [head(x) for head in self.heads]

        out = torch.cat(head_outputs, dim=-1)

        out = self.proj(out)

        return self.residual_dropout(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd,
        num_heads,
        block_size,
        dropout=0.0
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            n_embd,
            num_heads,
            block_size,
            dropout
        )

        self.feed_forward = FeedForward(n_embd, dropout)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        x = x + self.attention(self.ln1(x))

        x = x + self.feed_forward(self.ln2(x))

        return x


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd,
        block_size,
        num_heads,
        n_layer,
        dropout=0.0
    ):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                n_embd,
                num_heads,
                block_size,
                dropout
            )
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)

        positions = torch.arange(T, device=idx.device)

        position_embeddings = self.position_embedding_table(
            positions
        )

        x = token_embeddings + position_embeddings

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits_for_loss = logits.reshape(B * T, C)
            targets_for_loss = targets.reshape(B * T)

            loss = F.cross_entropy(
                logits_for_loss,
                targets_for_loss
            )

        return logits, loss