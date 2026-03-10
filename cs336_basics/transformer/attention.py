import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum
from einx import rearrange, dot
from jaxtyping import Float
from cs336_basics.transformer import softmax, Linear, RotaryPositionalEmbedding

def scaled_dot_product_attention(
        Q: Float[Tensor, "... seq d_k"],
        K: Float[Tensor, "... seq d_k"],
        V: Float[Tensor, "... seq d_v"],
        mask: Float[Tensor, "seq seq"] | None = None
) -> Float[Tensor, "... seq d_v"]:
    d_k = Q.shape[-1]
    qkT = einsum(Q, K, "... seq1 d_k, ... seq2 d_k -> ... seq1 seq2") / (d_k**0.5)

    if mask is not None:
        qkT = qkT + torch.where(mask, 0, -float('inf'))

    S = softmax(qkT, dimension=-1)

    return einsum(S, V, "... seq1 seq2, ... seq2 d_v -> ... seq1 d_v")


class CausalMultiHeadedSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            max_seq_len: int | None = None,
            theta: int | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()

        # here d_k = d_v = d_model / num_heads
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)


    def forward(
            self,
            x: Float[Tensor, "... seq d_model"],
            token_positions: Float[Tensor, "... seq"] | None = None
    ) -> Float[Tensor, "... seq d_model"]:

        # each weight has a shape of d_out x d_in = d_model x d_model
        # Hence combined W shape: 3*d_model x d_model
        W = torch.concat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight])
        q, k, v = dot(
            "... seq d_in, (num_concats d_out) d_in -> ... seq (num_concats d_out)", x, W, num_concats=3
        ).tensor_split(3, -1)

        # one of the culprit: location of h in input
        Q = rearrange("... seq (h d_k) -> ... h seq d_k", q, h=self.num_heads)
        K = rearrange("... seq (h d_k) -> ... h seq d_k", k, h=self.num_heads)
        V = rearrange("... seq (h d_k) -> ... h seq d_k", v, h=self.num_heads)

        # have to expand token_positions for all the heads to process (batch, seq_len) -> (batch, head, seq_len)
        if token_positions is not None:
            token_positions_expanded = rearrange("... seq -> ... h seq", token_positions, h=self.num_heads)

            Q = self.rope.forward(Q, token_positions_expanded)
            K = self.rope.forward(K, token_positions_expanded)

        seq_len = x.shape[-2]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        output = scaled_dot_product_attention(Q, K, V, causal_mask)

        # Combine values from all the heads (i.e. just reshape)
        O_concat = rearrange("... h seq d_k -> ... seq (h d_k)", output)

        return self.output_proj(O_concat)

