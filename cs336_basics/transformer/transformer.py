import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from cs336_basics.transformer import CausalMultiHeadedSelfAttention, RMSNorm, SwiGLU, Embedding, Linear

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int | None = None,
            theta: int | None = None,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

        self.attn = CausalMultiHeadedSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )

    
    def forward(self, x: Float[Tensor, "... seq d_model"], token_positions: Float[Tensor, "... seq"] | None = None) -> Float[Tensor, "... seq d_model"]:

        # There is no in-place operations on x
        # Nor there is a shape mismatch after the operations
        # So no need to use x.clone() for storing residual
        residual = x
        y = residual + self.attn(self.ln1(x), token_positions)

        residual = y
        out = residual + self.ffn(self.ln2(y))

        return out
    

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            theta: int | None = None,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    device=device,
                    dtype=dtype
                ) for _ in range(num_layers)
            ]
        )

    def forward(
            self,
            x: Float[Tensor, "... seq"],
            token_positions: Float[Tensor, "... seq"] | None = None
    ) -> Float[Tensor, "... seq vocab"]:
        
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x, token_positions)

        logits = self.lm_head(self.ln_final(x))

        return logits
