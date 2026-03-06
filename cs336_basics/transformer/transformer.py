import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from cs336_basics.transformer import CausalMultiHeadedSelfAttention, RMSNorm, SwiGLU

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int | None = None,
            theta: int | None = None,
            token_positions: Float[Tensor, "... seq"] | None = None,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.rmsnorm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.rmsnorm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

        self.causal_mhsa = CausalMultiHeadedSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            token_positions=token_positions,
            device=device,
            dtype=dtype
        )

    
    def forward(self, x: Float[Tensor, "... seq d_model"]) -> Float[Tensor, "... seq d_model"]:

        # There is no in-place operations on x
        # Nor there is a shape mismatch after the operations
        # So no need to use x.clone() for storing residual
        residual = x
        y = residual + self.causal_mhsa(self.rmsnorm1(x))

        residual = y
        out = residual + self.swiglu(self.rmsnorm2(y))

        return out