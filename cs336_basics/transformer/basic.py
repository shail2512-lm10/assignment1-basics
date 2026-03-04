import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from einops import einsum
from einx import elementwise, reduce
from jaxtyping import Float, Int


class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device:torch.device | str | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()

        # Define the shape of the Weight Tensor
        self.weight: Float[Tensor, "d_out d_in"] = nn.Parameter(
            torch.empty(size=(out_features, in_features), dtype=dtype, device=device)
        )

        # Initialize the Weight with Truncated Normal values
        sigma = 2/(in_features + out_features) ** 0.5
        nn.init.trunc_normal_(
                self.weight, mean=0, std=sigma, a=-3*sigma, b=3*sigma
            )

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """Forward pass for Linear Layer

        Args:
            x: Input tensor of shape (..., d_in)

        Returns:
            Output tensor of shape (..., d_out)

        FLOPs:
            2 * batch_size * seq_len * d_in * d_out
        """
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    

class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.weight: Float[Tensor, "vocab_size d_model"] = nn.Parameter(
            torch.empty(size=(num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        nn.init.trunc_normal_(
            self.weight, mean=0, std=1, a=-3, b=3
        )

    def forward(self, token_ids: Int[LongTensor, "... seq"]) -> Float[Tensor, "... seq d_model"]:
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None 
    ):
        super().__init__()

        self.eps = eps

        self.gain: Float[Tensor, " d_model"] = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: Float[Tensor, "... seq d_model"]) -> Float[Tensor, "... seq d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        def custom_squared_sum(z: Tensor, axis) -> Tensor:
            return torch.sqrt(torch.mean(z**2, dim=axis) + self.eps)

        rms = reduce("... seq d_model -> ... seq", x, op=custom_squared_sum)

        result = elementwise("... seq d_model, ... seq, d_model -> ... seq d_model", x, 1/rms, self.gain, op="multiply")

        return result.to(in_dtype)
    

class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        
    
    def forward(self, x: Float[Tensor, "... seq d_model"]) -> Float[Tensor, "... seq d_model"]:
        up_proj1 = self.w1(x)
        silu = up_proj1 * torch.sigmoid(up_proj1)  # elementwise

        up_proj2 = self.w3(x)

        swiglu = elementwise("... seq dff, ... seq dff", silu, up_proj2, op="multiply")
        # glu = silu * up_proj2

        return self.w2(swiglu)  # down projection

