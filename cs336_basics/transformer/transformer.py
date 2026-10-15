import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
import einx
from cs336_basics.transformer import CausalMultiHeadedSelfAttention, RMSNorm, SwiGLU, Embedding, Linear, softmax

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

        # The model may receive float32 activations even when the module parameters are
        # stored in BF16/FP16. Cast the residual stream to the module dtype before any
        # attention or feed-forward matmuls so the residual adds remain type-consistent.
        x = x.to(self.ln1.weight.dtype)

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
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.context_length = context_length
        
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

    @torch.inference_mode()
    def generate(
            self,
            input_ids: Float[Tensor, "... seq"],
            max_tokens: int,
            eos_token_id: int,
            temperature: float | None = None,
            top_p: float | None = None
    ) -> list:
        
        batch_size = input_ids.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        for _ in range(max_tokens):
            if input_ids.shape[-1] > self.context_length:
                input_ids_cond = input_ids[..., -self.context_length:]
            else:
                input_ids_cond = input_ids

            token_positions = torch.arange(input_ids_cond.shape[-1], device=input_ids.device)
            if len(input_ids_cond.shape) > 2:
                token_positions = einx.id("seq -> batch seq", token_positions, batch=input_ids_cond.shape[0])

            logits: Float[Tensor, "... seq vocab"] = self.forward(input_ids_cond, token_positions)

            next_token_logits: Float[Tensor, "... vocab"] = logits[..., -1, :]

            if temperature == 0:
                next_token_id = torch.argmax(next_token_logits, dim=-1)
            else:
                next_token_logits = (next_token_logits / temperature) if temperature else next_token_logits

                probs = softmax(next_token_logits, dimension=-1)

                if top_p is not None and 0 < top_p < 1:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False

                    sorted_probs[sorted_indices_to_remove] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                    sampled_sorted_idx: Float[Tensor, " batch"] = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)

                    batch_idxs = torch.arange(sorted_indices.shape[0], device=input_ids.device)
                    next_token_id = sorted_indices[batch_idxs, sampled_sorted_idx]
                else:
                    next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Freeze finished sequences: replace their token with EOS
            next_token_id[finished] = eos_token_id
    
            # Update which sequences are now finished
            finished |= (next_token_id == eos_token_id)

            input_ids = torch.cat((input_ids, next_token_id.unsqueeze(-1)), dim=-1)

            if finished.all():
                break

        outputs = []
        for i in range(batch_size):
            seq = input_ids[i]
            # Find first EOS and truncate
            eos_positions = (seq == eos_token_id).nonzero(as_tuple=True)[0]
            end = eos_positions[0].item() if len(eos_positions) > 0 else len(seq)
            outputs.append(seq[:end])

        return outputs
