import torch
from jaxtyping import Float
from torch import Tensor
from einx import rearrange, reduce


def cross_entropy_loss(
        logits: Float[Tensor, "batch_seq vocab"],
        targets: Float[Tensor, " batch_seq"]
) -> Float[Tensor, ""]:

    max_val = torch.max(logits, dim=-1, keepdim=True).values
    new_logits = logits - max_val

    log_probs = new_logits - torch.log(torch.exp(new_logits).sum(dim=-1, keepdim=True))
    
    # This one is a bit tricky, need to understand better
    target_log_probs = log_probs[torch.arange(logits.shape[0]), targets]

    return -target_log_probs.mean()


