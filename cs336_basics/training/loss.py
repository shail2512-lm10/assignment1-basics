import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy_loss(
        logits: Float[Tensor, "batch_seq vocab"],
        targets: Int[Tensor, " batch_seq"]
) -> Float[Tensor, ""]:

    max_val = torch.max(logits, dim=-1, keepdim=True).values
    new_logits = logits - max_val

    log_probs = new_logits - torch.log(torch.exp(new_logits).sum(dim=-1, keepdim=True))
    
    target_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    return -target_log_probs.mean()

