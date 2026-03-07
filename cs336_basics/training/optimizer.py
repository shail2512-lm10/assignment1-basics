from collections.abc import Callable
import torch
import math
from collections.abc import Iterable
from torch import linalg as LA

class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr: float = 0.001,
            betas: tuple[float, float] = (0.9, 0.999),
            weight_decay: float = 0.01,
            eps: float = 1e-8
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                print(p)
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                m = state.get("m", 0)
                v = state.get("v", 0)

                grad = p.grad.data # Get the gradient of loss with respect to p.
                m = (beta1 * m) + ((1 - beta1) * grad)
                v = (beta2 * v) + ((1 - beta2) * grad**2)
                mt = m / (1 - beta1**t)
                vt = v / (1 - beta2**t)


                p.data -= lr * ((mt / (vt**0.5 + eps)) + weight_decay*p.data) # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v

        return loss
    

def cosine_lr_schedule(current_t, lr_max, lr_min, total_warmup_t, total_cos_anneal_t):
    if current_t < total_warmup_t:
        return lr_max * current_t / total_warmup_t
    elif current_t <= total_cos_anneal_t:
        return lr_min + 0.5 * (1 + math.cos((current_t - total_warmup_t) / (total_cos_anneal_t - total_warmup_t) * math.pi)) * (lr_max - lr_min)
    elif current_t > total_cos_anneal_t:
        return lr_min
    

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6

    grads = [p.grad.detach() for p in params if p.grad is not None]

    if len(grads) == 0:
        return torch.tensor(0.0)

    total_norm = LA.matrix_norm(torch.concat(grads), 2)
    print(total_norm)

    if total_norm > max_l2_norm:
        # Scale all gradients by the same factor
        for g in grads:
            g.mul_(max_l2_norm / (total_norm + eps))
