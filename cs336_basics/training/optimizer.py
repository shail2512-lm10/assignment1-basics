from collections.abc import Callable
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params: torch.nn.Parameter,
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