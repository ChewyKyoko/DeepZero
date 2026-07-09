import math
import torch


def _newton_schulz(g: torch.Tensor, steps: int = 6) -> torch.Tensor:
    """Newton-Schulz iteration for matrix orthogonalization.

    Approximates polar decomposition: A → A (A^T A)^{-1/2}
    Used by Muon optimizer for weight matrices.
    """
    g = g.float()
    norm = g.norm() + 1e-10
    X = g / norm
    I = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)
    for _ in range(steps):
        XT_X = X.T @ X
        X = X @ (1.5 * I - 0.5 * XT_X)
    return (X * norm).to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer — momentum + Newton-Schulz orthogonalization.

    Paper: "Muon: An Optimizer for Language Models" (Keller Jordan, 2024)
    Applies Newton-Schulz iteration to 2D parameter gradients to encourage
    orthogonal updates, with SGD/momentum for 1D params.

    Default lr: 1e-3 (for small models), momentum: 0.95, ns_steps: 6
    """
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.95,
                 weight_decay: float = 0.1, ns_steps: int = 6):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            lr = group['lr']
            beta = group['momentum']
            wd = group['weight_decay']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad

                if wd > 0:
                    g = g + wd * p.data

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(beta).add_(g)

                if p.dim() >= 2:
                    update = _newton_schulz(buf, ns_steps)
                else:
                    update = buf

                p.data.add_(update, alpha=-lr)
        return loss


class Sophia(torch.optim.Optimizer):
    """Sophia-G optimizer — Hessian-aware preconditioned SGD with clipping.

    Paper: "Sophia: A Scalable Stochastic Second-order Optimizer" (Liu et al., 2023)
    Uses gradient-squared EMA as cheap Hessian diagonal estimate,
    then clips the momentum/preconditioner ratio.

    Default lr: 3e-4, betas: (0.9, 0.95), rho: 1.0
    """
    def __init__(self, params, lr: float = 3e-4, betas: tuple = (0.9, 0.95),
                 rho: float = 1.0, weight_decay: float = 0.1, eps: float = 1e-8):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            rho = group['rho']
            wd = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if 'm' not in state:
                    state['m'] = torch.zeros_like(p)
                    state['h'] = torch.zeros_like(p)
                m, h = state['m'], state['h']

                m.mul_(beta1).add_(g, alpha=1 - beta1)
                h.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                update = m / (h + eps)
                update.clamp_(-rho, rho)

                if wd > 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(update, alpha=-lr)
        return loss


class Lion(torch.optim.Optimizer):
    """Lion optimizer — sign-based evolutionary optimizer.

    Paper: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)
    Discovered via evolutionary search. Combines momentum and gradient via
    sign operation. Extremely simple and memory-efficient.

    Default lr: 1e-4, betas: (0.9, 0.99), weight_decay: 0.1
    """
    def __init__(self, params, lr: float = 1e-4, betas: tuple = (0.9, 0.99),
                 weight_decay: float = 0.1):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                m = state['exp_avg']

                c = m.mul(beta1).add(g, alpha=1 - beta1)
                m.mul_(beta2).add_(g, alpha=1 - beta2)

                if wd > 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(c.sign(), alpha=-lr)
        return loss


OPTIMIZER_REGISTRY = {
    "adamw": lambda params, **kw: torch.optim.AdamW(params, betas=(0.9, 0.95), fused=False, **kw),
    "muon": Muon,
    "sophia": Sophia,
    "lion": Lion,
}


def build_optimizer(model: torch.nn.Module, name: str = "sophia",
                    lr: float = 3e-4, weight_decay: float = 0.1,
                    **extra) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]

    fn = OPTIMIZER_REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"Unknown optimizer '{name}'. Options: {list(OPTIMIZER_REGISTRY)}")

    return fn([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr, **extra)
