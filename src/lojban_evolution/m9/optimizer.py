import torch
import math
from torch.optim import Optimizer

class AdaHessian(Optimizer):
    """
    M9.1 AdaHessian: Second-order optimization with Hutchinson trace approximation.
    Isolates expensive HVP calculations to targeted layers to prevent 16GB VRAM trap.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-4, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdaHessian, self).__init__(params, defaults)

    def zero_hessian(self):
        """Zero out the Hessian diagonal before next Hutchinson sampling."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                state['hessian'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['hessian'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                hessian = state['hessian']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Weight decay
                if group['weight_decay'] != 0:
                    p.grad.add_(p.data, alpha=group['weight_decay'])

                # Momentum (First order)
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                
                # Second order curvature (Hessian diagonal)
                # Note: Hutchinson sampling must be done before step()
                exp_avg_sq.mul_(beta2).add_(hessian.pow(2), alpha=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
