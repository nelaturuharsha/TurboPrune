from typing import TypeVar, Union, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as f

from torch import Tensor
from torch.autograd import Function
from torch.optim.optimizer import Optimizer
from torch.optim import Adam, SGD

# This function defines an optimizer from, https://github.com/bsridatta/Rethinking-Binarized-Neural-Network-Optimization/ which is a binary 
# optimizer to learn the appropriate signs for a network.

# parser.add_argument("--adaptivity-rate", default=10 ** -4, type=float)
# parser.add_argument("--threshold", default=10 ** -8, type=float)

class MomentumWithThresholdBinaryOptimizer(Optimizer):
    def __init__(
        self,
        binary_params,
        bn_params,
        ar: float = 0.0001,
        threshold: float = 0,
        adam_lr=0.01,
    ):
        if not 0 < ar < 1:
            raise ValueError(
                "given adaptivity rate {} is invalid; should be in (0, 1) (excluding endpoints)".format(
                    ar
                )
            )
        if threshold < 0:
            raise ValueError(
                "given threshold {} is invalid; should be > 0".format(threshold)
            )

        self.total_weights = {}
        self._adam = Adam(bn_params, lr=adam_lr)

        defaults = dict(adaptivity_rate=ar, threshold=threshold)
        super(MomentumWithThresholdBinaryOptimizer, self).__init__(binary_params, defaults)


    def step(self, closure: Optional[Callable[[], float]] = ..., ar=None):
        self._adam.step()
        flips = {None}

        for group in self.param_groups:
            params = group["params"]

            y = group["adaptivity_rate"]
            t = group["threshold"]
            flips = {}

            if ar is not None:
                y = ar

            for param_idx, p in enumerate(params):
                grad = p.grad.data
                state = self.state[p]

                if "moving_average" not in state:
                    m = state["moving_average"] = torch.clone(grad).detach()
                else:
                    m: Tensor = state["moving_average"]

                    m.mul_((1 - y))
                    m.add_(grad.mul(y))

                #wherever the gradient has a large positive value, the sign must be flipped
                mask = (m.abs() >= t) * (m.sign() == p.sign())
                mask = mask.double() * -1
                mask[mask == 0] = 1

                flips[param_idx] = (mask == -1).sum().item()

                p.data.mul_(mask)
        return flips

    def zero_grad(self) -> None:
        super().zero_grad()
        self._adam.zero_grad()