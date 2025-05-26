from copy import deepcopy

import torch.distributed as dist
from torch.nn import utils as nn_utils

from .communicate import *
from .gradient_strategy import GradientStrategy


class DualOptGradient(GradientStrategy):
    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)

        # Add a flag to control whether to use diff or just take gradients as is
        self.use_diff_for_outer = getattr(
            self.gradient_config, "use_diff_for_outer", True
        )

        if self.rank == 0:
            self.master_model = deepcopy(model).to("cpu")
            for param in self.master_model.parameters():
                param.requires_grad = True

            self.outer_optimizer = self.gradient_config.outer_optimizer_cls(
                self.master_model.parameters(),
                **self.gradient_config.outer_optimizer_kwargs,
            )
            # Store a snapshot of master model's parameters before local steps
            self._master_params_before = [
                p.data.clone() for p in self.master_model.parameters()
            ]

        self.optim = self.gradient_config.optimizer_class(
            model.parameters(), **self.gradient_config.optimizer_kwargs
        )
        self._setup_scheduler()

    def _average_models(self) -> None:
        for param in self.model.parameters():
            all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= self.config.num_nodes

    def _broadcast_model_params(self) -> None:
        for param in self.model.parameters():
            broadcast(param.data, src=0)

    def _set_master_grad(self) -> None:
        for name, param in self.master_model.named_parameters():
            param.grad = param.data - self.model.state_dict()[name].data.to("cpu")

    def _synchronize_master_model(self) -> None:
        for name, param in self.model.named_parameters():
            param.data = self.master_model.state_dict()[name].data.to(param.device)

    def step(self):
        if self.gradient_config.max_norm:
            nn_utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.gradient_config.max_norm
            )

        # Local (inner) step
        self.optim.step()

        if (
            self.local_step % self.gradient_config.diloco_interval == 0
            and self.local_step > 0
        ):
            self._set_master_grad()
            self._average_models()

            if self.rank == 0:
                self.outer_optimizer.zero_grad()
                self.outer_optimizer.step()
                self._synchronize_master_model()

            self._broadcast_model_params()

        super().step()
