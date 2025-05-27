from copy import deepcopy

import torch.distributed as dist
from torch.nn import utils as nn_utils

from .communicate import *
from .gradient_strategy import GradientStrategy


class FedAvgGlobalLookaheadGradient(GradientStrategy):
    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)

        if self.rank == 0:
            self.master_model = deepcopy(model).to("cpu")
            for param in self.master_model.parameters():
                param.requires_grad = True

            self.old_master_state = {}
            for name, param in self.master_model.named_parameters():
                self.old_master_state[name] = param.data.clone().cpu()

            self.outer_optimizer = self.gradient_config.outer_optimizer_cls(
                self.master_model.parameters(),
                **self.gradient_config.outer_optimizer_kwargs,
            )

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

    def step(self):
        if self.gradient_config.max_norm:
            nn_utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.gradient_config.max_norm
            )

        self.optim.step()

        if (
            self.local_step % self.gradient_config.diloco_interval == 0
            and self.local_step > 0
        ):
            self._average_models()

            if self.rank == 0:
                for name, param in self.master_model.named_parameters():
                    param.data = self.old_master_state[
                        name
                    ] * self.gradient_config.alpha + param.data * (
                        1 - self.gradient_config.alpha
                    )
                self.old_master_state = deepcopy(self.master_model.state_dict())

            self._broadcast_model_params()

        super().step()
