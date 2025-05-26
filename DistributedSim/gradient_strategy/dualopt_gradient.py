from copy import deepcopy

import torch.distributed as dist
from torch.nn import utils as nn_utils

from .communicate import all_reduce, broadcast
from .gradient_strategy import GradientStrategy


class DualOptGradient(GradientStrategy):
    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)
        self.rank = rank
        self.model = model
        self.config = config
        self.gradient_config = config.gradient_config

        self.saved_model_state = {}
        for name, param in self.model.named_parameters():
            self.saved_model_state[name] = param.data.clone().cpu()

        if self.rank == 0:
            self.master_model = deepcopy(model).cpu()
            for param in self.master_model.parameters():
                param.requires_grad = True

        self.optim = self.gradient_config.optimizer_class(
            model.parameters(), **self.gradient_config.optimizer_kwargs
        )

        self.outer_optimizer = self.gradient_config.outer_optimizer_cls(
            self.model.parameters(),
            **self.gradient_config.outer_optimizer_kwargs,
        )

        self._setup_scheduler()

    def _set_local_grad(self):
        """Compute gradients as difference between current local model and saved state"""
        for name, param in self.model.named_parameters():
            param.grad = param.data - self.saved_model_state[name].to(param.device)

    def _broadcast_model_params(self) -> None:
        """Broadcast model parameters from rank 0 to all other ranks"""
        for param in self.model.parameters():
            broadcast(param.data, src=0)

    def _average_models(self) -> None:
        """Average model parameters across all ranks"""
        for param in self.model.parameters():
            all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= self.config.num_nodes

    def _update_saved_state(self):
        """Update the saved model state after outer optimizer step"""
        for name, param in self.model.named_parameters():
            self.saved_model_state[name] = param.data.clone().cpu()

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
            if self.rank == 0:
                self.outer_optimizer.zero_grad()
                self._set_local_grad()
                self.outer_optimizer.step()

            self._average_models()
            self._broadcast_model_params()
            self._update_saved_state()

        super().step()
