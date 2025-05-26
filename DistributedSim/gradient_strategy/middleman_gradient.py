from copy import deepcopy

import torch.distributed as dist
from torch.nn import utils as nn_utils

from .communicate import *
from .gradient_strategy import GradientStrategy


class MiddleManGradient(GradientStrategy):
    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)

        if self.rank == 0:
            self.master_model = deepcopy(model).to("cpu")
            for param in self.master_model.parameters():
                param.requires_grad = True
            self.outer_optimizer = self.gradient_config.outer_optimizer_cls(
                self.master_model.parameters(),
                **self.gradient_config.outer_optimizer_kwargs,
            )

        self.optim = self.gradient_config.optimizer_class(
            model.parameters(), **self.gradient_config.optimizer_kwargs
        )
        self._setup_scheduler()
        self._midstep_done = False

    def _average_models(self):
        for param in self.model.parameters():
            all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= self.config.num_nodes

    def _broadcast_model_params(self):
        for param in self.model.parameters():
            broadcast(param.data, src=0)

    def _synchronize_master_model(self):
        for name, param in self.model.named_parameters():
            param.data = self.master_model.state_dict()[name].data.to(param.device)

    def step(self):
        if self.gradient_config.max_norm:
            nn_utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.gradient_config.max_norm
            )

        # Local (inner) step
        self.optim.step()

        # Check if we are at the mid-point (k//2)
        interval = self.gradient_config.diloco_interval
        if not self._midstep_done and (self.local_step % interval == interval // 2):
            if self.rank == 0:
                self.outer_optimizer.zero_grad()
                self._set_master_grad()
                self.outer_optimizer.step()
                self._synchronize_master_model()
            self._broadcast_model_params()
            self._midstep_done = True

        # At the end of k local steps, average and reset
        if (self.local_step % interval == 0) and (self.local_step > 0):
            self._average_models()
            self._broadcast_model_params()
            self._midstep_done = False

        super().step()
