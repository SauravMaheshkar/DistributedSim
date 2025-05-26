import torch
import torch.distributed as dist
from torch.nn import utils as nn_utils

from .communicate import all_reduce, broadcast
from .diloco_mixin import DiLoCoMixin
from .gradient_strategy import GradientStrategy
from .sparta_gradient import RandomIndexSelector


class DiLoCoSPARTAGradient(DiLoCoMixin, GradientStrategy):
    """
    DiLoCo outer optimizer with SPARTA updates at every step
    """

    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)
        self._diloco_init(rank, model, config)
        self.index_selector = RandomIndexSelector(self.gradient_config.p_sparta)

    def step(self):
        if self.gradient_config.max_norm:
            nn_utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.gradient_config.max_norm
            )

        self.optim.step()

        if (
            self.config.num_nodes > 1
            and self.local_step % self.gradient_config.sparta_interval == 0
        ):
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue

                    indices = self.index_selector.get_indices(param, self.local_step)
                    broadcast(indices, src=0)
                    sparse_data = param.data[indices]
                    all_reduce(sparse_data, op=dist.ReduceOp.SUM)
                    sparse_data /= dist.get_world_size()

                    param.masked_scatter_(indices, sparse_data)

        if (
            self.local_step % self.gradient_config.diloco_interval == 0
            and self.local_step > 0
        ):
            self._average_models()

            if self.rank == 0:
                self.outer_optimizer.zero_grad()
                self._set_master_grad()
                self.outer_optimizer.step()
                self._synchronize_master_model()

            self._broadcast_model_params()

        super().step()
