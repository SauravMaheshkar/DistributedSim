from torch.nn import utils as nn_utils

from .diloco_mixin import DiLoCoMixin
from .gradient_strategy import GradientStrategy


class MiddleManGradient(DiLoCoMixin, GradientStrategy):
    def step(self):
        if self.gradient_config.max_norm:
            nn_utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.gradient_config.max_norm
            )

        self.optim.step()

        interval = self.gradient_config.diloco_interval
        if not self._midstep_done and (self.local_step % interval == interval // 2):
            if self.rank == 0:
                self.outer_optimizer.zero_grad()
                self._set_master_grad()
                self.outer_optimizer.step()
                self._synchronize_master_model()
            self._broadcast_model_params()
            self._midstep_done = True

        if (self.local_step % interval == 0) and (self.local_step > 0):
            self._average_models()
            self._broadcast_model_params()
            self._midstep_done = False

        super().step()
