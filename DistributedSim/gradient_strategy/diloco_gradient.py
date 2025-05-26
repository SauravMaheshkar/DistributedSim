from torch.nn import utils as nn_utils

from .diloco_mixin import DiLoCoMixin
from .gradient_strategy import GradientStrategy


class DiLoCoGradient(DiLoCoMixin, GradientStrategy):
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
                self.outer_optimizer.zero_grad()
                self._set_master_grad()
                self.outer_optimizer.step()
                self._synchronize_master_model()

            self._broadcast_model_params()

        super().step()
