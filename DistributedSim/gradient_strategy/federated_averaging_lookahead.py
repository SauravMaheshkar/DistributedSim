import random
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.nn import utils as nn_utils

from .communicate import *
from .gradient_strategy import GradientStrategy


class Lookahead(torch.optim.Optimizer):
    r"""PyTorch implementation of the lookahead wrapper.

    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    """

    def __init__(
        self,
        optimizer=torch.optim.SGD,
        la_steps=5,
        la_alpha=0.8,
        pullback_momentum="none",
    ):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["cached_params"] = torch.zeros_like(p.data)
                param_state["cached_params"].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state["cached_mom"] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            "state": self.state,
            "optimizer": self.optimizer,
            "la_alpha": self.la_alpha,
            "_la_step": self._la_step,
            "_total_la_steps": self._total_la_steps,
            "pullback_momentum": self.pullback_momentum,
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["backup_params"] = torch.zeros_like(p.data)
                param_state["backup_params"].copy_(p.data)
                p.data.copy_(param_state["cached_params"])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                p.data.copy_(param_state["backup_params"])
                del param_state["backup_params"]

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    param_state = self.state[p]
                    p.data.mul_(self.la_alpha).add_(
                        param_state["cached_params"], alpha=1.0 - self.la_alpha
                    )  # crucial line
                    param_state["cached_params"].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = (
                            internal_momentum.mul_(self.la_alpha).add_(
                                1.0 - self.la_alpha, param_state["cached_mom"]
                            )
                        )
                        param_state["cached_mom"] = self.optimizer.state[p][
                            "momentum_buffer"
                        ]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )

        return loss


class FedAvgGradient(GradientStrategy):
    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)

        self.local_step = 0
        self.island_size = self.gradient_config.island_size

        if self.gradient_config.optimizer_class is Lookahead:
            # Get base optimizer class and kwargs
            base_opt_class = self.gradient_config.optimizer_base_class
            base_opt_kwargs = self.gradient_config.optimizer_base_kwargs
            base_optimizer = base_opt_class(model.parameters(), **base_opt_kwargs)
            self.optim = Lookahead(
                optimizer=base_optimizer, **self.gradient_config.optimizer_kwargs
            )
        else:
            self.optim = self.gradient_config.optimizer_class(
                model.parameters(), **self.gradient_config.optimizer_kwargs
            )
        self._setup_scheduler()

    def _select_partners(self):
        """
        Selects partners for goruped Federated Averaging. By default not used.
        """
        world_size = dist.get_world_size()

        # Only rank 0 creates the island assignments
        islands = None
        if self.rank == 0:
            # Create a list of all rank numbers and shuffle it.
            ranks = list(range(world_size))
            ## TODO: Switch to pytorch shuffle.
            random.shuffle(ranks)
        else:
            ## TODO: Switch to pytorch broadcast.
            ranks = [None] * world_size

        dist.broadcast_object_list(ranks, src=0)

        islands = []
        for i in range(0, len(ranks), self.island_size):
            islands.append(set(ranks[i : i + self.island_size]))

        # Ugh seems so unoptimal but it's fine for now.
        my_island = None
        for island in islands:
            if self.rank in island:
                my_island = island
                break

        # print(f'Rank {self.rank} has partners {my_island}')

        return my_island

    def _average_models(self, island_members) -> None:
        ## Average model parameters across all members in the island
        for param in self.model.parameters():
            ## At the moment we are doing a full all_gather - this will be optimized in a full-scale training implementation.
            tensor_list = [
                torch.zeros_like(param.data) for _ in range(self.config.num_nodes)
            ]
            all_gather(tensor_list, param.data)

            # Compute average only from ranks in the same island
            island_tensors = [tensor_list[rank] for rank in island_members]
            island_average = sum(island_tensors) / len(island_tensors)

            param.data = island_average

    def step(self):
        if self.gradient_config.max_norm:
            nn_utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.gradient_config.max_norm
            )

        # We have just calculated the loss and done the backward pass.
        # Therefore we do inner step first.
        self.optim.step()

        # Outer step if needed.
        if self.local_step % self.gradient_config.H == 0 and self.local_step > 0:
            if self.island_size < self.config.num_nodes:
                island_members = self._select_partners()
            else:
                island_members = list(range(self.config.num_nodes))

            self._average_models(island_members)

        super().step()
