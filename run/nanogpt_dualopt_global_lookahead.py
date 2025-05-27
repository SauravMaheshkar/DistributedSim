import torch
from nanogpt import arg_parse, config_gen, gen_gpt_config

from DistributedSim.dataset.nanogpt.build_dataset import *
from DistributedSim.gradient_strategy.federated_averaging_global_lookahead import (
    FedAvgGlobalLookaheadGradient,
)
from DistributedSim.gradient_strategy.gradient_strategy import *
from DistributedSim.sim_builder import *
from DistributedSim.sim_config import *


def main():
    parser = arg_parse()

    parser.add_argument("--diloco_interval", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--outer_lr", type=float, default=0.7)
    parser.add_argument("--nesterov", type=bool, default=True)
    parser.add_argument("--outer_momentum", type=float, default=0.9)

    args = parser.parse_args()

    gpt_config = gen_gpt_config(args)

    config = config_gen(args, gpt_config)

    config.gradient_class = FedAvgGlobalLookaheadGradient
    config.gradient_config.alpha = args.alpha
    config.gradient_config.diloco_interval = args.diloco_interval
    config.gradient_config.outer_optimizer_cls = torch.optim.SGD
    config.gradient_config.outer_optimizer_kwargs = {
        "lr": args.outer_lr,
        "nesterov": args.nesterov,
        "momentum": args.outer_momentum,
    }

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()


if __name__ == "__main__":
    main()
