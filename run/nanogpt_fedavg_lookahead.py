import torch
from nanogpt import arg_parse, config_gen, gen_gpt_config

from DistributedSim.dataset.nanogpt.build_dataset import *
from DistributedSim.gradient_strategy.federated_averaging_lookahead import *
from DistributedSim.gradient_strategy.gradient_strategy import *
from DistributedSim.sim_builder import *
from DistributedSim.sim_config import *


def main():
    parser = arg_parse()

    parser.add_argument("--H", type=int, default=100)
    parser.add_argument("--island_size", type=int, default=None)

    args = parser.parse_args()

    if args.island_size is None:
        args.island_size = args.num_nodes

    gpt_config = gen_gpt_config(args)

    config = config_gen(args, gpt_config)

    config.gradient_class = FedAvgGradient
    config.gradient_config.H = args.H
    config.gradient_config.island_size = args.island_size

    config.gradient_config.optimizer_class = Lookahead
    config.gradient_config.optimizer_kwargs = {
        "la_steps": args.H,
        "la_alpha": 0.8,
        "pullback_momentum": "none",
    }
    config.gradient_config.optimizer_base_class = torch.optim.Adam
    config.gradient_config.optimizer_base_kwargs = {
        "lr": 3e-4,
    }

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()


if __name__ == "__main__":
    main()
