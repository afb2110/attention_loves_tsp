#!/usr/bin/env python

import os
import pprint as pp
import json
from tensorboard_logger import configure

import torch
from torch import optim

from critic_network import CriticNetwork
from options import get_options
from baselines import NoBaseline, RolloutBaseline, CriticBaseline
from tsp import TSP as problem
from train import train_epoch, validate
from encoderdecoder import AttentionModel
from utils import log_values, maybe_cuda_model

from held_karp import held_karp

if __name__ == "__main__":
    opts = get_options()

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    if not opts.no_tensorboard:
        configure(os.path.join(opts.log_dir, "{}_{}".format(problem.NAME, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Load data from load_path
    load_data = {}
    if opts.load_path is not None:
        print('  [*] Loading data from {}'.format(opts.load_path))
        load_data = torch.load(opts.load_path, map_location=lambda storage, loc: storage)  # Load on CPU

    # Initialize model
    model = maybe_cuda_model(AttentionModel(
            opts.embedding_dim,
            opts.hidden_dim,
            problem,
            n_encode_layers=opts.n_encode_layers,
            mask_inner=True,
            mask_logits=True,
            normalization=opts.normalization
        ),
        opts.use_cuda
    )


    # Overwrite model parameters by parameters to load
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    baseline = opts.baseline
    
    if opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    elif opts.baseline == 'critic':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            maybe_cuda_model(
                CriticNetwork(
                    problem.NODE_DIM,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                ),
                opts.use_cuda
            )
        )
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': float(opts.lr_model)}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': float(opts.lr_critic)}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    val_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.val_size)

    val_dataset_tensor = torch.stack(val_dataset.data)
    dist = (val_dataset_tensor.transpose(1, 2).repeat_interleave(opts.graph_size, 2).transpose(1, 2).float() -
            val_dataset_tensor.repeat(1, opts.graph_size, 1).float()).norm(p=2, dim=2).view(opts.val_size,
                                                                                    opts.graph_size, opts.graph_size)
    DP_val_solution = [held_karp(dist[i])[0] for i in range(opts.val_size)]
    DP_val_solution = torch.tensor(DP_val_solution)
    DP_val_solution = DP_val_solution.mean()
    problem.DP_cost = DP_val_solution

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                opts
            )

