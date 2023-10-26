#!/bin/bash

torchrun --nproc_per_node 2 main.py train --config configs/opt_plain.toml
# OPT125M + RTE, 40 cpu + 2 gpu: 4min to setup & tokenize data splits before running fit