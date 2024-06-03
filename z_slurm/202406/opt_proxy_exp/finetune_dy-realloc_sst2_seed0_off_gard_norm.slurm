#!/bin/bash

python main.py train-dyrealloc opt_lora sst2 --load /home/sgh/.cache/huggingface/hub/models--facebook--opt-350m/snapshots/08ab08cc4b72ff5593870b5d527cf4230323703c \
--load-type hf --importance-test-name grad_norm --imp-limit-test-batches 32 --turn-on-percentile 0.25 --dyrealloc-ags-mode off \
--optimizer adamw --learning-rate 1e-4 --weight-decay 0.01 --max-epochs 10 --batch-size 8 --log-every-n-steps 20 \
--task classification --strategy ddp_find_unused_parameters_true --num-devices 1 --pretrained --project-dir \
./ags_output/202406 --lora-config ./configs/lora/lora_networkwise.toml
