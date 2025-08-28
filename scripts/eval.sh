#!/bin/sh

YAML_FILE="history/training_information_seperated.yaml"
CHECKPOINT="./history/full_model_checkpoints/DevModel7_exp_2_mse_bs_128_16710_checkpoint_best.pt"

python3 eval.py --config-file $YAML_FILE --checkpoint-path $CHECKPOINT
