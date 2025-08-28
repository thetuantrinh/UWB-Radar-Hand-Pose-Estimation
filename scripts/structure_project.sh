#!/bin/sh

mkdir -p history
mkdir -p history/plots
mkdir -p history/logs/training
mkdir -p history/logs/data_processing

TRANING_LOG_DIR=history/logs/training
DATA_LOG_DIR=history/logs/data_processing
export TRAINING_LOG_DIR
export DATA_LOG_DIR
