#!/bin/sh
#SBATCH --job-name=trial
#SBATCH --partition=full
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH --gpu-freq=2100
#SBATCH --cpu-freq=3.8
#SBATCH --output=history/logs/training/%x.%j.out
#SBATCH --error=history/logs/training/%x.%j.err

module load python/anaconda3
eval "$(conda shell.bash hook)"
conda activate HPE

expansion=2
epoch=200
JOBID=$SLURM_JOB_ID
export JOBID
# batch_sizes=(64 128 256 512)
batch_sizes=(16 32 64 96 128 256)
criterion=mse
arch="DevModel7"

DATA_DIRS=./datasets/


wandb offline
for i in $(seq 0 5); do
    echo "looping through different data set size"
    batch=${batch_sizes[i]} # 128
    data_dir=$DATA_DIRS
    csv_file=$data_dir
    saved_model_name="${arch}_exp_${expansion}_${criterion}_bs_${batch}_$JOBID"
    saved_model_path="./history/full_model_checkpoints/$saved_model_name"
    python3 train.py --criterion $criterion \
	    --arch $arch \
	    --lr 1e-2 \
	    --epochs $epoch \
	    --batch-size $batch \
	    --weight-decay 0.01 \
	    --mean "mean.npy" \
	    --std "std.npy" \
	    --data-dir $data_dir \
	    --background "avg_background.npy" \
	    --saved-model-path $saved_model_path \
	    --expansion $expansion &
wait
