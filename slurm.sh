#!/bin/bash
##SBATCH --time=2  # wall-clock time limit in minutes
#SBATCH -p special
#SBATCH --gres=gpu:4,gpu_mem:4000M  # number of GPUs (keep it at 3) and memory limit
#SBATCH --cpus-per-task=2            # number of CPU cores
#SBATCH --output=logging/HACAModel_bs80_maxmode.txt       # output file
##SBATCH --error=error.txt # error file
CUDA_VISIBLE_DEVICES=1
python3 main.py --batch-size 80 --model HACAModel_bs80_maxmode  --epoch 5000 --model-type haca --log-interval 50 --captions-per-vid 20 --lr 1  --optimizer adadelta