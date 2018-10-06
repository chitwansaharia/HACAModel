#!/bin/bash
##SBATCH --time=2  # wall-clock time limit in minutes
#SBATCH -p special
#SBATCH --gres=gpu:4,gpu_mem:4000M  # number of GPUs (keep it at 3) and memory limit
#SBATCH --cpus-per-task=2            # number of CPU cores
#SBATCH --output=logging/HACA_bs64_lr1_cpv20_ss0.9.txt       # output file
##SBATCH --error=error.txt # error file
CUDA_VISIBLE_DEVICES=1
python3 main.py --batch-size 64 --model HACA_bs64_lr1_cpv20_ss0.9 --ss 0.9 --epochs 5000 --log-interval 100 --captions-per-vid 20 --lr 1  --optimizer adadelta