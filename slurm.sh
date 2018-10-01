#!/bin/bash
##SBATCH --time=2  # wall-clock time limit in minutes
#SBATCH -p special
#SBATCH --gres=gpu:4,gpu_mem:4000M  # number of GPUs (keep it at 3) and memory limit
#SBATCH --cpus-per-task=2            # number of CPU cores
#SBATCH --output=logging/HACA_bs64_lr1_new_ss0.9_cpv20.txt       # output file
##SBATCH --error=error.txt # error file
CUDA_VISIBLE_DEVICES=2
python3 main.py --batch-size 64 --model HACA_bs64_lr1_new_ss0.9_cpv20 --epoch 500 --ss 0.9  --log-interval 20 --captions-per-vid 20 --lr 1  --optimizer adadelta
# python3 gen_features.py