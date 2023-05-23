#!/bin/bash
#SBATCH --mem=16G
#SBATCH -c2
#SBATCH --time=0:30:0
#SBATCH --gres=gpu:1,vmem:10g

python ex1.py 1 3 3 3