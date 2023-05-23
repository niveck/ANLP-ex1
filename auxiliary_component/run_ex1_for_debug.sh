#!/bin/bash
#SBATCH --mem=16G
#SBATCH -c2
#SBATCH --time=2:0:0
#SBATCH --gres=gpu:1,vmem:10g

../../venv.anlp.ex1/bin/python ex1.py 1 -1 -1 -1