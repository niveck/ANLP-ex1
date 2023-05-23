#!/bin/bash
#SBATCH --mem=32G
#SBATCH -c1
#SBATCH --time=5:30:0
#SBATCH --gres=gpu:a5000:1

../../venv.anlp.ex1/bin/python ex1.py 3 -1 -1 -1