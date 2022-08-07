#!/bin/bash
#SBATCH --partition plgrid-gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 6
#SBATCH --time 72:00:00
#SBATCH --job-name XC-P-1-05x05
#SBATCH --output XC-P-1-05x05/log-%J.txt

module load plgrid/tools/python


## install dependencies
pip install --upgrade tensorflow tensorflow_hub tensorflow_addons matplotlib scikit-learn

## start calculation in the background
jupyter nbconvert --allow-errors --to notebook --execute /net/people/plgamyshenin/masters-thesis---melanoma-analysis-with-fnn/experiment--2/XC-P-1-05x05/notebook.ipynb
