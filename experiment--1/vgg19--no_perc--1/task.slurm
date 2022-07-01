#!/bin/bash
#SBATCH --partition plgrid-gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 6
#SBATCH --time 72:00:00
#SBATCH --job-name vgg19--no_perc--1
#SBATCH --output vgg19--no_perc--1/log-%J.txt

module load plgrid/tools/python


## install dependencies
pip install --upgrade tensorflow tensorflow_hub tensorflow_addons matplotlib scikit-learn

## start calculation in the background
jupyter nbconvert --to notebook --execute /net/people/plgamyshenin/masters-thesis---melanoma-analysis-with-fnn/vgg19--no_perc--1/notebook.ipynb
