#!/bin/bash
#SBATCH --partition plgrid
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 6
#SBATCH --time 72:00:00
#SBATCH --job-name download-tunnel
#SBATCH --output download-log-%J.txt

module load plgrid/tools/python

## install dependencies
pip install --upgrade click

## start calculation in the background
python3 /net/people/plgamyshenin/masters-thesis---melanoma-analysis-with-fnn/utils/download_from_isic.py --limit 20000 --offset 0 --dir /net/scratch/people/plgamyshenin/data10000
