#!/bin/bash
#SBATCH --partition plgrid-gpu
#SBATCH --gres=gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 6
#SBATCH --time 72:00:00
#SBATCH --job-name jupyter-notebook-tunnel
#SBATCH --output jupyter-log-%J.txt

## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)
user=$USER

## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -o ServerAliveInterval=300 -N -L $ipnport:$ipnip:$ipnport ${user}@pro.cyfronet.pl
    -----------------------------------------------------------------

    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:$ipnport  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "

module load plgrid/tools/python

## install dependencies
pip install --upgrade tensorboard

## start an ipcluster instance and launch jupyter server
tensorboard --logdir ./masters-thesis---melanoma-analysis-with-fnn/logs/fit --host $ipnip --port $ipnport
