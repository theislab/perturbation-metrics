#!/bin/sh
#SBATCH -p lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --nice=10000
#SBATCH -t 1-00:00
#SBATCH --container-image='/dss/dsshome1/02/di93zoj/pytorch.sqsh'
#SBATCH --container-mounts=/dss/dsshome1/02/di93zoj/yuge:/mnt/yuge
#SBATCH -o out/run-%A_%a.out
#SBATCH -e out/run-%A_%a.err

pip install --upgrade pip
pip install -e /mnt/yuge/pertpy
python /mnt/yuge/perturbation-metrics/evaluate.py --save_file=metric_runs_spMCF7.pkl --dataset=sciplex_MCF7
