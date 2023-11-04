#!/bin/sh
#SBATCH -p lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --nice=10000
#SBATCH -t 1-00:00
#SBATCH --container-image='/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di35dul/pp_atlas.sqsh'
#SBATCH --container-mounts=/dss/dsshome1/02/di93zoj/yuge:/mnt/yuge
#SBATCH -o out/run-%A_%a.out
#SBATCH -e out/run-%A_%a.err

pip install -e /mnt/yuge/pertpy
python /mnt/yuge/perturbation-metrics/evaluate.py --save_file=spK562 --dataset=sciplex_K562 --with_DEGs  # full run
python /mnt/yuge/perturbation-metrics/evaluate.py --save_file=spMCF7 --dataset=sciplex_MCF7 --eval_mode
python /mnt/yuge/perturbation-metrics/evaluate.py --save_file=spA549 --dataset=sciplex_A549 --eval_mode
python /mnt/yuge/perturbation-metrics/evaluate.py --save_file=norman --dataset=norman --eval_mode
python /mnt/yuge/perturbation-metrics/evaluate.py --save_file=mcfarland --dataset=mcfarland --eval_mode
python /mnt/yuge/perturbation-metrics/evaluate.py --save_file=schiebinger --dataset=schiebinger --eval_mode
python /mnt/yuge/perturbation-metrics/evaluate.py --save_file=satinha --dataset=satinha --eval_mode
python /mnt/yuge/perturbation-metrics/evaluate.py --save_file=garcia --dataset=garcia --eval_mode
