#!/bin/bash
#SBATCH --job-name=metrics_eval
#SBATCH --output=err/job_output_%A_%a.txt
#SBATCH --error=err/job_error_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=150GB
#SBATCH --cpus-per-task=6
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=0
#SBATCH --array=0-0

settings=(
    "sim simulated"
    "spMCF7 sciplex_MCF7"
    "spA549 sciplex_A549"
    "spK562 sciplex_K562"
    "norman norman"
    "mcfarland mcfarland"
    "schiebinger schiebinger"
    "satinha satinha"
    "garcia garcia"
    "tahoe tahoe_A549"
    "tahoeBT474 tahoe_BT474"
    "tahoeHT29 tahoe_HT29"
)

source activate pertpy_clean2
python evaluate.py --save_file=$(echo ${settings[$SLURM_ARRAY_TASK_ID]} | cut -d' ' -f1) --dataset=$(echo ${settings[$SLURM_ARRAY_TASK_ID]} | cut -d' ' -f2) --eval_mode
