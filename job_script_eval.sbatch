#!/bin/bash
#SBATCH --job-name=normal
#SBATCH --output=err/job_output_%A_%a.txt
#SBATCH --error=err/job_error_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=150GB
#SBATCH --cpus-per-task=6
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=0
#SBATCH --array=0-11

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
    # "saunders saunders"
    # "spMCF7_rep1 sciplex_MCF7-batch-replicate-rep1"
    # "spMCF7_rep2 sciplex_MCF7-batch-replicate-rep2"
    # "spA549_rep1 sciplex_A549-batch-replicate-rep1"
    # "spA549_rep2 sciplex_A549-batch-replicate-rep2"
    # "spK562_rep1 sciplex_K562-batch-replicate-rep1"
    # "spK562_rep2 sciplex_K562-batch-replicate-rep2"
    # "tahoe_rep1 tahoe_A549-batch-plate-plate_12"
    # "tahoe_rep2 tahoe_A549-batch-plate-plate_2"
    # "tahoeBT474_rep1 tahoe_BT474-batch-plate-plate_12"
    # "tahoeBT474_rep2 tahoe_BT474-batch-plate-plate_6"
    # "tahoeHT29_rep1 tahoe_HT29-batch-plate-plate_12"
    # "tahoeHT29_rep2 tahoe_HT29-batch-plate-plate_2"
    # "schiebinger_rep1 schiebinger-batch-replicate-1"
    # "schiebinger_rep2 schiebinger-batch-replicate-2"
)

source activate pertpy_clean2
# python evaluate.py \
#     --save_file=$(echo ${settings[$SLURM_ARRAY_TASK_ID]} | cut -d' ' -f1) \
#     --dataset=$(echo ${settings[$SLURM_ARRAY_TASK_ID]} | cut -d' ' -f2) \
#     --n_random_states=10 \
#     --eval_mode
python evaluate.py \
    --save_file=$(echo ${settings[$SLURM_ARRAY_TASK_ID]} | cut -d' ' -f1) \
    --dataset=$(echo ${settings[$SLURM_ARRAY_TASK_ID]} | cut -d' ' -f2) \
    --n_random_states=5 \
    --eval_mode \
    --signal_injection \
    --dir=signal_injection