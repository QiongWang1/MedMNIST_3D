#!/bin/bash
#SBATCH --job-name=MedMNIST_experiments                # Job name
#SBATCH --output=%x_%j.out               # Output file
#SBATCH --error=%x_%j.err                # Error file
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --partition=short                  # GPU node partition
#SBATCH --nodelist=g[001-009]
#SBATCH --time=00:10:00                  # Time limit (hrs:min:sec)
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=wangqion@bc.edu # Email for notifications

module load miniconda/3
cd /projects/weilab/qiongwang/MedMNIST_experiments/MedMNIST_experiments/MedMNIST3D
python3 train_and_eval_pytorch.py
