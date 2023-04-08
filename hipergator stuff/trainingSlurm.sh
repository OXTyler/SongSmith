#!/bin/bash
#SBATCH --job-name=songsmithTrain
#SBATCH --output=ss.out
#SBATCH --error=ss.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mveramiranda@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=32
#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=512mb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:8
#SBATCH --time=20:00:00
pwd; hostname; date

module load conda
conda activate hfrl

python3 ./songsmith_training.py
