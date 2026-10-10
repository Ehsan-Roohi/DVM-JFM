#!/bin/bash
#SBATCH --job-name=shock-v19-fig5
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --chdir=/scratch3/workspace/roohie_umass_edu-jfm1439-r26-m2-20261007/shock_v19_figure5_regen_20261010
#SBATCH --output=/scratch3/workspace/roohie_umass_edu-jfm1439-r26-m2-20261007/shock_v19_figure5_regen_20261010/slurm-%j.log
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
sha256sum -c INPUT_SHA256SUMS
/project/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dde-tf/bin/python run_frozen_figure5.py
