#!/bin/bash
#SBATCH --job-name=shock-tol6-build
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:10:00
#SBATCH --chdir=/scratch3/workspace/roohie_umass_edu-jfm1439-r26-m2-20261007/shock_matrix_tol6_20261011
#SBATCH --output=build-%j.out
#SBATCH --error=build-%j.err
set -eo pipefail
source /etc/profile
module load python/3.11.7
sha256sum -c SOURCE_SHA256SUMS
gcc -O3 -ffp-contract=off -fPIC -shared -Wall -Werror bgk_kernel.c -o kernel.so -lm
gcc -O3 -ffp-contract=off -fPIC -shared -Wall -Werror fp_kernel.c -o fp_kernel.so -lm
sha256sum kernel.so fp_kernel.so > COMPILED_SHA256SUMS
