#!/bin/bash

#SBATCH --job-name=omol25-spice  ## job name
#SBATCH -p standard              ## use free partition
#SBATCH -t 5-00:00:00
#SBATCH --nodes=1            ## use 1 node, don't ask for multiple
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=32
#SBATCH --account DMOBLEY_LAB
#SBATCH --mem-per-cpu=1G     ## ask for 1Gb memory per CPU
#SBATCH --constraint="intel&fastscratch"
#SBATCH -o stdout.txt
#SBATCH -e stderr.txt

date
hn=`hostname`
echo "Running job on host $hn"

source ~/.bashrc
mamba activate bts

# Need about 225 MB per 100 molecules in a shard

mkdir -p ../../ani2x

python ../2_get_smiles_sharded.py \
  --dataset-path "../../ani2x" \
  --output-path "../../ani2x" \
  --ds-name "ani2x" \
  --shard-size "10000" \
  --workers "$SLURM_CPUS_ON_NODE" \
  --executor process \
  --max-inflight 128 \
  --write-workers 2 \
  --suppress-toolkit-warnings \
  2>&1 | tee log.txt

date
