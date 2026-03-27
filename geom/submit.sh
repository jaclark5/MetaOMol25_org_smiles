#!/bin/bash

#SBATCH --job-name=Benchmark  ## job name
#SBATCH -p standard              ## use free partition
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1            ## use 1 node, don't ask for multiple
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=32
#SBATCH --account DMOBLEY_LAB
#SBATCH --mem-per-cpu=4G     ## ask for 1Gb memory per CPU
#SBATCH --constraint="intel&fastscratch"
#SBATCH -o stdout.txt
#SBATCH -e stderr.txt

date
hn=`hostname`
echo "Running job on host $hn"

source ~/.bashrc
mamba activate bts

# Need about 225 MB per 100 molecules in a shard

mkdir -p ../geom

python 2_get_smiles_sharded.py \
  --dataset-path "../geom" \
  --output-path "../geom" \
  --ds-name "geom_orca6" \
  --shard-size "10000" \
  --workers $SLURM_CPUS_ON_NODE 2>&1 | tee log.txt

date
