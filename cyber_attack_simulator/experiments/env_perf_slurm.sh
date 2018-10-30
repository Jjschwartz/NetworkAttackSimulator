#!/bin/bash
#SBATCH --partition=cosc
#SBATCH --time=01:30:00
#SBATCH --job-name=env_experiment
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=14GB

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

DATE=$(date +"%Y%m%d%H%M%S")
echo "time started  "$DATE
echo "This is job '$SLURM_JOB_NAME' (id: $SLURM_JOB_ID) running on the following nodes:"
echo $SLURM_NODELIST
echo "running with OMP_NUM_THREADS= $OMP_NUM_THREADS "
echo "running with SLURM_TASKS_PER_NODE= $SLURM_TASKS_PER_NODE "
echo "running with SLURM_NPROCS= $SLURM_NPROCS "
echo "Now we start the show:"
export TIMEFORMAT="%E sec"

outfile="test.csv"
minM=3
maxM=3
intM=1
minS=1
maxS=3
intS=1
aPerRun=100000
runs=10
append=0

module load anaconda3
stdbuf -oL -eL time python env_perf_exp.py $outfile $minM $maxM $intM $minS $maxS $intS $aPerRun $runs $append

DATE=$(date +"%Y%m%d%H%M%S")
echo "time finished "$DATE

# echo "we just ran with the following SLURM environment variables"
# env | grep SLURM
