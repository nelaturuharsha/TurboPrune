#!/bin/bash

#SBATCH --job-name=imagenet-test
#SBATCH --output=/home/c02hane/CISPA-projects/neuron_pruning-2024/lottery-ticket-harness/job-%j.out

#SBATCH --gres=gpu:A100:8
#SBATCH --partition=tmp 
#SBATCH --time=10000

JOBDATADIR=`ws create work --space "$SLURM_JOB_ID" --duration "7 00:00:00"`
JOBTMPDIR=/tmp/job-"$SLURM_JOB_ID"

srun mkdir "$JOBTMPDIR"
srun mkdir -p "$JOBDATADIR" "$JOBTMPDIR"

# test for the credentials files
echo "Running job $1"
echo "Job Data Dir: $JOBDATADIR"
echo "Job tmp dir: $JOBTMPDIR"

srun --container-image=projects.cispa.saarland:5005#c02hane/test-project:imagenetv2 --container-mounts="$JOBTMPDIR":/tmp bash /home/c02hane/CISPA-projects/neuron_pruning-2024/lottery-ticket-harness/"$1" 
srun mv /tmp/job-"$SLURM_JOB_ID".out ~/CISPA-scratch/c02hane/job_"$SLURM_JOB_ID".out.txt
srun mv "$JOBTMPDIR"/ "$JOBDATADIR"/data
