#!/bin/bash
#
#SBATCH --job-name=FL_experiment
#SBATCH --output=/vinserver_user/21thinh.dd/FedBackdoor/source/output.txt
#
#SBATCH --ntasks=1 --cpus-per-task=8 --gpus=1
#
sbcast -f /vinserver_user/21thinh.dd/FedBackdoor/source/jobs/run.sh run.sh
srun sh run.sh