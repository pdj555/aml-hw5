#!/bin/bash

#SBATCH --partition=gpu         # Target the general GPU partition
#SBATCH --gres=gpu:ada:1      # Request 1 Ada GPU specifically (node c973)
#SBATCH --nodes=1             # Request 1 node
#SBATCH --ntasks-per-node=1     # 1 task for the python script
#SBATCH --cpus-per-task=8      # Request CPUs
#SBATCH --mem=32G             # Request Memory
#SBATCH --job-name=hw5_mesonet # Job name for identification
#SBATCH --output=hw5_mesonet_%j.out # Standard output file (%j = job ID)
#SBATCH --error=hw5_mesonet_%j.err  # Standard error file (%j = job ID)
#SBATCH --time=1-00:00:00     # Request wall time (1 day)

# --- Email notifications ---
#SBATCH --mail-user=preston.jones@ou.edu
#SBATCH --mail-type=BEGIN,END,FAIL       # Notify on job start, end, and failure

# --- Environment Setup ---
echo "========================================================"
echo "Job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "GPUs Assigned: ${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "CPUs Requested: ${SLURM_CPUS_PER_TASK}"
echo "Memory Requested: ${SLURM_MEM_PER_NODE} MiB"
echo "Start Time: $(date)"
echo "========================================================"

# --- Load TensorFlow setup script (as done in reference scripts) ---
echo "Loading TensorFlow setup..."
. /home/fagg/tf_setup.sh

# --- Activate the conda environment ---
echo "Activating conda environment..."
conda activate dnn  # This is the environment used in reference scripts

# --- Set TensorFlow debug flags ---
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=0

# --- Navigate to your working directory ---
# !!! IMPORTANT: Replace with the actual absolute path to your homework directory !!!
WORKDIR="/home/cs504313/aml/aml-hw5"  # <<< REPLACE THIS with your actual path
echo "Changing to directory: ${WORKDIR}"
cd "${WORKDIR}"
if [ $? -ne 0 ]; then
  echo "Error: Failed to change directory to ${WORKDIR}. Exiting."
  exit 1
fi

# --- Set PYTHONPATH to include current directory (as done in reference) ---
export PYTHONPATH=$PYTHONPATH:.

# --- Run your Python script ---
echo "Running Python script: mesonet_precipitation.py"
python mesonet_precipitation.py
EXIT_CODE=$?
echo "Python script finished with exit code: ${EXIT_CODE}"

# --- End Job ---
echo "========================================================"
echo "Job finished at $(date)"
echo "========================================================"

exit ${EXIT_CODE} 
