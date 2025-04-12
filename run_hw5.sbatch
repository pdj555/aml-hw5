#!/bin/bash

#SBATCH --partition=gpu         # Target the general GPU partition
#SBATCH --gres=gpu:ada:1      # Request 1 Ada GPU specifically
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

# Exit on error for better error handling
set -e

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
export TF_CPP_MIN_LOG_LEVEL=1  # Suppress some TensorFlow warnings

# --- Navigate to your working directory ---
WORKDIR="/home/cs504313/aml/aml-hw5"
echo "Changing to directory: ${WORKDIR}"
cd "${WORKDIR}"
if [ $? -ne 0 ]; then
  echo "Error: Failed to change directory to ${WORKDIR}. Exiting."
  exit 1
fi

# --- Create a directory for all results ---
RESULTS_DIR="rotation_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# --- Save configuration details ---
echo "Creating results directory: ${RESULTS_DIR}"
echo "# Experiment Configuration" > ${RESULTS_DIR}/config.txt
echo "Date: $(date)" >> ${RESULTS_DIR}/config.txt
echo "Results Directory: ${RESULTS_DIR}" >> ${RESULTS_DIR}/config.txt
echo "Hidden Layers: 256,128,64,32" >> ${RESULTS_DIR}/config.txt
echo "Learning Rate: 0.0005" >> ${RESULTS_DIR}/config.txt
echo "Dropout: 0.2" >> ${RESULTS_DIR}/config.txt
echo "Batch Size: 128" >> ${RESULTS_DIR}/config.txt
echo "Epochs: 500" >> ${RESULTS_DIR}/config.txt
echo "GPU: ${CUDA_VISIBLE_DEVICES:-NONE}" >> ${RESULTS_DIR}/config.txt

# --- Set PYTHONPATH to include current directory ---
export PYTHONPATH=$PYTHONPATH:.

# --- Run your Python script ---
echo "Running Python script: mesonet_precipitation.py"
echo "Results will be saved to: ${RESULTS_DIR}"
python mesonet_precipitation.py 2>&1 | tee "${RESULTS_DIR}/output.log"
EXIT_CODE=$?

# --- Process results if successful ---
if [ $EXIT_CODE -eq 0 ]; then
    echo "Run completed successfully!"
    
    # Copy the reflection template to the main directory for easier editing
    if [ -f "${RESULTS_DIR}/reflection_template.md" ]; then
        cp "${RESULTS_DIR}/reflection_template.md" ./reflection_template.md
        echo "* Copied reflection template to current directory: reflection_template.md"
    fi
    
    # Count figures
    NUM_FIGURES=$(find "${RESULTS_DIR}" -name "figure*.png" | wc -l)
    echo "Generated figures: ${NUM_FIGURES}"
    
    # List the figures
    echo "Figures:"
    find "${RESULTS_DIR}/combined_analysis" -name "figure*.png" -exec basename {} \; | sort
    
    # Create link to latest results
    rm -f latest_results
    ln -s "${RESULTS_DIR}" latest_results
    echo "* Created symlink 'latest_results' pointing to ${RESULTS_DIR}"
else
    echo "Run failed with exit code: ${EXIT_CODE}"
    echo "Check the logs in ${RESULTS_DIR} for details."
fi

# --- End Job ---
echo "========================================================"
echo "Job finished at $(date)"
echo "========================================================"

exit ${EXIT_CODE} 
