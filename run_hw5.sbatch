#!/bin/bash

#SBATCH --partition=gpu         # Target the general GPU partition
#SBATCH --gres=gpu:ada:1      # Request 1 Ada GPU specifically
#SBATCH --nodes=1             # Request 1 node
#SBATCH --ntasks-per-node=1     # 1 task per python script (change if parallelizing)
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

# Load TensorFlow setup script
echo "Loading TensorFlow setup..."
. /home/fagg/tf_setup.sh

# Activate conda environment
echo "Activating conda environment..."
conda activate dnn

# Set TensorFlow debug flags
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=0

# Create a directory for all results
RESULTS_DIR="rotation_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# Save configuration details
echo "Creating results directory: ${RESULTS_DIR}"
echo "# Experiment Configuration" > ${RESULTS_DIR}/config.txt
echo "Date: $(date)" >> ${RESULTS_DIR}/config.txt
echo "Results Directory: ${RESULTS_DIR}" >> ${RESULTS_DIR}/config.txt
echo "Hidden Layers: 64,32,16" >> ${RESULTS_DIR}/config.txt
echo "Learning Rate: 0.0005" >> ${RESULTS_DIR}/config.txt
echo "Dropout: 0.2" >> ${RESULTS_DIR}/config.txt
echo "Batch Size: 128" >> ${RESULTS_DIR}/config.txt
echo "Epochs: 100" >> ${RESULTS_DIR}/config.txt
echo "GPU: ${CUDA_VISIBLE_DEVICES:-NONE}" >> ${RESULTS_DIR}/config.txt

# Set PYTHONPATH to include current directory
export PYTHONPATH=$PYTHONPATH:.

# Run the model for all 8 rotations
for rotation in {0..7}
do
    output_dir="${RESULTS_DIR}/rotation_${rotation}"
    
    echo "Running rotation ${rotation}"
    
    # Run the model with this rotation
    python mesonet_precipitation.py \
        --hidden_layers "64,32,16" \
        --epochs 100 \
        --batch_size 128 \
        --learning_rate 0.0005 \
        --dropout 0.2 \
        --dataset_path /home/fagg/datasets/mesonet/allData1994_2000.csv \
        --output_dir "${output_dir}" \
        --rotation ${rotation} \
        --debug False
done

# Run the analysis script
echo "Running combined analysis script"
mkdir -p "${RESULTS_DIR}/combined_analysis"
python analyze_rotations.py --results_dir "${RESULTS_DIR}" --output_dir "${RESULTS_DIR}/combined_analysis"

# --- Process results ---
echo "Processing results..."

# Copy the reflection template to the current directory for easier editing
if [ -f "${RESULTS_DIR}/combined_analysis/reflection_template.md" ]; then
    cp "${RESULTS_DIR}/combined_analysis/reflection_template.md" ./reflection_template.md
    echo "* Copied reflection template to current directory: reflection_template.md"
fi

# Count figures
NUM_FIGURES=$(find "${RESULTS_DIR}/combined_analysis" -name "figure*.png" | wc -l)
echo "Generated figures: ${NUM_FIGURES}"

# List the figures
echo "Figures:"
find "${RESULTS_DIR}/combined_analysis" -name "figure*.png" -exec basename {} \; | sort

# Create a symlink to the latest results
rm -f latest_results
ln -s "${RESULTS_DIR}" latest_results
echo "* Created symlink 'latest_results' pointing to ${RESULTS_DIR}"

# Print summary information
echo "========================================================"
echo "Summary of Results:"
echo "Total Rotations: 8"
echo "Combined Figures: ${RESULTS_DIR}/combined_analysis/"
echo "Individual Rotation Results: ${RESULTS_DIR}/rotation_*/"
echo "Reflection Template: ./reflection_template.md"
echo "========================================================"

# --- End Job ---
echo "========================================================"
echo "Job finished at $(date)"
echo "========================================================"