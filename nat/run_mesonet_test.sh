#!/bin/bash
#
# Test run for PNN with Sinh-Arcsinh distribution for rainfall prediction
#
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --output=mesonet_test_%j_stdout.txt
#SBATCH --error=mesonet_test_%j_stderr.txt
#SBATCH --time=30:00
#SBATCH --job-name=mesonet_test
#SBATCH --mail-user=natalie.a.hill-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/nhill/aml/aml-hw5

set -e

. /home/fagg/tf_setup.sh

conda activate dnn

# Create a directory for test results
TEST_DIR="test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $TEST_DIR

# Set PYTHONPATH to include current directory
export PYTHONPATH=$PYTHONPATH:.

# Run a simple test configuration
echo "Running test configuration"

# Set TF debug flags to catch NaN issues
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=0

# Add debugging option to the script
python pnn-rainfall-solution.py \
    --hidden_layers 16,8 \
    --epochs 5 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --dataset_path /home/fagg/datasets/mesonet/allData1994_2000.csv \
    --output_dir $TEST_DIR/test_run \
    --rotation 0 \
    --debug True

echo "Test completed. Results saved to $TEST_DIR" 