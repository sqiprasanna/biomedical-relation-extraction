#!/bin/bash
#SBATCH --mail-user=saiprasanakumar.kumaru@sjsu.edu
#SBATCH --mail-user=/dev/null
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=gpuTest_016651544
#SBATCH --output=gpuTest_%j.out
#SBATCH --error=gpuTest_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00     
##SBATCH --mem-per-cpu=2000
##SBATCH --gres=gpu:p100:1
#SBATCH --partition=gpu


# on coe-hpc1 cluster load
# module load python3/3.8.8
#
# on coe-hpc2 cluster load:
module load python-3.10.8-gcc-11.2.0-c5b5yhp slurm

export http_proxy=http://172.16.1.2:3128; export https_proxy=http://172.16.1.2:3128


cd /home/016651544/llama2/


source /home/016651544/llama2/llama_env/bin/activate

TRAIN_SAMPLES=250
TEST_SAMPLES=50
EPOCHS=5
BATCH_SIZE=16
TRAIN_FILE_PATH="./train_data.tsv" 
OUTPUT_DIR="./experiments/3000_5_FT_model"
MODEL_NAME="meta-llama/Llama-2-7b-hf"

python llama2_fine_tuning_main.py --train_samples ${TRAIN_SAMPLES} --val_samples ${TEST_SAMPLES} --n_epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --train_file_path ${TRAIN_FILE_PATH} --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME}

