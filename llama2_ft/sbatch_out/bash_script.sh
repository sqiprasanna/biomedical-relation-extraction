#!/bin/bash
#SBATCH --mail-user=saiprasanakumar.kumaru@sjsu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=llama2_016651544
#SBATCH --output=llama2_%j.out
#SBATCH --error=llama2_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu


# on coe-hpc1 cluster load
# module load python-3.10.8-gcc-12.2.0-rq7r6nv
# module load cuda/12.2

#
# on coe-hpc2 cluster load:
# module load python-3.10.8-gcc-11.2.0-c5b5yhp slurm

export http_proxy=http://172.16.1.2:3128; export https_proxy=http://172.16.1.2:3128
export HUGGINGFACE_TOKEN=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


cd /home/016651544/llama2/


source /home/016651544/llama2/llama_env/bin/activate

TRAIN_SAMPLES=10000
TEST_SAMPLES=1000
EPOCHS=25
BATCH_SIZE=64
TRAIN_FILE_PATH="./train_data.tsv" 
# OUTPUT_DIR="./experiments/3000_5_FT_model"
OUTPUT_DIR="./experiments/${TRAIN_SAMPLES}_${EPOCHS}_FT_model"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
# MODEL_NAME="./experiments/250_50_FT_model"

python /home/016651544/llama2/llama2_fine_tuning_main.py --train_samples ${TRAIN_SAMPLES} --val_samples ${TEST_SAMPLES} --n_epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --train_file_path ${TRAIN_FILE_PATH} --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME}