# biomedical-relation-extraction
BioMedical Relation Extraction - Masters Project
1. **Connect HPC**
* Terminal -1
    * ssh SJSU_ID@coe-hpc1.sjsu.edu
    * srun -p gpu -n 1 -N 1 -c 2 --pty /bin/bash 
    * srun -n 1 -N 1 -c 4 --pty /bin/bash 

* Terminal -2
    * ssh -L 52001:localhost:52001 SJSU_ID@coe-hpc1.sjsu.edu
    * jupyter lab --no-browser --port=52001
    * Dataset loading - Dataset Visualize

2. **BIOBERT PYTORCH (github - https://github.com/dmis-lab/biobert-pytorch/)**
    * git clone https://github.com/dmis-lab/biobert-pytorch/
    * cd biobert-pytorch/
    * conda create -n biobert-pytorch python=3.7
    * conda activate biobert-pytorch
    * pip install transformers==3.0.0
    * Run ./download.sh
    * conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    * pip install chardet pandas
    * cd relation-extraction
    * Follow these instructions - https://github.com/dmis-lab/biobert-pytorch/tree/master/relation-extraction
    * python run_re.py \
        --task_name SST-2 \
        --config_name bert-base-cased \
        --data_dir ${DATA_DIR} \
        --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
        --max_seq_length ${MAX_LENGTH} \
        --num_train_epochs ${NUM_EPOCHS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --save_steps ${SAVE_STEPS} \
        --seed ${SEED} \
        --do_train \
        --do_predict \
        --learning_rate 5e-5 \
        --output_dir ${SAVE_DIR}/${ENTITY} \
        --overwrite_output_dir
    * Total time for training - 22:34 minutes

3. **BIOBERT - EUADR training (main repo - https://github.com/dmis-lab/biobert)**
    * git clone https://github.com/dmis-lab/biobert
    * cd biobert; pip install -r requirements.txt
    * conda create -n biobert
    * conda activate biobert
    * conda install python=3.7
    * pip install -r requirements.txt
    * pip install pandas
    * pip install tensorflow==1.15
    * conda install numpy~=1.19.5
    * wget https://github.com/naver/biobert-pretrained/releases/download/v1.0-pubmed-pmc/biobert_v1.0_pubmed_pmc.tar.gz
    * tar -xf biobert_v1.0_pubmed_pmc.tar.gz 
    * export BIOBERT_DIR=./biobert_v1.1_pubmed_pmc
    * echo $BIOBERT_DIR
    * export TASK_NAME=euadr
    * export RE_DIR=./datasets/RE/euadr/1
    * export OUTPUT_DIR=./re_ouptut_euadr
    * python run_re.py --task_name=$TASK_NAME --do_train=true --do_eval=true --do_predict=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/biobert_model.ckpt.data-00000-of-00001 --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --do_lower_case=false --data_dir=$RE_DIR --output_dir=$OUTPUT_DIR > euadr_output.txt 2>&1 &



4. **PPI - Relation Extraction (https://github.com/BNLNLP/PPI-Relation-Extraction/tree/main)**
    * conda create -n PPI-RE
    * conda activate PPI-RE
    * conda install python=3.9
    * pip install -r requirements.txt 
    * export DATASET_DIR=./datasets
    * export OUTPUT_DIR=./output
    * export DATASET_NAME=PPI/original/AImed
    * export SEED=1
    * python src/relation_extraction/run_re.py   --model_list dmis-lab/biobert-base-cased-v1.1   --task_name "re"   --dataset_dir $DATASET_DIR   --dataset_name $DATASET_NAME   --output_dir $OUTPUT_DIR   --do_train   --do_predict   --seed $SEED   --remove_unused_columns False   --save_steps 100000   --per_device_train_batch_size 16   --per_device_eval_batch_size 32   --num_train_epochs 10   --optim "adamw_torch"   --learning_rate 5e-05   --warmup_ratio 0.0   --weight_decay 0.0   --relation_representation "EM_entity_start"   --use_context "attn_based"   --overwrite_cache   --overwrite_output_dir
