#!/usr/bin/env bash
set -x;

# PROJECT INPUT FILES
# VOCAB_DIR: VOCAB DIRECTORY - JUR AND ROBERTA
# TGT_INIT: DIRECTORY FOR INIT WEIGHTS AND CONFIG
# PROB PATH: PATH TO TRANSLATION MATRIX
VOCAB_DIR=/work/vocabs
TGT_INIT=/work/roberta_target_init
PROB_PATH=/work/probs/probs.mono.pt-jur.pth

# HYPERPARAMETERS AND CONFIG
EPOCH_SIZE=20000
MAX_EPOCH=6
LOG_FREQ=50
BATCH_SIZE=4
BPTT=256
GRAD_ACC_STEPS=8
LEARNING_RATE=0.0001

# VOLUMES
# INPUT DATA DIR (.pth files) AND PRETRAINED MODEL
BIN_PATH=/work/binary_data
PRETRAINED_DIR=/work/models/roberta_pretrained
# SAVE PATH
EXP_PATH=/work/models/ramen_jur_roberta

#WANDB
export WANDB_API_KEY="wandb_api_key"
PROJECT_NAME="test_ramen"
RUN_NAME="test_ramen_v8"


python init_weight_alignment.py --src_vocab "${PRETRAINED_DIR}/vocab.json" --src_model ${PRETRAINED_DIR} --prob ${PROB_PATH} --tgt_model "$TGT_INIT/pytorch_model.bin" --tgt_vocab "$VOCAB_DIR/jur_vocab.txt" --src_merge "$PRETRAINED_DIR/merges.txt"
python ramen_pytorch.py --lr ${LEARNING_RATE} --tgt_lang jur --src_lang pt --batch_size ${BATCH_SIZE} --bptt ${BPTT} --src_pretrained_path ${PRETRAINED_DIR} --tgt_pretrained_path ${TGT_INIT} --data_path ${BIN_PATH} --epoch_size ${EPOCH_SIZE} --max_epoch ${MAX_EPOCH} --fp16 --exp_path ${EXP_PATH} --grad_acc_steps ${GRAD_ACC_STEPS} --src_model "roberta" --tokenizer "roberta"  --log_wandb --project_name ${PROJECT_NAME} --run_name ${RUN_NAME} --log_freq ${LOG_FREQ}
