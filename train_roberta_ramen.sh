#!/usr/bin/env bash
set -x;

# PROJECT INPUT FILES
# VOCAB_DIR: VOCAB DIRECTORY - JUR AND ROBERTA
# TGT_INIT: DIRECTORY FOR INIT WEIGHTS AND CONFIG
# PROB PATH: PATH TO TRANSLATION MATRIX
VOCAB_DIR=/work/vocabs
TGT_INIT=/work/roberta_target_init
PROB_PATH=/work/probs/probs.mono.pt-jur.pth

# HYPERPARAMETERS
N_UPDATES=20000
MAX_EPOCH=6

# VOLUMES
# INPUT DATA DIR (.pth files) AND PRETRAINED MODEL
BIN_PATH=/work/binary_data
PRETRAINED_DIR=/work/models/roberta_pretrained
# SAVE PATH
EXP_PATH=/work/models/ramen_jur_roberta

python init_weight_alignment.py --src_vocab "${PRETRAINED_DIR}/vocab.json" --src_model ${PRETRAINED_DIR} --prob ${PROB_PATH} --tgt_model "$TGT_INIT/pytorch_model.bin" --tgt_vocab "$VOCAB_DIR/jur_vocab.txt" --src_merge "$PRETRAINED_DIR/merges.txt"
python ramen_pytorch.py --lr 0.0001 --tgt_lang jur --src_lang pt --batch_size 4 --bptt 256 --src_pretrained_path ${PRETRAINED_DIR} --tgt_pretrained_path ${TGT_INIT} --data_path ${BIN_PATH} --epoch_size ${N_UPDATES} --max_epoch ${MAX_EPOCH} --fp16 --exp_path ${EXP_PATH} --grad_acc_steps 8 --src_model "roberta" --tokenizer "roberta"
