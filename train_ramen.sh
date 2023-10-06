#!/usr/bin/env bash
set -x;

# PROJECT INPUT FILES AND MODEL NAME
MODEL_NAME="bert"
VOCAB_DIR="/work/vocabs_${MODEL_NAME}"
TGT_INIT="/work/${MODEL_NAME}_target_init"
PROB_PATH="/work/probs_${MODEL_NAME}/probs.mono.pt-jur.pth"

# HYPERPARAMETERS AND CONFIG
EPOCH_SIZE=20000
MAX_EPOCH=6
LOG_FREQ=50
BATCH_SIZE=4
BPTT=256
GRAD_ACC_STEPS=8
LEARNING_RATE=0.0001

# VOLUMES
BIN_PATH=/work/binary_data
PRETRAINED_DIR="/work/models/${MODEL_NAME}_pretrained"
EXP_PATH="/work/models/ramen_jur_${MODEL_NAME}"

# Check if TGT_INIT exists, if not, create it
if [ ! -d "$TGT_INIT" ]; then
  mkdir -p "$TGT_INIT"
fi

# Copy the config.json from PRETRAINED_DIR to TGT_INIT
cp "${PRETRAINED_DIR}/config.json" "${TGT_INIT}/"

# WANDB
export WANDB_API_KEY="wandb_api_key"
PROJECT_NAME="test_ramen"
RUN_NAME="test_ramen_v8"

# Choose the appropriate command based on the MODEL_NAME
if [ "$MODEL_NAME" == "bert" ]; then
    python init_weight_alignment.py --src_vocab "${PRETRAINED_DIR}/vocab.txt" --src_model ${PRETRAINED_DIR} --prob ${PROB_PATH} --tgt_model "$TGT_INIT/pytorch_model.bin" --tgt_vocab "$VOCAB_DIR/jur_vocab.txt"
elif [ "$MODEL_NAME" == "roberta" ]; then
    python init_weight_alignment.py --src_vocab "${PRETRAINED_DIR}/vocab.json" --src_model ${PRETRAINED_DIR} --prob ${PROB_PATH} --tgt_model "$TGT_INIT/pytorch_model.bin" --tgt_vocab "$VOCAB_DIR/jur_vocab.txt" --src_merge "$PRETRAINED_DIR/merges.txt"
fi

python ramen_pytorch.py --lr ${LEARNING_RATE} --tgt_lang jur --src_lang pt --batch_size ${BATCH_SIZE} --bptt ${BPTT} --src_pretrained_path ${PRETRAINED_DIR} --tgt_pretrained_path ${TGT_INIT} --data_path ${BIN_PATH} --epoch_size ${EPOCH_SIZE} --max_epoch ${MAX_EPOCH} --fp16 --exp_path ${EXP_PATH} --grad_acc_steps ${GRAD_ACC_STEPS} --src_model ${MODEL_NAME} --tokenizer ${MODEL_NAME} --log_wandb --project_name ${PROJECT_NAME} --run_name ${RUN_NAME} --log_freq ${LOG_FREQ}