# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import math
import torch
import numpy as np
import time
import wandb
import argparse
from torch.cuda.amp import autocast, GradScaler  # Native AMP
from models import RobertaMLM, RobertaAdaptor, BertMLM, BertAdaptor
from ramen.code.src.data.loader import DataIterator
from ramen.code.src.logger import init_logger
from ramen.code.src.optim import AdamInverseSquareRootWithWarmup


def get_parser():
    parser = argparse.ArgumentParser(description="RAMEN-LM")
    parser.add_argument("--exp_path", type=str, default="../experiments",
                        help="path to store experiments dir model")
    parser.add_argument("--src_pretrained_path", type=str, default="",
                        help="path to [bert|xlnet] pretrained dir model")
    parser.add_argument("--tgt_pretrained_path", type=str, default="",
                        help="path to [bert|xlnet] pretrained dir model")
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="path to [bert|roberta|xlnet] pretrained dir model")
    parser.add_argument("--xnli_model", type=str, default="",
                        help="path to trained xnli model")

    parser.add_argument("--ud_model", type=str, default="",
                        help="path to trained parser")
    # masked LM params
    parser.add_argument("--bptt", type=int, default=256,
                        help="number of tokens in a sentence.")
    parser.add_argument("--word_pred", type=float, default=0.15,
                        help="% of masked words")
    # batch parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="number of sentences per batch")

    parser.add_argument("--fp16", action='store_true',
                        help="use float16 for training")
    parser.add_argument("--opt_level", type=str, default='O2',
                        choices=['O1', 'O2'],
                        help='optimization level')
    # data
    parser.add_argument("--data_path", type=str, default="../data/bert/binary",
                        help="Data path")
    parser.add_argument("--src_lang", type=str, default="",
                        help="source languages")
    parser.add_argument("--tgt_lang", type=str, default="",
                        help="target languages")
    parser.add_argument("--max_epoch", type=int, default=100,
                        help="max number of training epoch")
    parser.add_argument("--epoch_size", type=int, default=50000,
                        help="number of updates per epoch")
    parser.add_argument("--lr", type=float, default=0.000005,
                        help="learning rate of Adam")
    parser.add_argument("--optim", type=str, default='adam',
                        help="optimizer.")
    parser.add_argument("--grad_acc_steps", type=int, default=1,
                        help="gradient accumulation steps")
    parser.add_argument("--debug_train", action='store_true',
                        help='fast debugging model')
    parser.add_argument("--tokenizer", choices=["bert", "roberta"],
                        default="bert", type=str,
                        help="tokenizer, needed for special tokens")
    parser.add_argument("--src_model", choices=["bert", "roberta"],
                        default="bert", type=str,
                        help="source pre-trained model")
    parser.add_argument("--log_freq", type=int, default=50,
                        help="Number of steps for logging")
    parser.add_argument("--log_wandb", action='store_true',
                        help="Log using wandb")
    parser.add_argument("--project_name", type=str,
                        help="Wandb project name")
    parser.add_argument("--run_name", type=str,
                        help="Wandb run name")

    return parser


# util for printing time
def _to_hours_mins_secs(time_taken):
    """Convert seconds to hours, mins, and seconds."""
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs


def get_model(params):
    """return pre-trained model, adaptor, and mask_index"""
    model_path = params.src_pretrained_path
    print(f"THIS IS THE MODEL PATH: {model_path}")
    assert 'roberta' in model_path or 'bert' in model_path

    if 'roberta' in model_path:
        mask_index = 52008
        return RobertaMLM, RobertaAdaptor, mask_index
    else:
        mask_index = 103
        return BertMLM, BertAdaptor, mask_index


def mask_input(x, word_pred, mask_index):
    """
    mask the input with a certain probability
    :param x: a tensor of size (bsize, slen)
    :param word_pred: float type, indicate the percentage of masked words
    :param mask_index: int, masking index
    :return: a masked input and the original input
    """
    bsize, slen = x.size()
    npred = math.ceil(bsize * slen * word_pred)
    # make it a multiplication of 8
    npred = (npred // 8) * 8
    # masked words to predict
    y_idx = np.random.choice(bsize * slen, npred, replace=False)
    # keep some identity words
    i_idx = np.random.choice(npred, int(0.10 * npred), replace=False)
    mask = torch.zeros(slen * bsize, dtype=torch.long)
    mask[y_idx] = 1
    # identity (i.e copy)
    mask_ = mask.clone()
    mask_[y_idx[i_idx]] = 0

    mask = mask.view(bsize, slen)
    # do not predict CLS
    mask[:, 0] = 0
    y = mask * x.clone() + (mask - 1)
    # process x
    mask_ = mask_.view(bsize, slen)
    mask_[:, 0] = 0
    x.masked_fill_(mask_ == 1, mask_index)  # mask_index

    return x.cuda(), y.cuda()


def main():
    parser = get_parser()
    params = parser.parse_args()
    if params.log_wandb:
        wandb.init(project=params.project_name, config=vars(params), name=params.run_name)
    if not os.path.exists(params.exp_path):
        os.makedirs(params.exp_path)

    src_lg = params.src_lang
    tgt_lg = params.tgt_lang

    log_file = os.path.join(params.exp_path, f'ramen_{src_lg}-{tgt_lg}.log')

    logger = init_logger(log_file)
    logger.info(params)

    pretrained_model, adaptor, mask_index = get_model(params)
    src_model = pretrained_model.from_pretrained(params.src_pretrained_path)
    tgt_model = pretrained_model.from_pretrained(params.tgt_pretrained_path)
    model = adaptor(src_model, tgt_model)
    model = model.cuda()
    if params.log_wandb:
        wandb.watch(model, log_freq=params.log_freq)

    optimizer = AdamInverseSquareRootWithWarmup(model.parameters(), lr=params.lr, warmup_updates=4000)

    scaler = GradScaler()  # Initialize GradScaler for native AMP

    data = DataIterator(params)

    train_loader_fr = data.get_iter(tgt_lg, 'train')
    train_loader_en = data.get_iter(src_lg, 'train')

    valid_loader_fr = data.get_iter(tgt_lg, 'valid')
    valid_loader_en = data.get_iter(src_lg, 'valid')

    def evaluate(lang, loader):
        model.eval()
        losses = []
        print("Evaluating model")
        for x in loader:
            x, y = mask_input(x.squeeze(0), params.word_pred, mask_index)
            with autocast():  # Native AMP
                loss = model(lang, x, masked_lm_labels=y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    def step(lg, x, update=True):
        x, y = mask_input(x.squeeze(0), params.word_pred, mask_index)
        with autocast():  # Native AMP
            loss = model(lg, x, masked_lm_labels=y)
        scaler.scale(loss).backward()  # Native AMP
        if update:
            scaler.unscale_(optimizer)  # Native AMP
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)  # Native AMP
            scaler.update()  # Native AMP
            optimizer.zero_grad()
        return loss.item()

    n_iter = 0
    n_epoch = 0
    start_time = time.time()
    best_valid_nll = 1e8
    model_prefix = 'roberta' if 'roberta' in params.src_pretrained_path else 'bert'

    while True:
        for batch_en, batch_fr in zip(train_loader_en, train_loader_fr):
            n_iter += 1
            loss_en = step(src_lg, batch_en, False)
            loss_fr = step(tgt_lg, batch_fr, n_iter % params.grad_acc_steps == 0)

            if n_iter % params.log_freq == 0:
                time_taken = time.time() - start_time
                hours, mins, secs = _to_hours_mins_secs(time_taken)
                cur_lr = optimizer.get_lr()
                if params.log_wandb:
                    wandb.log({"loss_source": loss_en, "loss_target": loss_fr, "learning_rate": cur_lr}, step=n_iter)
                logger.info(
                    f" Iter {n_iter:>7} - MLM-{src_lg} {loss_en:.4f} -"
                    f" MLM-{tgt_lg} {loss_fr:.4f} - lr {cur_lr:.7f}"
                    f" elapsed {int(hours)}:{int(mins)}"
                )

            if n_iter % params.epoch_size == 0:
                print(f" Evaluation and CKP Epoch {n_epoch} - Step: {n_iter}")
                n_epoch += 1
                valid_src_nll = evaluate(src_lg, valid_loader_en)
                valid_tgt_nll = evaluate(tgt_lg, valid_loader_fr)
                if params.log_wandb:
                    wandb.log({"source_validation_loss": valid_src_nll, "target_validation_loss": valid_tgt_nll},
                              step=n_iter)
                logger.info(
                    f" Validation - Iter {n_iter} |"
                    f" MLM-{src_lg} {valid_src_nll:.4f} MLM-{tgt_lg} {valid_tgt_nll:.4f}"
                )

                avg_nll = (valid_src_nll + valid_tgt_nll) / 2
                if avg_nll < best_valid_nll:
                    best_valid_nll = avg_nll
                    logger.info(f"| Best checkpoint at epoch: {n_epoch}")

                src_model = f'{model_prefix}_{src_lg}_ep{n_epoch}'
                tgt_model = f'{model_prefix}_{tgt_lg}_ep{n_epoch}'
                src_path = os.path.join(params.exp_path, src_model)
                tgt_path = os.path.join(params.exp_path, tgt_model)

                if not os.path.exists(src_path): os.makedirs(src_path)
                if not os.path.exists(tgt_path): os.makedirs(tgt_path)

                logger.info(f'save ({src_lg}) model to: {src_path}')
                model.src_model.save_pretrained(src_path)
                logger.info(f'save ({tgt_lg}) model to: {tgt_path}')
                model.tgt_model.save_pretrained(tgt_path)

                if n_epoch == params.max_epoch:
                    exit()


if __name__ == '__main__':
    main()