#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

CUDA_VISIBLE_DEVICES=1 python3 main_base_train.py \
--dataset_name sunrgbd \
--num_base_class 9 \
--max_epoch 1080 \
--enc_type masked \
--nqueries 128 \
--base_lr 7e-4 \
--matcher_giou_cost 3 \
--matcher_cls_cost 1 \
--matcher_center_cost 5 \
--matcher_objectness_cost 5 \
--loss_giou_weight 0 \
--loss_sem_cls_weight 0.8 \
--loss_no_object_weight 0.1 \
--save_separate_checkpoint_every_epoch 50 \
--eval_every_epoch 50 \
--dataset_num_workers 6 \
--batchsize_per_gpu 12 \
--seed 42 \
--enc_type masked \
--checkpoint_dir ckpts_sunrgbd/sunrgbd_base_train_9
