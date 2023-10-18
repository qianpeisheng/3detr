#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

CUDA_VISIBLE_DEVICES=2 python3 main_SDCoT_pseudo_ema.py \
--dataset_name scannet \
--model_name 3detr \
--num_base_class 14 \
--num_novel_class 4 \
--max_epoch 310 \
--nqueries 256 \
--matcher_giou_cost 2 \
--matcher_cls_cost 1 \
--matcher_center_cost 0 \
--matcher_objectness_cost 0 \
--loss_giou_weight 1 \
--loss_no_object_weight 0.25 \
--save_separate_checkpoint_every_epoch 20 \
--eval_every_epoch 20 \
--dataset_num_workers 4 \
--batchsize_per_gpu 8 \
--enc_type masked \
--seed 42 \
--consistency_ramp_len 100 \
--checkpoint_dir ckpts_scannet/debug_scannet_SDCoT_14_4_pseudo_ema_masked \
--checkpoint_name checkpoint_best_5973.pth