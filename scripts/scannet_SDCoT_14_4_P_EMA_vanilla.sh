#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

CUDA_VISIBLE_DEVICES=1 python3 main_SDCoT_pseudo_ema.py \
--dataset_name scannet \
--model_name 3detr \
--num_base_class 14 \
--num_novel_class 4 \
--max_epoch 510 \
--nqueries 256 \
--matcher_giou_cost 2 \
--matcher_cls_cost 1 \
--matcher_center_cost 0 \
--matcher_objectness_cost 0 \
--loss_giou_weight 1 \
--loss_no_object_weight 0.25 \
--save_separate_checkpoint_every_epoch 50 \
--eval_every_epoch 50 \
--dataset_num_workers 6 \
--batchsize_per_gpu 12 \
--enc_type vanilla \
--seed 42 \
--consistency_ramp_len 100 \
--checkpoint_dir ckpts_scannet/scannet_SDCoT_14_4_pseudo_ema_vanilla \
--checkpoint_name checkpoint_best_5634.pth