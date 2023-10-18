#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

CUDA_VISIBLE_DEVICES=0 python3 main_SDCoT_pseudo_only.py \
--dataset_name scannet \
--num_base_class 9 \
--num_novel_class 9 \
--max_epoch 510 \
--nqueries 256 \
--matcher_giou_cost 2 \
--matcher_cls_cost 1 \
--matcher_center_cost 0 \
--matcher_objectness_cost 0 \
--loss_giou_weight 1 \
--loss_no_object_weight 0.25 \
--save_separate_checkpoint_every_epoch 25 \
--eval_every_epoch 25 \
--dataset_num_workers 6 \
--batchsize_per_gpu 12 \
--seed 42 \
--enc_type masked \
--checkpoint_dir ckpts_scannet/scannet_SDCoT_9_9_pseudo_masked \
--checkpoint_name checkpoint_best_6480.pth