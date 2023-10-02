#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

CUDA_VISIBLE_DEVICES=1 python3 main_finetune.py \
--dataset_name scannet \
--num_base_class 9 \
--num_novel_class 9 \
--max_epoch 500 \
--nqueries 256 \
--matcher_giou_cost 2 \
--matcher_cls_cost 1 \
--matcher_center_cost 0 \
--matcher_objectness_cost 0 \
--loss_giou_weight 1 \
--loss_no_object_weight 0.25 \
--save_separate_checkpoint_every_epoch 50 \
--eval_every_epoch 50 \
--dataset_num_workers 8 \
--batchsize_per_gpu 16 \
--checkpoint_dir ckpts_scannet/scannet_freeze_9 \
--checkpoint_name checkpoint_best_6198.pth \
--freeze