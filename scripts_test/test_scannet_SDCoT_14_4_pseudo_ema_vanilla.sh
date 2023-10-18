#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

CUDA_VISIBLE_DEVICES=2 python3 main_SDCoT_pseudo_ema.py \
--dataset_name scannet \
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
--dataset_num_workers 16 \
--batchsize_per_gpu 32 \
--seed 42 \
--enc_type vanilla \
--test_only \
--test_ckpt /home1/peisheng/3detr/ckpts_scannet/scannet_SDCoT_14_4_pseudo_distill_vanilla/checkpoint_0100.pth
# --checkpoint_dir ckpts_scannet/scannet_SDCoT_17_1_pseudo_masked \
# --checkpoint_name checkpoint_best_6537.pth