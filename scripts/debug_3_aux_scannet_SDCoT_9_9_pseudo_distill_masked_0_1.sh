#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

CUDA_VISIBLE_DEVICES=1 python3 main_SDCoT_pseudo_distill.py \
--dataset_name scannet \
--model_name 3detr_distill \
--num_base_class 9 \
--num_novel_class 9 \
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
--distillation_ramp_len 100 \
--loss_distill_weight 0.1 \
--checkpoint_dir ckpts_scannet/debug_3_aux_scannet_SDCoT_9_9_pseudo_distill_masked_0_1 \
--checkpoint_name checkpoint_best_6480.pth