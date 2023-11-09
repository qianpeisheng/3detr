#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

CUDA_VISIBLE_DEVICES=3 python3 main_SDCoT_pseudo_2_source_distill_ema_free_match_new.py \
--dataset_name scannet \
--model_name 3detr_distill \
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
--dataset_num_workers 6 \
--batchsize_per_gpu 12 \
--enc_type masked \
--distillation_ramp_len 100 \
--consistency_ramp_len 100 \
--loss_distill_weight 0.05 \
--loss_center_consistency_weight 1. \
--loss_cls_consistency_weight 0.1 \
--loss_size_consistency_weight 10. \
--use_ema_pseudo_label \
--ema_nms_threshold 0.5 \
--use_cls_threshold \
--checkpoint_dir ckpts_scannet/2_source_2_nms_scannet_SDCoT_14_4_ema_thresh_0_5 \
--checkpoint_name checkpoint_best_5973.pth