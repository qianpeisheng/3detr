#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

CUDA_VISIBLE_DEVICES=2 python3 main_SDCoT_pseudo_distill_ema.py \
--dataset_name scannet \
--model_name 3detr_distill \
--num_base_class 9 \
--num_novel_class 9 \
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
--dec_dropout 0. \
--enc_dropout 0. \
--seed 42 \
--enc_type masked \
--test_only \
--test_ckpt ckpts_scannet/debug_4_aux_scannet_SDCoT_9_9_pseudo_distill_ema_masked_0_1/checkpoint.pth
# --checkpoint_dir ckpts_scannet/scannet_SDCoT_17_1_pseudo_masked \
# --checkpoint_name checkpoint_best_6537.pth