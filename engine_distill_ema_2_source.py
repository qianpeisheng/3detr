# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import datetime
import logging
import math
import time
import sys
import numpy as np
from torch.distributed.distributed_c10d import reduce
from utils.ap_calculator import APCalculator
from utils.misc import SmoothedValue
from utils.dist import (
    all_gather_dict,
    all_reduce_average,
    is_primary,
    reduce_dict,
    barrier,
)
from utils import ramps


def get_current_weight(epoch, weight, ramp_len):
    # ramp-up from https://arxiv.org/abs/1610.02242
    return weight * ramps.sigmoid_rampup(epoch, ramp_len)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


TRAIN_SET_COUNTS = {
    9: {0: 113, 1: 307, 2: 300, 3: 1427, 4: 4357, 5: 216, 6: 292, 7: 551, 8: 2026},
    14: {0: 113, 1: 307, 2: 300, 3: 1427, 4: 4357, 5: 216, 6: 292, 7: 551, 8: 2026, 9: 1985, 10: 661, 11: 186, 12: 116, 13: 390},
    17: {0: 113, 1: 307, 2: 300, 3: 1427, 4: 4357, 5: 216, 6: 292, 7: 551, 8: 2026, 9: 1985, 10: 661, 11: 186, 12: 116, 13: 390, 14: 406, 15: 1271, 16: 201, 17: 928}  # TODO update for 17
}


def balanced_softmax(logits, counts_log):
    # counts is a list of counts for each class
    # counts_sum is the sum of counts
    # logits is a tensor of logits of shape (num_class, )
    # Refer to implementation at https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification/blob/main/loss/BalancedSoftmaxLoss.py

    logits_add_log_counts = logits + counts_log
    softmax_logits_add_log_counts = torch.nn.functional.softmax(
        torch.tensor(logits_add_log_counts), dim=0)
    # no need gradient
    softmax_logits_add_log_counts = softmax_logits_add_log_counts.detach()
    return softmax_logits_add_log_counts


def train_one_epoch(
    args,
    curr_epoch,
    model,
    ema_model,
    optimizer,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
    static_teacher,
    tau_global,
    p_class,
    dataset_train
):
    '''
    p_class is a list of float numbers, each of which is the threshold for each class. The size of p_class is num_base_class.
    '''

    train_count_dict = TRAIN_SET_COUNTS[dataset_train.dataset_config.num_base_class]
    counts_list = [train_count_dict[i]
                   for i in range(dataset_train.dataset_config.num_base_class)]
    counts_log = np.log(counts_list)
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=False,
    )

    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    ema_model.train()  # set to train mode for batch norm and dropout to work

    barrier()

    # ramp up weight for consistency loss
    curr_distill_weight_scale = get_current_weight(
        curr_epoch, 1., args.distillation_ramp_len)
    criterion.set_distill_weight_scale(curr_distill_weight_scale)

    # ramp up weight for consistency loss
    curr_consistent_weight_scale = get_current_weight(
        curr_epoch, 1., args.consistency_ramp_len)
    criterion.set_consistency_weight_scale(curr_consistent_weight_scale)

    # log the current weight scale
    if is_primary():
        logger.log_scalars(
            {'distill_weight scale': curr_distill_weight_scale}, curr_iter, prefix="Train_details/")
        print(
            f"Current distill weight scale: {curr_distill_weight_scale:.6f}")

    # log the current weight scale
    if is_primary():
        logger.log_scalars(
            {'consistent_weight scale': curr_consistent_weight_scale}, curr_iter, prefix="Train_details/")
        print(
            f"Current consistent weight scale: {curr_consistent_weight_scale:.6f}")
    num_no_obj = 0
    num_no_base_obj = 0

    # move threshold updates outside the loop, i.e., update tau_global and p_class only once per epoch
    sum_count = 0
    max_prob_dynamic_list = []
    epoch_balanced_softmax_list = []
    for batch_idx, batch_data_label in enumerate(dataset_loader):
        # a numpy array of shape [B, N]
        arr_of_cls_dynamic = batch_data_label['arr_of_cls_dynamic']
        unique_elements, counts = np.unique(
            arr_of_cls_dynamic, return_counts=True)
        for index, count in zip(unique_elements, counts):
            if index < 0:
                continue
            else:
                sum_count += count

        # numpy array of shape [B, N, num_class]
        arr_of_logit_dynamic = batch_data_label['arr_of_logit_dynamic'].cpu(
        ).numpy()
        # numpy array of shape [B, N, num_class]
        arr_of_prob_dynamic = batch_data_label['arr_of_prob_dynamic'].cpu(
        ).numpy()

        # get max prob for each proposal, shape [B, N]
        max_prob_dynamic = np.max(arr_of_prob_dynamic, axis=2)
        # change -1 to 0 in max_prob_dynamic
        max_prob_dynamic[max_prob_dynamic < 0] = 0
        # append max_prob_dynamic to max_prob_dynamic_list
        max_prob_dynamic_list.append(np.sum(max_prob_dynamic))

        if sum_count > 0:
            # update tau_global
            # tau_global = args.ema_decay * tau_global + (1 - args.ema_decay) * np.sum(max_prob_dynamic) / sum_count
            # update p_class

            # loop through arr_of_logit_dynamic of shape [B, N, num_class] along B and N
            # if the arr_of_logit_dynamic[i, j] are not all 0, compute balanced_softmax(logits, counts_log), else skip
            combined_balanced_softmax = []
            for i in range(arr_of_logit_dynamic.shape[0]):
                for j in range(arr_of_logit_dynamic.shape[1]):
                    if np.all(arr_of_logit_dynamic[i, j] == 0):
                        continue
                    else:
                        # compute balanced_softmax(logits, counts_log)
                        balanced_softmax_logits = balanced_softmax(
                            arr_of_logit_dynamic[i, j], counts_log)
                        combined_balanced_softmax.append(
                            balanced_softmax_logits)

            # convert a list of tensors to a tensor
            combined_balanced_softmax = torch.stack(
                combined_balanced_softmax, dim=0)

            # append combined_balanced_softmax to epoch_balanced_softmax_list
            epoch_balanced_softmax_list.append(combined_balanced_softmax)

            # change -1 to 0 in arr_of_logit_dynamic
            # arr_of_logit_dynamic[arr_of_logit_dynamic < 0] = 0
            # arr_of_logit_dynamic[arr_of_logit_dynamic < 0] = 0

            # sum over all batches and proposals for each class
            # sum_prob_dynamic = np.sum(arr_of_logit_dynamic, axis=(0, 1)) # shape [num_class], equation (6) in the paper.
            # This assumes that the number of proposals for each class is the same, which is not true.

            # so weight sum_prob_dynamic by ratios of count of each class divided by total count, which are known and saved in dataset_train
            # sum_prob_dynamic /= dataset_train.train_set_count_ratios # TODO check if the magnitude works
            # normalize by sum_count
            # sum_prob_dynamic = sum_prob_dynamic / sum_count

            # update p_class
            # p_class = args.ema_decay * p_class + (1 - args.ema_decay) * sum_prob_dynamic

            # p_class = [args.ema_decay * p_class[i] + (1 - args.ema_decay) * sum_prob_dynamic[i] for i in range(dataset_config.num_base_class)]
            # normalize p_class by max value
            # p_class = p_class / np.max(p_class)

            # update p_class by multiplying with tau_global
            # p_class = p_class * tau_global

            # update p_class in dataset['train']
            # dataset_train.update_dynamic_base_pseudo_thresholds_list(p_class)

        else:
            print('sum_count is 0!')

        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        for key in batch_data_label:
            # do not move scanname, no_obj and no_cls_obj to GPU as they are used for debugging
            if key not in ['no_obj', 'no_base_obj']:
                batch_data_label[key] = batch_data_label[key].to(net_device)
        # scan_names = batch_data_label['scan_name']
        # # filter scan_names by batch_data_label['no_obj'] which are 0s and 1s
        # scan_names_no_obj = [scan_names[i] for i in range(
        #     len(scan_names)) if batch_data_label['no_obj'][i] == 1]
        # # filter scan_names by batch_data_label['no_base_obj'] which are 0s and 1s
        # scan_names_no_base_obj = [scan_names[i] for i in range(
        #     len(scan_names)) if batch_data_label['no_base_obj'][i] == 1]
        # log scan_names_no_obj
        # torch.sum(batch_data_label['no_obj'])
        num_no_obj += sum(batch_data_label['no_obj'])
        # torch.sum(batch_data_label['no_base_obj'])
        num_no_base_obj += sum(batch_data_label['no_base_obj'])

        #     logger.log_scalars(
        #         {'scan_names_no_obj': scan_names_no_obj}, curr_iter, prefix="Train_details/")
        # # log scan_names_no_base_obj
        #     logger.log_scalars(
        #         {'scan_names_no_base_obj': scan_names_no_base_obj}, curr_iter, prefix="Train_details/")
        # Forward pass
        optimizer.zero_grad()
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        outputs, query_xyz, pos_embed, enc_inds, interim_inds, *_ = model(
            inputs)

        # base_cls_probs = outputs['outputs']['sem_cls_prob'][...,:dataset_config.num_base_class]
        # # base_cls_probs has shape [B, N, num_base_class]
        # max_base_cls_probs, _ = torch.max(base_cls_probs, dim=2) # shape [B, N]

        # # select top 5% points with highest base class probability
        # # max_base_cls_probs has shape [B, N]
        # max_base_cls_probs_top_10_pct = torch.topk(max_base_cls_probs, int(0.05 * max_base_cls_probs.shape[1]), dim=1, largest=True, sorted=True)[0] # shape [B, int(0.1 * N)]

        # # tau_batch
        # tau_global_batch = torch.sum(max_base_cls_probs_top_10_pct) / (max_base_cls_probs_top_10_pct.shape[0] * max_base_cls_probs_top_10_pct.shape[1])
        # # update tau_global
        # tau_global = args.ema_decay * tau_global + (1 - args.ema_decay) * tau_global_batch

        # # # update max_pred_prob_list
        # # for i in range(dataset_config.num_base_class):
        # #     if max_base_cls_probs[i] > max_pred_prob_list[i]:
        # #         max_pred_prob_list[i] = max_base_cls_probs[i]

        # # avg
        # avg_base_cls_probs = torch.mean(base_cls_probs, dim=1) # shape [B, num_base_class]
        # avg_base_cls_probs = torch.mean(avg_base_cls_probs, dim=0) # shape [num_base_class]

        # # normalize by max avg
        # avg_base_cls_probs_norm = avg_base_cls_probs / torch.max(avg_base_cls_probs)

        # # to list

        # # update p_class
        # p_class = args.ema_decay * p_class + (1 - args.ema_decay) * avg_base_cls_probs_norm

        # # update avg_pred_prob_list
        # for i in range(dataset_config.num_base_class):
        #     avg_pred_prob_list[i].append(avg_base_cls_probs[i].item())

        # get the predicted probability for each class if the class is a base class
        # outputs['outputs']['sem_cls_prob'] is [B, N, num_class]
        # batch_data_label['sem_cls_label'] is [B, N]
        # batch_data_label['no_obj'] is [B]

        # Add augmentation related information to outputs to facilitate consistency loss computation.
        outputs['outputs']['flip_x_axis'] = batch_data_label['flip_x_axis']
        outputs['outputs']['flip_y_axis'] = batch_data_label['flip_y_axis']
        outputs['outputs']['rot_mat'] = batch_data_label['rot_mat']

        # outputs = {'outputs': {'sem_cls_logits', 'center_normalized', 'center_unnormalized', 'size_normalized', 'size_unnormalized',
        # 'angle_logits', 'angle_residual', 'angle_residual_normalized', 'angle_continuous', 'objectness_prob', 'sem_cls_prob', 'box_corners'}
        # 'aux_outputs':[same as outputs], length = num_decoder_layers - 1 (default: 8-1 = 7), and the last one is the final output}
        #

        # query_xyz: [B, N, 3], the query points for the decoder, N =256 which is the number of proposals/nqueries
        # pos_embed: [B, N, N], the positional embedding for the decoder, N =256 which is the number of proposals/nqueries
        # enc_inds: [B, N_1], the indices of the encoder points for the decoder, N_1 = 2048 which is the number of encoder points after pre-encoder
        # interim_inds: [B, N_2], the indices of the encoder points for the decoder, N_2 = 1024 which is downsampled from N_1 = 2048 by half.

        # query_xyz and pos_embed are for static teacher to use the same query points and pos_embed as the student
        # end_inds is for the static teacher to use the same points after pre-encoder as the student
        # interim_inds is for the static teacher to use the same points after encoder as the student (optional)
        # because vallina transformer does not have point downsampling in the encoder, in which interim_inds is the same as enc_inds.
        # also set the static teacher to train mode to make sure the batchnorm is updated
        static_teacher.train()
        # no gradient for static_teacher
        with torch.no_grad():
            # feed pos_embed to static_teacher; query_xyz and pos_embed are not used
            outputs_static, *_ = static_teacher(
                inputs=inputs, query_xyz=query_xyz, pos_embed=pos_embed, student_inds=enc_inds, interim_inds=interim_inds)
            # interim_inds are 0, 1, ..., 1023

        # check if pos_embed and pos_embed_static are the same
        # print(torch.all(torch.eq(pos_embed, pos_embed_static)))

        # EMA forward pass
        ema_model.train()  # set to train mode for batch norm and dropout to work
        ema_inputs = {
            "point_clouds": batch_data_label["ema_point_clouds"],
            "point_cloud_dims_min": batch_data_label["ema_point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["ema_point_cloud_dims_max"],
        }
        ema_outputs, *_ = ema_model(ema_inputs)

        # Compute loss
        loss, loss_dict = criterion(
            outputs, ema_outputs, batch_data_label, outputs_static)
        loss_reduced = all_reduce_average(loss)
        loss_dict_reduced = reduce_dict(loss_dict)
        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)

        loss.backward()
        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip_gradient)
        optimizer.step()

        if curr_iter % args.log_metrics_every == 0:
            # This step is slow. AP is computed approximately and locally during training.
            # It will gather outputs and ground truth across all ranks.
            # It is memory intensive as point_cloud ground truth is a large tensor.
            # If GPU memory is not an issue, uncomment the following lines.
            # outputs["outputs"] = all_gather_dict(outputs["outputs"])
            # batch_data_label = all_gather_dict(batch_data_label)
            batch_pred_map_cls, batch_gt_map_cls = ap_calculator.step_meter(
                outputs, batch_data_label)
            # logger.log_scalars({'tau_global': tau_global},
            #                    curr_iter, prefix="Train_details/")
            # for idx, phi in enumerate(p_class):
            #     logger.log_scalars({f'phi_{idx}': float(phi)},
            #                        curr_iter, prefix="Train/")
            # print(
            #     f"Iter {curr_iter}; Tau {tau_global:.4f}; Phi {[f'{_phi:.4f}' for _phi in p_class]}")
        time_delta.update(time.time() - curr_time)
        loss_avg.update(loss_reduced.item())

        # logging
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            eta_seconds = (max_iters - curr_iter) * time_delta.avg
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            log_every_times_batchsize = args.log_every * args.batchsize_per_gpu
            no_obj_rate = num_no_obj / log_every_times_batchsize
            no_base_obj_rate = num_no_base_obj / log_every_times_batchsize
            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; No_obj: {no_obj_rate:.3f}; No_base_obj: {no_base_obj_rate:.3f} Loss {loss_avg.avg:0.2f}; Distill {loss_dict_reduced['loss_distill']:.3f}; Center_con {loss_dict_reduced['loss_center_consistency']:.3f}; Cls_con {loss_dict_reduced['loss_cls_consistency']:.3f}; Size_con {loss_dict_reduced['loss_size_consistency']:.3f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
            )
            logger.log_scalars(loss_dict_reduced, curr_iter,
                               prefix="Train_details/")

            train_dict = {}
            train_dict["lr"] = curr_lr
            train_dict["memory"] = mem_mb
            train_dict["loss"] = loss_avg.avg
            train_dict["batch_time"] = time_delta.avg
            train_dict['no_obj_rate'] = no_obj_rate
            train_dict['no_base_obj_rate'] = no_base_obj_rate

            # log losses
            # "loss_sem_cls": self.loss_sem_cls,
            # "loss_angle": self.loss_angle,
            # "loss_center": self.loss_center,
            # "loss_size": self.loss_size,
            # "loss_giou": self.loss_giou,
            # "loss_distill": self.loss_distill_aux,
            # "loss_center_consistency": self.loss_center_consistency_all,
            # 'loss_cls_consistency': self.loss_class_consistency_all,
            # 'loss_size_consistency': self.loss_size_consistency_all,

            # note that they are for final outputs (not aux_outputs although they are also computed and saved in loss_dict_reduced)
            train_dict['loss_sem_cls'] = loss_dict_reduced['loss_sem_cls']
            train_dict['loss_angle_cls'] = loss_dict_reduced['loss_angle_cls']
            train_dict['loss_angle_reg'] = loss_dict_reduced['loss_angle_reg']
            train_dict['loss_center'] = loss_dict_reduced['loss_center']
            train_dict['loss_size'] = loss_dict_reduced['loss_size']
            train_dict['loss_giou'] = loss_dict_reduced['loss_giou']
            train_dict['loss_distill'] = loss_dict_reduced['loss_distill']
            train_dict['loss_center_consistency'] = loss_dict_reduced['loss_center_consistency']
            train_dict['loss_cls_consistency'] = loss_dict_reduced['loss_cls_consistency']
            train_dict['loss_size_consistency'] = loss_dict_reduced['loss_size_consistency']

            logger.log_scalars(train_dict, curr_iter, prefix="Train/")

            # reset num_no_obj and num_no_base_obj
            num_no_obj = 0
            num_no_base_obj = 0

        curr_iter += 1

        # Update EMA model
        update_ema_variables(model, ema_model, args.ema_decay, curr_iter)
        barrier()

    # update tau_global and p_class
    tau_global = args.ema_decay_dt * tau_global + \
        (1 - args.ema_decay_dt) * np.sum(max_prob_dynamic_list) / sum_count
    # update p_class
    # epoch_balanced_softmax_list is a list of tensors of shape [num_object, num_class]
    # combine into a tensor
    epoch_balanced_softmax = torch.cat(
        epoch_balanced_softmax_list, dim=0)  # shape [num_object, num_class]
    # sum over all objects for each class
    sum_prob_dynamic = torch.sum(epoch_balanced_softmax, dim=0)
    # normalize by sum_count
    sum_prob_dynamic = sum_prob_dynamic / sum_count
    # to np
    sum_prob_dynamic = sum_prob_dynamic.cpu().numpy()
    # update p_class
    p_class = args.ema_decay_dt * p_class + (1 - args.ema_decay_dt) * sum_prob_dynamic
    # normalize p_class by max value
    p_class = p_class / np.max(p_class)

    # update p_class by multiplying with tau_global
    p_class = p_class * tau_global

    # update p_class in dataset['train']
    dataset_train.update_dynamic_base_pseudo_thresholds_list(p_class)

    if curr_iter % args.log_metrics_every == 0:
        logger.log_scalars({'tau_global': tau_global},
                           curr_iter, prefix="Dynamic_Threshold/")
        for idx, phi in enumerate(p_class):
            logger.log_scalars({f'phi_{idx}': float(phi)},
                               curr_iter, prefix="Dynamic_Threshold/")
        print(
            f"Iter {curr_iter}; Tau {tau_global:.4f}; Phi {[f'{_phi:.4f}' for _phi in p_class]}")

    return ap_calculator, tau_global, p_class
    # average the avg_pred_prob_list
    # avg_pred_prob_list = [sum(avg_pred_prob_list[i]) / len(avg_pred_prob_list[i]) for i in range(dataset_config.num_base_class)]

    # use max norm to normalize avg_pred_prob_list
    # max_avg_pred_prob = max(avg_pred_prob_list)
    # avg_pred_prob_list_norm = [avg_pred_prob_list[i] / max_avg_pred_prob for i in range(dataset_config.num_base_class)]
    # return ap_calculator, avg_pred_prob_list_norm, max_pred_prob_list # avg_pred_prob_list_norm is a list of float numbers, max_pred_prob_list is a list of cuda tensors


@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
    curr_train_iter,
):

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        # query_xyz and pos_embed are not used
        outputs, _, _, _, _ = model(inputs)

        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)

            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        batch_data_label = all_gather_dict(batch_data_label)
        ap_calculator.step_meter(outputs, batch_data_label)
        time_delta.update(time.time() - curr_time)
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            if criterion is not None:
                test_dict["loss"] = loss_avg.avg
        curr_iter += 1
        barrier()
    if is_primary():
        if criterion is not None:
            logger.log_scalars(
                loss_dict_reduced, curr_train_iter, prefix="Test_details/"
            )
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")

    return ap_calculator


@torch.no_grad()
def evaluate_incremental(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
    curr_train_iter,
    test_prefix="Test",
):

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        # query_xyz, pos_embed and enc_inds are not used
        outputs, *_ = model(inputs)

        # Compute loss, skipped for faster evaluation
        loss_str = ""
        # if criterion is not None:
        #     # loss, loss_dict = criterion(outputs, batch_data_label)
        #     # Compute loss
        #     loss, loss_dict = criterion(outputs=outputs, ema_outputs=ema_outputs,
        #                                 targets=batch_data_label)

        #     loss_reduced = all_reduce_average(loss)
        #     loss_dict_reduced = reduce_dict(loss_dict)
        #     loss_avg.update(loss_reduced.item())
        #     loss_str = f"Loss {loss_avg.avg:0.2f};"

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        batch_data_label = all_gather_dict(batch_data_label)
        ap_calculator.step_meter(outputs, batch_data_label)
        time_delta.update(time.time() - curr_time)
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"{test_prefix} Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            # if criterion is not None:
            #     test_dict["loss"] = loss_avg.avg
        curr_iter += 1
        barrier()
    if is_primary():
        # if criterion is not None:
        # logger.log_scalars(
        #     loss_dict_reduced, curr_train_iter, prefix="Test_details/"
        # )
        logger.log_scalars(test_dict, curr_train_iter, prefix="{test_prefix}/")

    return ap_calculator
