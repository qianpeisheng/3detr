# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.box_util import generalized_box3d_iou
from utils.dist import all_reduce_average
from utils.misc import huber_loss
from scipy.optimize import linear_sum_assignment

from utils.nn_distance import nn_distance, huber_loss


class Matcher(nn.Module):
    def __init__(self, cost_class, cost_objectness, cost_giou, cost_center):
        """
        Parameters:
            cost_class:
        Returns:

        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_objectness = cost_objectness
        self.cost_giou = cost_giou
        self.cost_center = cost_center

    @torch.no_grad()
    def forward(self, outputs, targets):

        batchsize = outputs["sem_cls_prob"].shape[0]
        nqueries = outputs["sem_cls_prob"].shape[1]
        ngt = targets["gt_box_sem_cls_label"].shape[1]
        nactual_gt = targets["nactual_gt"]

        # classification cost: batch x nqueries x ngt matrix
        pred_cls_prob = outputs["sem_cls_prob"]
        gt_box_sem_cls_labels = (
            targets["gt_box_sem_cls_label"]
            .unsqueeze(1)
            .expand(batchsize, nqueries, ngt)
        )
        class_mat = -torch.gather(pred_cls_prob, 2, gt_box_sem_cls_labels)

        # objectness cost: batch x nqueries x 1
        objectness_mat = -outputs["objectness_prob"].unsqueeze(-1)

        # center cost: batch x nqueries x ngt
        center_mat = outputs["center_dist"].detach()

        # giou cost: batch x nqueries x ngt
        giou_mat = -outputs["gious"].detach()

        final_cost = (
            self.cost_class * class_mat
            + self.cost_objectness * objectness_mat
            + self.cost_center * center_mat
            + self.cost_giou * giou_mat
        )

        final_cost = final_cost.detach().cpu().numpy()
        assignments = []

        # auxiliary variables useful for batched loss computation
        batch_size, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=pred_cls_prob.device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=pred_cls_prob.device
        )
        for b in range(batchsize):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(
                    final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=pred_cls_prob.device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }


class SetCriterion(nn.Module):
    def __init__(self, matcher, dataset_config, loss_weight_dict):
        super().__init__()
        self.dataset_config = dataset_config
        self.matcher = matcher
        self.loss_weight_dict = loss_weight_dict

        semcls_percls_weights = torch.ones(dataset_config.num_semcls + 1)
        semcls_percls_weights[-1] = loss_weight_dict["loss_no_object_weight"]
        del loss_weight_dict["loss_no_object_weight"]
        self.register_buffer("semcls_percls_weights", semcls_percls_weights)

        self.loss_functions = {
            "loss_sem_cls": self.loss_sem_cls,
            "loss_angle": self.loss_angle,
            "loss_center": self.loss_center,
            "loss_size": self.loss_size,
            "loss_giou": self.loss_giou,
            # this isn't used during training and is logged for debugging.
            # thus, this loss does not have a loss_weight associated with it.
            "loss_cardinality": self.loss_cardinality,
            "loss_distill": self.loss_distill_aux,
            "loss_center_consistency": self.loss_center_consistency_all,
            'loss_cls_consistency': self.loss_class_consistency_all,
            'loss_size_consistency': self.loss_size_consistency_all,
        }

    def set_distill_weight_scale(self, distill_weight_scale):
        self.distill_weight_scale = distill_weight_scale

    def set_consistency_weight_scale(self, consistency_weight_scale):
        self.consistency_weight_scale = consistency_weight_scale

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, assignments):
        # Count the number of predictions that are objects
        # Cardinality is the error between predicted #objects and ground truth objects

        pred_logits = outputs["sem_cls_logits"]
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        pred_objects = (pred_logits.argmax(-1) !=
                        pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(pred_objects.float(), targets["nactual_gt"])
        return {"loss_cardinality": card_err}

    def loss_center_consistency_all(self, outputs, ema_outputs):
        # similar to distillation loss, mask out where the no-object class is the max.
        # following the 3DIoUMatch implementation

        # output_cls_logits is of shape [batch, nprop, nclass + 1] (last class is no-object)
        # The last dimension is the logits for each class.
        # Create a torch binary mask of shape [batch, nprop] for output_cls_logits where the no-objcect class is the max.

        # binary_mask_output = (outputs["sem_cls_logits"].argmax(-1) !=
        #                       outputs["sem_cls_logits"].shape[-1] - 1)

        # sum the binary mask along the second dimension to get the number of objects in each batch
        # nprop = binary_mask_output.sum(1)

        # Create a torch binary mask of shape [batch, nprop] for target_cls_logits where the no-objcect class is the max.
        # binary_mask_ema_output = (ema_outputs["sem_cls_logits"].argmax(-1) !=
        #                           ema_outputs["sem_cls_logits"].shape[-1] - 1)

        # follow the SDCoT implementation
        # center = outputs["center_normalized"].clone()
        # ema_center = ema_outputs["center_normalized"].clone()
        center = outputs["center_normalized"]  # shape (B, num_proposal, 3)
        # shape (B, num_proposal, 3)
        ema_center = ema_outputs["center_normalized"]

        # filter center with binary_mask_output, centeris of shape (B, num_proposal, 3), binary_mask_output is of shape (B, num_proposal)
        # the filtering should be done on both the first and second dimension.
        # center *= binary_mask_output.unsqueeze(-1)
        # ema_center *= binary_mask_ema_output.unsqueeze(-1)

        # # keep only the non-zero elements and keep the dimension
        # center = center[center != 0].view(center.shape[0], -1, 3)

        # masked center and ema_center do not participate in the gradient computation

        # show min, max, mean, std of dist1 and dist2
        dist1, ind1, dist2, ind2 = nn_distance(center, ema_center)

        # create a binary mask for dist2 where dist2[i] < dist2[i].mean() - dist2[i].std()
        masked_dist2 = dist2.clone().detach()
        for i in range(masked_dist2.shape[0]):
            _threshold_2 = dist2[i].min() + 0.2*(dist2[i].max() - dist2[i].min())
            masked_dist2[i][dist2[i] > _threshold_2] = False
            masked_dist2[i][dist2[i] <= _threshold_2] = True

        # masked_dist2 = [ind2[i][dist2[i] < dist2[i].mean() - dist2[i].std()]
        #              for i in range(ind2.shape[0])]

        # average dist1 only where dist1[i] < dist1[i].mean() - dist1[i].std()
        _threshold_1 = dist1.min() + 0.2*(dist1.max() - dist1.min())
        dist1_list = [dist1[i][dist1[i] <= _threshold_1]
                      for i in range(dist1.shape[0])]
        dist1 = torch.cat(dist1_list)
        dist1 = torch.mean(dist1)

        # average dist2 only where dist2[i] < dist2[i].mean() - dist2[i].std()
        dist2_list = [dist2[i][dist2[i] <= _threshold_2]
                      for i in range(dist2.shape[0])]
        dist2 = torch.cat(dist2_list)
        dist2 = torch.mean(dist2)
        # dist1_list = []
        # dist2_list = []
        # ind1_list = []
        # masked_dist2 = []
        # for i in range(center.shape[0]):
        #     # maks center[i] with binary_mask_output[i]
        #     # center[i] is of shape (num_proposal, 3)
        #     # binary_mask_output[i] is of shape (num_proposal)
        #     center_i = center[i].unsqueeze(0)
        #     center_i_masked = center_i[:, binary_mask_output[i]]
        #     ema_center_i = ema_center[i].unsqueeze(0)
        #     ema_center_i_masked = ema_center_i[:, binary_mask_ema_output[i]]
        #     dist1, ind1, dist2, ind2 = nn_distance(
        #         center_i_masked, ema_center_i_masked)
        #     dist1_list.append(dist1)
        #     dist2_list.append(dist2)
        #     ind1_list.append(ind1)
        #     masked_dist2.append(ind2)

        # dist1 is the sum of dist1_list
        # dist2 is the sum of dist2_list

        # ind1 is (B, num_proposal): ema_center index closest to center
        # ind2 is (B, num_proposal): center index closest to ema_center

        # TODO: use both dist1 and dist2 or only use dist1
        # dist = dist1 + dist2
        # return torch.mean(dist), ind2
        # return {"loss_center_consistency": torch.mean(dist)}, ind2, masked_dist2
        return {"loss_center_consistency": dist1 + dist2}, ind2, masked_dist2

    def loss_class_consistency_all(self, outputs, ema_outputs, map_ind, masked_dist2):

        # follow the SDCoT implementation
        sem_cls_logits = outputs["sem_cls_logits"]

        # sem_cls_prob_log = F.log_softmax(
        #     sem_cls_logits, dim=-1)  # for kl_div loss

        ema_cls_logits = ema_outputs["sem_cls_logits"]

        # ema_sem_cls_prob = F.softmax(ema_cls_logits, dim=-1)

        cls_logits_aligned = torch.cat([torch.index_select(
            a, 0, i).unsqueeze(0) for a, i in zip(sem_cls_logits, map_ind)])

        # cls_log_prob_aligned = torch.cat([torch.index_select(
        #     a, 0, i).unsqueeze(0) for a, i in zip(sem_cls_prob_log, map_ind)])

        # class_consistency_loss = F.kl_div(
        #     cls_log_prob_aligned, ema_sem_cls_prob, reduction='mean')

        # filter by masked_dist2
        cls_logits_aligned = cls_logits_aligned * masked_dist2.unsqueeze(-1)
        ema_cls_logits = ema_cls_logits * masked_dist2.unsqueeze(-1)
        # class_consistency_loss = F.mse_loss(cls_log_prob_aligned, ema_cls_prob)
        # exclude the last class (no-object class)
        class_consistency_loss = F.mse_loss(
            cls_logits_aligned[..., :-1], ema_cls_logits[..., :-1])

        # return class_consistency_loss*2
        return {"loss_cls_consistency": class_consistency_loss}

    # def loss_class_consistency_all(self, outputs, ema_outputs, map_ind, masked_dist2):

    #     # follow the SDCoT implementation
    #     sem_cls_logits = outputs["sem_cls_logits"]

    #     sem_cls_prob_log = F.log_softmax(
    #         sem_cls_logits, dim=-1)  # for kl_div loss

    #     ema_cls_logits = ema_outputs["sem_cls_logits"]

    #     ema_sem_cls_prob = F.softmax(ema_cls_logits, dim=-1)

    #     cls_log_prob_aligned = torch.cat([torch.index_select(
    #         a, 0, i).unsqueeze(0) for a, i in zip(sem_cls_prob_log, map_ind)])

    #     class_consistency_loss = F.kl_div(
    #         cls_log_prob_aligned, ema_sem_cls_prob, reduction='mean')
    #     # class_consistency_loss = F.mse_loss(cls_log_prob_aligned, ema_cls_prob)

    #     # return class_consistency_loss*2
    #     return {"loss_cls_consistency": class_consistency_loss}

    def loss_size_consistency_all(self, outputs, ema_outputs, map_ind, masked_dist2):
        # follow the SDCoT implementation
        size_normalized = outputs["size_normalized"]
        ema_size_normalized = ema_outputs["size_normalized"]

        size_aligned = torch.cat([torch.index_select(
            a, 0, i).unsqueeze(0) for a, i in zip(size_normalized, map_ind)])

        # filter by masked_dist2
        size_aligned = size_aligned * masked_dist2.unsqueeze(-1)
        ema_size_normalized = ema_size_normalized * masked_dist2.unsqueeze(-1)

        size_consistency_loss = F.mse_loss(
            size_aligned, ema_size_normalized, reduction='mean')

        # return size_consistency_loss
        return {"loss_size_consistency": size_consistency_loss}

    def loss_distill_aux(self, outputs, targets, assignments=None):
        # TODO try to include background class in distillation loss
        # distillation loss

        # TODO need to align this!
        # output_cls_logits = outputs["sem_cls_logits"][..., :targets["outputs"]["sem_cls_logits"].shape[-1] - 1]
        # target_cls_logits = targets["outputs"]["sem_cls_logits"][..., :-1]

        # include background class in distillation loss
        # in this version, ["outputs"] or ["aux_outputs"] is handled by the caller.

        logits_indices = list(
            range(targets["sem_cls_logits"].shape[-1] - 1))
        logits_indices.append(-1)
        output_cls_logits = outputs["sem_cls_logits"][..., logits_indices]
        target_cls_logits = targets["sem_cls_logits"]

        # output_cls_logits is of shape [batch, nprop, nclass_base + 1] (last class is no-object)
        # The last dimension is the logits for each class.
        # Create a torch binary mask of shape [batch, nprop] for output_cls_logits where the no-objcect class is the max.
        binary_mask_output = (output_cls_logits.argmax(-1) !=
                              output_cls_logits.shape[-1] - 1)

        # Create a torch binary mask of shape [batch, nprop] for target_cls_logits where the no-objcect class is the max.
        binary_mask_target = (target_cls_logits.argmax(-1) !=
                              target_cls_logits.shape[-1] - 1)

        # combine the two masks. Either one of them is True -> the combined mask is True.
        binary_mask = binary_mask_output | binary_mask_target

        # count the number of True values in the combined mask.
        # nprop = binary_mask.sum()

        # remove the no-object class from the logits, because their magnitudes are very large.
        output_cls_logits = output_cls_logits[..., :-1]
        target_cls_logits = target_cls_logits[..., :-1]

        # normalize logits
        output_cls_logits = output_cls_logits - \
            torch.mean(output_cls_logits, dim=-1,
                       keepdim=True)  # shape: [batch, nprop, nclass_base]
        target_cls_logits = target_cls_logits - \
            torch.mean(target_cls_logits, dim=-1,
                       keepdim=True)  # shape: [batch, nprop, nclass_base]

        # mask the logits with the combined mask.
        output_cls_logits = output_cls_logits[binary_mask]
        target_cls_logits = target_cls_logits[binary_mask]

        distill_loss = F.mse_loss(
            output_cls_logits, target_cls_logits)
        # -1 because last class is no-object
        return {"loss_distill": distill_loss}

    def loss_distill(self, outputs, targets, assignments=None):
        # TODO try to include background class in distillation loss
        # distillation loss

        # TODO need to align this!
        # output_cls_logits = outputs["sem_cls_logits"][..., :targets["outputs"]["sem_cls_logits"].shape[-1] - 1]
        # target_cls_logits = targets["outputs"]["sem_cls_logits"][..., :-1]

        # include background class in distillation loss
        logits_indices = list(
            range(targets["outputs"]["sem_cls_logits"].shape[-1] - 1))
        logits_indices.append(-1)
        output_cls_logits = outputs["sem_cls_logits"][..., logits_indices]
        target_cls_logits = targets["outputs"]["sem_cls_logits"]
        # normalize logits
        output_cls_logits = output_cls_logits - \
            torch.mean(output_cls_logits, dim=-1, keepdim=True)
        target_cls_logits = target_cls_logits - \
            torch.mean(target_cls_logits, dim=-1, keepdim=True)
        distill_loss = F.mse_loss(
            output_cls_logits, target_cls_logits)
        # -1 because last class is no-object
        return {"loss_distill": distill_loss}

    def loss_sem_cls(self, outputs, targets, assignments):

        # # Not vectorized version
        # pred_logits = outputs["sem_cls_logits"]
        # assign = assignments["assignments"]

        # sem_cls_targets = torch.ones((pred_logits.shape[0], pred_logits.shape[1]),
        #                         dtype=torch.int64, device=pred_logits.device)

        # # initialize to background/no-object class
        # sem_cls_targets *= (pred_logits.shape[-1] - 1)

        # # use assignments to compute labels for matched boxes
        # for b in range(pred_logits.shape[0]):
        #     if len(assign[b]) > 0:
        #         sem_cls_targets[b, assign[b][0]] = targets["gt_box_sem_cls_label"][b, assign[b][1]]

        # sem_cls_targets = sem_cls_targets.view(-1)
        # pred_logits = pred_logits.reshape(sem_cls_targets.shape[0], -1)
        # loss = F.cross_entropy(pred_logits, sem_cls_targets, self.semcls_percls_weights, reduction="mean")

        pred_logits = outputs["sem_cls_logits"]
        gt_box_label = torch.gather(
            targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
            pred_logits.shape[-1] - 1
        )
        loss = F.cross_entropy(
            pred_logits.transpose(2, 1),
            gt_box_label,
            self.semcls_percls_weights,
            reduction="mean",
        )

        return {"loss_sem_cls": loss}

    def loss_angle(self, outputs, targets, assignments):
        angle_logits = outputs["angle_logits"]
        angle_residual = outputs["angle_residual_normalized"]

        if targets["num_boxes_replica"] > 0:
            gt_angle_label = targets["gt_angle_class_label"]
            gt_angle_residual = targets["gt_angle_residual_label"]
            gt_angle_residual_normalized = gt_angle_residual / (
                np.pi / self.dataset_config.num_angle_bin
            )

            # # Non vectorized version
            # assignments = assignments["assignments"]
            # p_angle_logits = []
            # p_angle_resid = []
            # t_angle_labels = []
            # t_angle_resid = []

            # for b in range(angle_logits.shape[0]):
            #     if len(assignments[b]) > 0:
            #         p_angle_logits.append(angle_logits[b, assignments[b][0]])
            #         p_angle_resid.append(angle_residual[b, assignments[b][0], gt_angle_label[b][assignments[b][1]]])
            #         t_angle_labels.append(gt_angle_label[b, assignments[b][1]])
            #         t_angle_resid.append(gt_angle_residual_normalized[b, assignments[b][1]])

            # p_angle_logits = torch.cat(p_angle_logits)
            # p_angle_resid = torch.cat(p_angle_resid)
            # t_angle_labels = torch.cat(t_angle_labels)
            # t_angle_resid = torch.cat(t_angle_resid)

            # angle_cls_loss = F.cross_entropy(p_angle_logits, t_angle_labels, reduction="sum")
            # angle_reg_loss = huber_loss(p_angle_resid.flatten() - t_angle_resid.flatten()).sum()

            gt_angle_label = torch.gather(
                gt_angle_label, 1, assignments["per_prop_gt_inds"]
            )
            angle_cls_loss = F.cross_entropy(
                angle_logits.transpose(2, 1), gt_angle_label, reduction="none"
            )
            angle_cls_loss = (
                angle_cls_loss * assignments["proposal_matched_mask"]
            ).sum()

            gt_angle_residual_normalized = torch.gather(
                gt_angle_residual_normalized, 1, assignments["per_prop_gt_inds"]
            )
            gt_angle_label_one_hot = torch.zeros_like(
                angle_residual, dtype=torch.float32
            )
            gt_angle_label_one_hot.scatter_(2, gt_angle_label.unsqueeze(-1), 1)

            angle_residual_for_gt_class = torch.sum(
                angle_residual * gt_angle_label_one_hot, -1
            )
            angle_reg_loss = huber_loss(
                angle_residual_for_gt_class - gt_angle_residual_normalized, delta=1.0
            )
            angle_reg_loss = (
                angle_reg_loss * assignments["proposal_matched_mask"]
            ).sum()

            angle_cls_loss /= targets["num_boxes"]
            angle_reg_loss /= targets["num_boxes"]
        else:
            angle_cls_loss = torch.zeros(
                1, device=angle_logits.device).squeeze()
            angle_reg_loss = torch.zeros(
                1, device=angle_logits.device).squeeze()
        return {"loss_angle_cls": angle_cls_loss, "loss_angle_reg": angle_reg_loss}

    def loss_center(self, outputs, targets, assignments):
        center_dist = outputs["center_dist"]
        if targets["num_boxes_replica"] > 0:

            # # Non vectorized version
            # assign = assignments["assignments"]
            # center_loss = torch.zeros(1, device=center_dist.device).squeeze()
            # for b in range(center_dist.shape[0]):
            #     if len(assign[b]) > 0:
            #         center_loss += center_dist[b, assign[b][0], assign[b][1]].sum()

            # select appropriate distances by using proposal to gt matching
            center_loss = torch.gather(
                center_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            # zero-out non-matched proposals
            center_loss = center_loss * assignments["proposal_matched_mask"]
            center_loss = center_loss.sum()

            if targets["num_boxes"] > 0:
                center_loss /= targets["num_boxes"]
        else:
            center_loss = torch.zeros(1, device=center_dist.device).squeeze()

        return {"loss_center": center_loss}

    def loss_giou(self, outputs, targets, assignments):
        gious_dist = 1 - outputs["gious"]

        # # Non vectorized version
        # giou_loss = torch.zeros(1, device=gious_dist.device).squeeze()
        # assign = assignments["assignments"]

        # for b in range(gious_dist.shape[0]):
        #     if len(assign[b]) > 0:
        #         giou_loss += gious_dist[b, assign[b][0], assign[b][1]].sum()

        # select appropriate gious by using proposal to gt matching
        giou_loss = torch.gather(
            gious_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
        ).squeeze(-1)
        # zero-out non-matched proposals
        giou_loss = giou_loss * assignments["proposal_matched_mask"]
        giou_loss = giou_loss.sum()

        if targets["num_boxes"] > 0:
            giou_loss /= targets["num_boxes"]

        return {"loss_giou": giou_loss}

    def loss_size(self, outputs, targets, assignments):
        gt_box_sizes = targets["gt_box_sizes_normalized"]
        pred_box_sizes = outputs["size_normalized"]

        if targets["num_boxes_replica"] > 0:

            # # Non vectorized version
            # p_sizes = []
            # t_sizes = []
            # assign = assignments["assignments"]
            # for b in range(pred_box_sizes.shape[0]):
            #     if len(assign[b]) > 0:
            #         p_sizes.append(pred_box_sizes[b, assign[b][0]])
            #         t_sizes.append(gt_box_sizes[b, assign[b][1]])
            # p_sizes = torch.cat(p_sizes)
            # t_sizes = torch.cat(t_sizes)
            # size_loss = F.l1_loss(p_sizes, t_sizes, reduction="sum")

            # construct gt_box_sizes as [batch x nprop x 3] matrix by using proposal to gt matching
            gt_box_sizes = torch.stack(
                [
                    torch.gather(
                        gt_box_sizes[:, :,
                                     x], 1, assignments["per_prop_gt_inds"]
                    )
                    for x in range(gt_box_sizes.shape[-1])
                ],
                dim=-1,
            )
            size_loss = F.l1_loss(pred_box_sizes, gt_box_sizes, reduction="none").sum(
                dim=-1
            )

            # zero-out non-matched proposals
            size_loss *= assignments["proposal_matched_mask"]
            size_loss = size_loss.sum()

            size_loss /= targets["num_boxes"]
        else:
            size_loss = torch.zeros(1, device=pred_box_sizes.device).squeeze()
        return {"loss_size": size_loss}

    def single_output_forward(self, outputs, ema_outputs, targets, outputs_static=None, is_last=True):
        gious = generalized_box3d_iou(
            outputs["box_corners"],
            targets["gt_box_corners"],
            targets["nactual_gt"],
            rotated_boxes=torch.any(targets["gt_box_angles"] > 0).item(),
            needs_grad=(self.loss_weight_dict["loss_giou_weight"] > 0),
        )

        outputs["gious"] = gious
        center_dist = torch.cdist(
            outputs["center_normalized"], targets["gt_box_centers_normalized"], p=1
        )
        outputs["center_dist"] = center_dist
        assignments = self.matcher(outputs, targets)
        if is_last:
            # revert augmentation for ema_outputs
            flip_x_axis = outputs["flip_x_axis"]
            flip_y_axis = outputs["flip_y_axis"]
            rot_mat = outputs["rot_mat"]
            rot_mat_transposed = rot_mat.transpose(1, 2)
            inds_to_flip_x_axis = torch.nonzero(flip_x_axis).squeeze(1)
            inds_to_flip_y_axis = torch.nonzero(flip_y_axis).squeeze(1)

            # ema_outputs shape is (batch, nprop, 3)
            ema_outputs["center_normalized"][inds_to_flip_x_axis, :, 0] = - \
                ema_outputs["center_normalized"][inds_to_flip_x_axis, :, 0]
            ema_outputs["center_normalized"][inds_to_flip_y_axis, :, 1] = - \
                ema_outputs["center_normalized"][inds_to_flip_y_axis, :, 1]
            ema_outputs["center_normalized"] = torch.bmm(
                ema_outputs["center_normalized"], rot_mat_transposed)

            # repeat for center_unnormalized
            ema_outputs["center_unnormalized"][inds_to_flip_x_axis, :, 0] = - \
                ema_outputs["center_unnormalized"][inds_to_flip_x_axis, :, 0]
            ema_outputs["center_unnormalized"][inds_to_flip_y_axis, :, 1] = - \
                ema_outputs["center_unnormalized"][inds_to_flip_y_axis, :, 1]
            ema_outputs["center_unnormalized"] = torch.bmm(
                ema_outputs["center_unnormalized"], rot_mat_transposed)

            # box_corners_copy = ema_outputs["box_corners"].clone().detach() # for debugging
            # also update the box corners
            # ema_outputs["box_corners"] shape is (batch, nprop, 8, 3)
            ema_outputs["box_corners"][inds_to_flip_x_axis, :, :, 0] = - \
                ema_outputs["box_corners"][inds_to_flip_x_axis, :, :, 0]
            ema_outputs["box_corners"][inds_to_flip_y_axis, :, :, 1] = - \
                ema_outputs["box_corners"][inds_to_flip_y_axis, :, :, 1]

            # Expand dimensions of rot_mat
            rot_mat_transposed_expanded = rot_mat_transposed.unsqueeze(
                1)  # Adds a new dimension after the first dimension

            # Perform batch matrix multiplication
            ema_outputs["box_corners"] = torch.matmul(
                ema_outputs["box_corners"], rot_mat_transposed_expanded)  # shape: (batch, nprop, 8, 3)

            # ema_outputs["box_corners"] = torch.bmm(
            #     ema_outputs["box_corners"], rot_mat.transpose(1, 2))

            # for debugging
            # process the box_corners_copy and compare with ema_outputs["box_corners"]
            # box_center_upright = flip_axis_to_camera_tensor(ema_outputs['center_unnormalized'])
            # box_corners_copy_2 = get_3d_box_batch_tensor(
            #     ema_outputs["size_unnormalized"], ema_outputs["angle_continuous"], box_center_upright
            # )

            # calculate ema gious
            ema_gious = generalized_box3d_iou(
                ema_outputs["box_corners"],
                targets["gt_box_corners"],
                targets["nactual_gt"],
                rotated_boxes=torch.any(targets["gt_box_angles"] > 0).item(),
                # needs_grad=(self.loss_weight_dict["loss_giou_weight"] > 0),
                # do not need grad for ema gious
                needs_grad=False,
            )
            ema_outputs["gious"] = ema_gious

            # calculate ema center dist
            ema_center_dist = torch.cdist(
                ema_outputs["center_normalized"], targets["gt_box_centers_normalized"], p=1
            )
            ema_outputs["center_dist"] = ema_center_dist

            # assignments for ema
            ema_assignments = self.matcher(ema_outputs, targets)

        losses = {}

        for k in self.loss_functions:
            loss_wt_key = k + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                if 'center_consistency' in k and is_last:
                    curr_loss, indices, masked_dist2 = self.loss_functions[k](
                        outputs, ema_outputs)
                elif 'cls_consistency' in k or 'size_consistency' in k and is_last:
                    curr_loss = self.loss_functions[k](
                        outputs, ema_outputs, indices, masked_dist2)
                # only compute losses with loss_wt > 0
                # certain losses like cardinality are only logged and have no loss weight
                # use static outputs for distillation loss
                elif k == "loss_distill" and is_last:
                    if outputs_static is None:
                        # during evaluation outputs_static is None, and there is no distillation loss.
                        # curr_loss = torch.zeros(1, device='cuda:0') # dummy loss TODO: does not work with multi-gpu
                        curr_loss = None
                    else:
                        # print(k, " Using static outputs for distillation loss")
                        curr_loss = self.loss_functions[k](
                            outputs, outputs_static, assignments)

                else:
                    curr_loss = self.loss_functions[k](
                        outputs, targets, assignments)

                if curr_loss is not None:
                    losses.update(curr_loss)

        final_loss = 0
        for k in self.loss_weight_dict:
            loss_name = k.replace("_weight", "")
            if self.loss_weight_dict[k] > 0 and loss_name in losses:
                losses[loss_name] *= self.loss_weight_dict[k]
                # further adjust the consistency loss by a scale factor
                if 'consistency' in loss_name and is_last:
                    losses[loss_name] *= self.consistency_weight_scale
                # scale distillation loss
                elif "distill" in loss_name and is_last:
                    losses[loss_name] *= self.distill_weight_scale
                final_loss += losses[loss_name]
        return final_loss, losses

    def forward(self, outputs, ema_outputs, targets, outputs_static=None):
        nactual_gt = targets["gt_box_present"].sum(axis=1).long()
        num_boxes = torch.clamp(all_reduce_average(
            nactual_gt.sum()), min=1).item()
        targets["nactual_gt"] = nactual_gt
        targets["num_boxes"] = num_boxes
        targets[
            "num_boxes_replica"
        ] = nactual_gt.sum().item()  # number of boxes on this worker for dist training

        if outputs_static is None:
            # to supress errors during evaluation
            # outputs_static['aux_outputs'] is a list of size 8, each element is None.
            outputs_static = {'outputs': None, 'aux_outputs': [None] * 8}
            # outputs_static = {'outputs': None, 'aux_outputs': [None]}

        loss, loss_dict = self.single_output_forward(outputs=outputs["outputs"],
                                                     ema_outputs=ema_outputs['outputs'], targets=targets,
                                                     outputs_static=outputs_static['outputs'], is_last=True)
        
        # not using aux_outputs for distillation loss and consistency loss
        # also needs to distill and scale this.
        if "aux_outputs" in outputs:
            for k in range(len(outputs["aux_outputs"])):

                # # update aux_outputs with the same augmentations as outputs
                # outputs["aux_outputs"][k]["flip_x_axis"] = outputs["outputs"]["flip_x_axis"]
                # outputs["aux_outputs"][k]["flip_y_axis"] = outputs["outputs"]["flip_y_axis"]
                # outputs["aux_outputs"][k]["rot_mat"] = outputs["outputs"]["rot_mat"]

                interm_loss, interm_loss_dict = self.single_output_forward(
                    outputs=outputs["aux_outputs"][k], ema_outputs=ema_outputs["aux_outputs"][
                        k], targets=targets, outputs_static=outputs_static['aux_outputs'][k], is_last=False
                )

                loss += interm_loss
                for interm_key in interm_loss_dict:
                    loss_dict[f"{interm_key}_{k}"] = interm_loss_dict[interm_key]
        return loss, loss_dict


def build_criterion(args, dataset_config):
    matcher = Matcher(
        cost_class=args.matcher_cls_cost,
        cost_giou=args.matcher_giou_cost,
        cost_center=args.matcher_center_cost,
        cost_objectness=args.matcher_objectness_cost,
    )

    loss_weight_dict = {
        "loss_giou_weight": args.loss_giou_weight,
        "loss_sem_cls_weight": args.loss_sem_cls_weight,
        "loss_no_object_weight": args.loss_no_object_weight,
        "loss_angle_cls_weight": args.loss_angle_cls_weight,
        "loss_angle_reg_weight": args.loss_angle_reg_weight,
        "loss_center_weight": args.loss_center_weight,
        "loss_size_weight": args.loss_size_weight,
        "loss_distill_weight": args.loss_distill_weight,
        # consistency
        'loss_center_consistency_weight': args.loss_center_consistency_weight,
        'loss_cls_consistency_weight': args.loss_cls_consistency_weight,
        'loss_size_consistency_weight': args.loss_size_consistency_weight,
    }
    criterion = SetCriterion(matcher, dataset_config, loss_weight_dict)
    return criterion
