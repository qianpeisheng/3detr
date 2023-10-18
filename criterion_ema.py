# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.box_util import generalized_box3d_iou, flip_axis_to_camera_tensor, get_3d_box_batch_tensor
from utils.dist import all_reduce_average
from utils.misc import huber_loss
from scipy.optimize import linear_sum_assignment

from utils.nn_distance import nn_distance, huber_loss


class Matcher(nn.Module):
    def __init__(self, cost_class, cost_objectness, cost_giou, cost_center):
        """
        Explaination what the matcher does and how it works:
        - cost_class: cost of classifying a proposal as a particular class
        - cost_objectness: cost of classifying a proposal as an object
        - cost_giou: cost of giou between a proposal and a ground truth box
        - cost_center: cost of distance between a proposal and a ground truth box
        - cost_angle: cost of angle between a proposal and a ground truth box

        The Matcher computes a cost matrix between all proposals and all ground truth boxes.
        The cost matrix is of shape [batch x nqueries x ngt].
        The cost matrix is computed as:
        cost_matrix = cost_class * class_cost_matrix + cost_objectness * objectness_cost_matrix
                    + cost_giou * giou_cost_matrix + cost_center * center_cost_matrix
                    + cost_angle * angle_cost_matrix
        where each cost matrix is of shape [batch x nqueries x ngt].
        The class_cost_matrix is computed as:
        class_cost_matrix[b, i, j] = -log(p_i(c_j))
        where p_i(c_j) is the probability of proposal i being classified as class j.
        The objectness_cost_matrix is computed as:
        objectness_cost_matrix[b, i, j] = -log(p_i(o_j))
        where p_i(o_j) is the probability of proposal i being classified as an object.
        The giou_cost_matrix is computed as:
        giou_cost_matrix[b, i, j] = 1 - IoU_i(j)
        where IoU_i(j) is the IoU between proposal i and ground truth box j.
        The center_cost_matrix is computed as:
        center_cost_matrix[b, i, j] = ||c_i - c_j||_1
        where c_i is the center of proposal i and c_j is the center of ground truth box j.
        The angle_cost_matrix is computed as:
        angle_cost_matrix[b, i, j] = ||a_i - a_j||_1
        where a_i is the angle of proposal i and a_j is the angle of ground truth box j.


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
        '''
        Args:
            outputs: dict of torch.tensor
                "sem_cls_prob": batch x nqueries x num_semcls tensor
                "objectness_prob": batch x nqueries x 1 tensor
                "center_dist": batch x nqueries x ngt tensor
                "gious": batch x nqueries x ngt tensor
            targets: dict of torch.tensor
                "gt_box_sem_cls_label": batch x ngt tensor
                "gt_box_centers_normalized": batch x ngt x 2 tensor
                "gt_box_sizes_normalized": batch x ngt x 3 tensor
                "gt_box_angles": batch x ngt tensor
                "nactual_gt": batch tensor
        Return:
            assignments: dict of torch.tensor
                "assignments": list of list of torch.tensor
                    assignments[b] is a list of length nactual_gt[b] where each element is a
                    torch.tensor of shape [2] containing the indices of the matched proposal
                    and ground truth box.
                "per_prop_gt_inds": batch x nqueries tensor
                    per_prop_gt_inds[b, i] is the index of the ground truth box matched to
                    proposal i.
                "proposal_matched_mask": batch x nqueries tensor
                    proposal_matched_mask[b, i] is 1 if proposal i is matched to a ground
                    truth box and 0 otherwise.

        '''

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

        # class_mat is the cost of classifying a proposal as a particular class.
        # class_mat[b, i, j] is the cost of classifying proposal i as class j.
        # class_mat[b, i, j] = -log(p_i(c_j))
        # where p_i(c_j) is the probability of proposal i being classified as class j.
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
            "loss_center_consistency": self.loss_center_consistency_all,
            'loss_cls_consistency': self.loss_class_consistency_all,
            'loss_size_consistency': self.loss_size_consistency_all,
        }

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
        # follow the SDCoT implementation
        # center = outputs["center_normalized"].clone()
        # ema_center = ema_outputs["center_normalized"].clone()
        center = outputs["center_normalized"]  # shape (B, num_proposal, 3)
        # shape (B, num_proposal, 3)
        ema_center = ema_outputs["center_normalized"]

        dist1, ind1, dist2, ind2 = nn_distance(center, ema_center)
        # ind1 is (B, num_proposal): ema_center index closest to center
        # ind2 is (B, num_proposal): center index closest to ema_center

        # TODO: use both dist1 and dist2 or only use dist1
        dist = dist1 + dist2
        # return torch.mean(dist), ind2
        return {"loss_center_consistency": torch.mean(dist)}, ind2

    def loss_center_consistency(self, outputs, ema_outputs, assignments, ema_assignments):
        # TODO the following could be vectorized
        # loop through assignments['assignments']
        # for i in range(len(assignments['assignments'])):
        #     t1 = assignments['assignments'][i]
        #     sorted_indices = torch.argsort(t1[1])
        #     # Reorder t1[0] and t1[1] based on the sorted indices
        #     t1[0] = torch.index_select(t1[0], dim=0, index=sorted_indices)
        #     t1[1] = torch.index_select(t1[1], dim=0, index=sorted_indices)

        # for i in range(len(ema_assignments['assignments'])):
        #     t1 = ema_assignments['assignments'][i]
        #     sorted_indices = torch.argsort(t1[1])
        #     # Reorder t1[0] and t1[1] based on the sorted indices
        #     t1[0] = torch.index_select(t1[0], dim=0, index=sorted_indices)
        #     t1[1] = torch.index_select(t1[1], dim=0, index=sorted_indices)
        for i in range(len(assignments['assignments'])):
            t1 = assignments['assignments'][i]
            try:
                sorted_indices = torch.argsort(t1[1])
            except IndexError:
                # empty tensor. skip
                # print('empty prediction from the student model')
                continue
            sorted_t1_0 = torch.index_select(t1[0], dim=0, index=sorted_indices)
            sorted_t1_1 = torch.index_select(t1[1], dim=0, index=sorted_indices)
            assignments['assignments'][i] = [sorted_t1_0, sorted_t1_1]

        for i in range(len(ema_assignments['assignments'])):
            t1 = ema_assignments['assignments'][i]
            try:
                sorted_indices = torch.argsort(t1[1])
            except IndexError:
                # empty tensor. skip
                # print('empty prediction from the dynamic teacher model')
                continue
            sorted_t1_0 = torch.index_select(t1[0], dim=0, index=sorted_indices)
            sorted_t1_1 = torch.index_select(t1[1], dim=0, index=sorted_indices)
            ema_assignments['assignments'][i] = [sorted_t1_0, sorted_t1_1]

        # assignments['assignments'][0] = assignments['assignments'][0].sort()[0]
        # clone to avoid modifying the original outputs
        center = outputs["center_normalized"].clone()
        ema_center = ema_outputs["center_normalized"].clone()
        # import pdb; pdb.set_trace()

        center_dist_2 = torch.cdist(
            center, ema_center, p=2
        )  # the reverse has the same mean value

        # center cost: batch x nqueries x ngt
        # This is to debug modified gradients
        center_dist_2_clone = center_dist_2  # .clone()

        # giou cost: batch x nqueries x ngt
        # giou_mat = -outputs["gious"].detach()

        # Extract indices from assignments and ema_assignments
        # handle empty tensors, which will not be saved in indices
        indices = [(i, inner[0], ema_inner[0]) for i, (inner, ema_inner) in enumerate(zip(
            assignments['assignments'], ema_assignments['assignments'])) if inner and ema_inner]
        # indices = [(i, assignments['assignments'][i][0], ema_assignments['assignments'][i][0]) for i in range(len(assignments['assignments'])) if len(assignments['assignments'][i][0]) > 0]
        # indices = [(i, assignments['assignments'][i][0], ema_assignments['assignments'][i][0]) for i in range(len(assignments['assignments']))]
        # return {"loss_center_consistency": sum(center_dist_2_clone) / len(center_dist_2_clone)}, indices#assignments, ema_assignments # magnitude 8

        # Select elements from center_dist_2 based on the extracted indices
        selected_elements = [center_dist_2_clone[i, assignments_index, ema_assignments_index]
                             for i, assignments_index, ema_assignments_index in indices]
        center_consistency_loss = [selected_element.sum()
                                   for selected_element in selected_elements]
        # center shape: (batch_size, nprop, 3), so is ema_center
        # for each prediction in the batch, use the assignments to select the centers
        # of the matched proposals. Repeat for ema_center.
        # And calculate the distance between the two centers.
        # # align ema_center to center based on the augmentations in outputs
        # flip_x_axis = outputs["flip_x_axis"]
        # flip_y_axis = outputs["flip_y_axis"]
        # rot_mat = outputs["rot_mat"]
        # inds_to_flip_x_axis = torch.nonzero(flip_x_axis).squeeze(1)
        # inds_to_flip_y_axis = torch.nonzero(flip_y_axis).squeeze(1)

        # # ema_center shape is (batch, nprop, 3)
        # ema_center[inds_to_flip_x_axis,:, 0] = -ema_center[inds_to_flip_x_axis,:, 0]
        # ema_center[inds_to_flip_y_axis, :, 1] = -ema_center[inds_to_flip_y_axis, :, 1]
        # ema_center = torch.bmm(ema_center, rot_mat.transpose(1,2))

        # referring to SESS, we only consider the matched proposals which are
        # closest pairs of the two sets of proposals.

        # assignments, ema_assignments # magnitude 8
        return {"loss_center_consistency": sum(center_consistency_loss) / len(center_consistency_loss)}, indices

    def loss_class_consistency_all(self, outputs, ema_outputs, map_ind):

        # follow the SDCoT implementation
        sem_cls_logits = outputs["sem_cls_logits"]
        sem_cls_prob_log = F.log_softmax(
            sem_cls_logits, dim=-1)  # for kl_div loss
        ema_cls_logits = ema_outputs["sem_cls_logits"]
        ema_sem_cls_prob = F.softmax(ema_cls_logits, dim=-1)

        cls_log_prob_aligned = torch.cat([torch.index_select(
            a, 0, i).unsqueeze(0) for a, i in zip(sem_cls_prob_log, map_ind)])

        class_consistency_loss = F.kl_div(
            cls_log_prob_aligned, ema_sem_cls_prob, reduction='mean')
        # class_consistency_loss = F.mse_loss(cls_log_prob_aligned, ema_cls_prob)

        # return class_consistency_loss*2
        return {"loss_cls_consistency": class_consistency_loss*2}

    def loss_class_consistency(self, outputs, ema_outputs, indices):
        # need to use logits to compute prob and kl_div loss, because prob does not have
        # the no-object class, and the sum of prob is not 1, which is required by kl_div loss.
        sem_cls_logits = outputs["sem_cls_logits"]
        # sem_cls_prob = outputs["sem_cls_prob"]
        sem_cls_prob_log = F.log_softmax(
            sem_cls_logits, dim=-1)  # for kl_div loss
        # sem_cls_prob_log = sem_cls_prob.log()
        # ema_sem_cls_prob = ema_outputs["sem_cls_prob"]
        ema_cls_logits = ema_outputs["sem_cls_logits"]
        ema_sem_cls_prob = F.softmax(ema_cls_logits, dim=-1)

        # sem_cls_aligned = [sem_cls_prob_log[i, assignments_index, :-1]
        sem_cls_aligned = [sem_cls_prob_log[i, assignments_index, :]  # prob already excludes the no-object class
                           for i, assignments_index, _ in indices]  # -1 is the no-object class
        ema_sem_cls_aligned = [ema_sem_cls_prob[i, ema_assignments_index]
                               for i, _, ema_assignments_index in indices]
        cls_consistency_loss = [F.kl_div(
            sem_cls_aligned[i], ema_sem_cls_aligned[i], reduction='mean') for i in range(len(indices))]
        # use len(indices) instead of sem_cls_prob_log.shape[0] to avoid empty tensors
        # magnitude 0.0073
        return {"loss_cls_consistency": sum(cls_consistency_loss)/len(cls_consistency_loss)}

    def loss_size_consistency_all(self, outputs, ema_outputs, map_ind):
        # follow the SDCoT implementation
        size_normalized = outputs["size_normalized"]
        ema_size_normalized = ema_outputs["size_normalized"]

        size_aligned = torch.cat([torch.index_select(
            a, 0, i).unsqueeze(0) for a, i in zip(size_normalized, map_ind)])
        size_consistency_loss = F.mse_loss(
            size_aligned, ema_size_normalized, reduction='mean')

        # return size_consistency_loss
        return {"loss_size_consistency": size_consistency_loss}

    def loss_size_consistency(self, outputs, ema_outputs, indices):
        output_sizes = outputs["size_normalized"]
        ema_output_sizes = ema_outputs["size_normalized"]
        size_aligned = [output_sizes[i, assignments_index]
                        for i, assignments_index, _ in indices]
        ema_size_aligned = [ema_output_sizes[i, ema_assignments_index]
                            for i, _, ema_assignments_index in indices]
        size_consistency_loss = [F.mse_loss(
            size_aligned[i], ema_size_aligned[i]) for i in range(len(indices))]

        # magnitude 0.0121
        return {"loss_size_consistency": sum(size_consistency_loss) / len(size_consistency_loss)}

    # not used

    # def loss_consistency(self, outputs, ema_outputs, assignments, ema_assignments):
    #     # assignments is not used. It is only passed to keep the same function signature as other losses
    #     loss_center_consistency, indices = self.loss_center_consistency(
    #         outputs, ema_outputs, assignments, ema_assignments)
    #     loss_class_consistency = self.loss_class_consistency(
    #         outputs, ema_outputs, indices)
    #     loss_size_consistency = self.loss_size_consistency(
    #         outputs, ema_outputs, indices)
    #     loss_consistency = loss_center_consistency + \
    #         loss_class_consistency + loss_size_consistency
    #     return {"loss_consistency": loss_consistency}

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

    def single_output_forward(self, outputs, ema_outputs, targets):
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
                if 'center_consistency' in k:
                    curr_loss, indices = self.loss_functions[k](
                        outputs, ema_outputs)
                elif 'cls_consistency' in k or 'size_consistency' in k:
                    curr_loss = self.loss_functions[k](
                        outputs, ema_outputs, indices)
                else:
                    # only compute losses with loss_wt > 0
                    # certain losses like cardinality are only logged and have no loss weight
                    curr_loss = self.loss_functions[k](
                        outputs, targets, assignments)
                losses.update(curr_loss)

        final_loss = 0
        for k in self.loss_weight_dict:
            if self.loss_weight_dict[k] > 0:
                losses[k.replace("_weight", "")] *= self.loss_weight_dict[k]
                # further adjust the consistency loss by a scale factor
                if 'consistency' in k:
                    losses[k.replace("_weight", "")
                           ] *= self.consistency_weight_scale
                final_loss += losses[k.replace("_weight", "")]
        return final_loss, losses

    def forward(self, outputs, ema_outputs, targets):
        nactual_gt = targets["gt_box_present"].sum(axis=1).long()
        num_boxes = torch.clamp(all_reduce_average(
            nactual_gt.sum()), min=1).item()
        targets["nactual_gt"] = nactual_gt
        targets["num_boxes"] = num_boxes
        targets[
            "num_boxes_replica"
        ] = nactual_gt.sum().item()  # number of boxes on this worker for dist training

        loss, loss_dict = self.single_output_forward(outputs=outputs["outputs"],
                                                     ema_outputs=ema_outputs['outputs'], targets=targets)

        if "aux_outputs" in outputs:
            for k in range(len(outputs["aux_outputs"])):

                # update aux_outputs with the same augmentations as outputs
                outputs["aux_outputs"][k]["flip_x_axis"] = outputs["outputs"]["flip_x_axis"]
                outputs["aux_outputs"][k]["flip_y_axis"] = outputs["outputs"]["flip_y_axis"]
                outputs["aux_outputs"][k]["rot_mat"] = outputs["outputs"]["rot_mat"]

                interm_loss, interm_loss_dict = self.single_output_forward(
                    # outputs=outputs["aux_outputs"][k], ema_outputs=ema_outputs['outputs'], targets=targets
                    outputs=outputs["aux_outputs"][k], ema_outputs=ema_outputs["aux_outputs"][k], targets=targets

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
        # consistency
        'loss_center_consistency_weight': args.loss_center_consistency_weight,
        'loss_cls_consistency_weight': args.loss_cls_consistency_weight,
        'loss_size_consistency_weight': args.loss_size_consistency_weight,
    }
    criterion = SetCriterion(matcher, dataset_config, loss_weight_dict)
    return criterion
