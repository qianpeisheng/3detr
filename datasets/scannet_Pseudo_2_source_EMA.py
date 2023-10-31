# Copyright (c) Facebook, Inc. and its affiliates.

"""
Modified from https://github.com/facebookresearch/votenet
Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.

# Following SDCoT https://github.com/Na-Z/SDCoT/blob/main/cfg/scannet_cfg.py
# We change the order of class labels to match the order in the SDCoT paper.
# The order is alphabetical (a to z).
# "otherfurniture" in SDCoT is "garbagebin" in 3DETR.

"""
import os
import sys

import numpy as np
import torch
import utils.pc_util as pc_util
from torch.utils.data import Dataset
from utils.box_util import (flip_axis_to_camera_np, flip_axis_to_camera_tensor,
                            get_3d_box_batch_np, get_3d_box_batch_tensor)
from utils.pc_util import scale_points, shift_scale_points
from utils.random_cuboid import RandomCuboid
from utils.ap_calculator import get_ap_config_dict, parse_predictions_SDCoT
from utils.nms import nms_3d_faster_samecls, nms_3d_faster_base_novel

IGNORE_LABEL = -100
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DATASET_ROOT_DIR = "scannet_data/scannet_all"  # Replace with path to dataset
# Replace with path to dataset
DATASET_METADATA_DIR = "scannet_data/scannet/meta_data"
# In this dataset, all scans are saved in the same folder.

NUM_CLASS_BASE = 9  # depending on the base training classes.
NUM_CLASS_INCREMENTAL = 9  # depending on the incremental training classes.

SCANNET_9_9_BASE_PSEUDO_THRESHOLDS = np.array([
    0.93, 0.95, 0.89, 0.86, 0.83, 0.89, 0.88, 0.9, 0.21])
SCANNET_14_4_BASE_PSEUDO_THRESHOLDS = np.array([
    0.93, 0.95, 0.84, 0.8, 0.73, 0.75, 0.7, 0.82, 0.22, 0.69, 0.85, 0.68, 0.85, 0.85])
SCANNET_17_1_BASE_PSEUDO_THRESHOLDS = np.array([
    0.93, 0.95, 0.89, 0.86, 0.83, 0.89, 0.88, 0.9, 0.21])  # TODO update this

SCANNET_BASE_PSEUDO_THRESHOLDS = {
    9: SCANNET_9_9_BASE_PSEUDO_THRESHOLDS,
    14: SCANNET_14_4_BASE_PSEUDO_THRESHOLDS,
    17: SCANNET_17_1_BASE_PSEUDO_THRESHOLDS,
}


class ScannetDatasetConfig_Pseudo_2_source_EMA(object):
    def __init__(self, num_base_class=NUM_CLASS_BASE, num_novel_class=NUM_CLASS_INCREMENTAL):
        self.num_semcls = num_base_class + num_novel_class  # 18 means all classes
        self.num_base_class = num_base_class
        self.num_novel_class = num_novel_class
        self.num_angle_bin = 1
        self.max_num_obj = 64

        self.type2class = {
            "bathtub": 0,
            "bed": 1,
            "bookshelf": 2,
            "cabinet": 3,
            "chair": 4,
            "counter": 5,
            "curtain": 6,
            "desk": 7,
            "door": 8,
            "garbagebin": 9,  # otherfurniture in SDCoT
            "picture": 10,
            "refrigerator": 11,
            "showercurtrain": 12,
            "sink": 13,
            "sofa": 14,
            "table": 15,
            "toilet": 16,
            "window": 17,
        }

        # select only self.num_semcls classes
        self.type2class = {
            k: self.type2class[k] for k in list(self.type2class)[: self.num_semcls]
        }
        # note that dictionaries are ordered in Python 3.7+

        # check that the size of the dictionary is correct
        assert len(self.type2class) == self.num_semcls

        # original 3detr order
        # self.type2class = {
        #     "cabinet": 0,
        #     "bed": 1,
        #     "chair": 2,
        #     "sofa": 3,
        #     "table": 4,
        #     "door": 5,
        #     "window": 6,
        #     "bookshelf": 7,
        #     "picture": 8,
        #     "counter": 9,
        #     "desk": 10,
        #     "curtain": 11,
        #     "refrigerator": 12,
        #     "showercurtrain": 13,
        #     "toilet": 14,
        #     "sink": 15,
        #     "bathtub": 16,
        #     "garbagebin": 17,
        # }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.nyu40ids = np.array(
            # [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39] # 3detr original order
            np.array([36, 4, 10, 3, 5, 12, 16, 14, 8, 39, 11,
                     24, 28, 34, 6, 7, 33, 9])  # SDCoT order
        )

        # select only the novel classes
        self.nyu40ids_novel = self.nyu40ids[self.num_base_class:]

        # select only self.num_semcls classes
        self.nyu40ids = self.nyu40ids[: self.num_semcls]

        self.nyu40id2class = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))
        }

        # Semantic Segmentation Classes. Not used in 3DETR

        # self.num_class_semseg = 20
        # self.type2class_semseg = {
        #     "wall": 0,
        #     "floor": 1,
        #     "cabinet": 2,
        #     "bed": 3,
        #     "chair": 4,
        #     "sofa": 5,
        #     "table": 6,
        #     "door": 7,
        #     "window": 8,
        #     "bookshelf": 9,
        #     "picture": 10,
        #     "counter": 11,
        #     "desk": 12,
        #     "curtain": 13,
        #     "refrigerator": 14,
        #     "showercurtrain": 15,
        #     "toilet": 16,
        #     "sink": 17,
        #     "bathtub": 18,
        #     "garbagebin": 19,
        # }
        # self.class2type_semseg = {
        #     self.type2class_semseg[t]: t for t in self.type2class_semseg
        # }
        # self.nyu40ids_semseg = np.array(
        #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        # )
        # self.nyu40id2class_semseg = {
        #     nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids_semseg))
        # }

    def angle2class(self, angle):
        raise ValueError("ScanNet does not have rotated bounding boxes.")

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        zero_angle = torch.zeros(
            (pred_cls.shape[0], pred_cls.shape[1]),
            dtype=torch.float32,
            device=pred_cls.device,
        )
        return zero_angle

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        zero_angle = np.zeros(pred_cls.shape[0], dtype=np.float32)
        return zero_angle

    def param2obb(
        self,
        center,
        heading_class,
        heading_residual,
        size_class,
        size_residual,
        box_size=None,
    ):
        heading_angle = self.class2angle(heading_class, heading_residual)
        if box_size is None:
            box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    @staticmethod
    def rotate_aligned_boxes(input_boxes, rot_mat):
        centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
        new_centers = np.dot(centers, np.transpose(rot_mat))

        dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
        new_x = np.zeros((dx.shape[0], 4))
        new_y = np.zeros((dx.shape[0], 4))

        for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            crnrs = np.zeros((dx.shape[0], 3))
            crnrs[:, 0] = crnr[0] * dx
            crnrs[:, 1] = crnr[1] * dy
            crnrs = np.dot(crnrs, np.transpose(rot_mat))
            new_x[:, i] = crnrs[:, 0]
            new_y[:, i] = crnrs[:, 1]

        new_dx = 2.0 * np.max(new_x, 1)
        new_dy = 2.0 * np.max(new_y, 1)
        new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

        return np.concatenate([new_centers, new_lengths], axis=1)


class ScannetDetectionDataset_Pseudo_2_source_EMA(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set="train",
        root_dir=None,
        meta_data_dir=None,
        num_points=40000,
        use_color=False,
        use_height=False,
        augment=False,  # True for training, False for testing
        use_random_cuboid=True,
        random_cuboid_min_points=30000,
        use_ema_pseudo_label=False,  # whether to use ema pseudo label
        nms_threshold=0.5,
        use_cls_threshold=True,
    ):
        self.use_cls_threshold = use_cls_threshold
        self.use_ema_pseudo_label = use_ema_pseudo_label
        self.nms_threshold = nms_threshold
        if self.nms_threshold > 0.999:
            print(
                'Warning: No NMS is used to combine static and dynamic teacher pseudo labels.')

        self.nms_threshold_base_novel = 0.75

        self.dataset_config = dataset_config
        assert split_set in ["train", "val"]
        if root_dir is None:
            root_dir = DATASET_ROOT_DIR

        if meta_data_dir is None:
            meta_data_dir = DATASET_METADATA_DIR

        self.data_path = root_dir
        all_scan_names = list(
            set(
                [
                    os.path.basename(x)[0:12]
                    for x in os.listdir(self.data_path)
                    if x.startswith("scene")
                ]
            )
        )
        if split_set == "all":
            self.scan_names = all_scan_names
        elif split_set in ["train", "val", "test"]:
            split_filenames = os.path.join(
                meta_data_dir, f"scannetv2_{split_set}.txt")
            with open(split_filenames, "r") as f:
                self.scan_names = f.read().splitlines()
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [
                sname for sname in self.scan_names if sname in all_scan_names
            ]
            print(f"kept {len(self.scan_names)} scans out of {num_scans}")
        else:
            raise ValueError(f"Unknown split name {split_set}")

        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(
            min_points=random_cuboid_min_points)
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]

        # for pseuo label
        self.base_detector = None

    # Set the base detector for pseudo label.
    def set_base_detector(self, base_detector):
        self.base_detector = base_detector
        self.base_detector.cuda()
        # share memory
        # self.base_detector.share_memory()
        self.base_detector.eval()

    def set_ema_detector(self, ema_detector):
        self.ema_detector = ema_detector

    # Use the base detector to generate pseudo labels.

    def generate_pseudo_labels(self, point_clouds, mins, maxes, model, threshold_list=None, return_all_cls_probs=False):
        '''
            inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
            }
            outputs = model(inputs)
        '''

        # Note that DataLoader will convert the numpy array to torch tensor.
        # So we need to convert manually here.
        # https://pytorch.org/docs/stable/data.html

        batch_data_label = {}
        batch_data_label["point_clouds"] = torch.tensor(
            point_clouds).unsqueeze(0).cuda()
        batch_data_label["point_cloud_dims_min"] = torch.tensor(
            mins).unsqueeze(0).cuda()
        batch_data_label["point_cloud_dims_max"] = torch.tensor(
            maxes).unsqueeze(0).cuda()

        # self.base_detector.eval()
        model.eval()

        with torch.no_grad():
            outputs = model(batch_data_label)
            # if outputs is a list, set outputs to the first element
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                outputs = outputs[0]
            # This is to cater for distillation

            parsed_predictions = parse_predictions_SDCoT(
                predicted_boxes=outputs['outputs']['box_corners'],
                sem_cls_probs=outputs['outputs']['sem_cls_prob'],
                objectness_probs=outputs['outputs']['objectness_prob'],
                point_cloud=batch_data_label["point_clouds"],
                config_dict=self.ap_config_dict,
                use_cls_threshold=self.use_cls_threshold,
                threshold_list=threshold_list,
                return_all_cls_probs=return_all_cls_probs,
            )

        # convert to __getitem__ output format

        return parsed_predictions

    def pseudo_label_to_instance_bbox(self, pseudo_label):
        # pseudo_label is a list of length 3. The first element is the class index.
        # The second element is an array of (8, 3), which is the x,y,z coordinates of 8 corners of the bounding box.
        # The third element is the probality of the class.

        # PL to BB: 0 -> 0, -1 -> 2, 2 -> 1
        instance_bbox = np.zeros((7, ))
        instance_bbox[0] = pseudo_label[1][:, 0].mean(axis=0)
        # swap instance_bbox[1] and instance_bbox[2], and time -1 to instance_bbox[2]
        # to convert from SDCoT format to 3DETR format
        instance_bbox[1] = pseudo_label[1][:, 2].mean(axis=0)
        instance_bbox[2] = pseudo_label[1][:, 1].mean(axis=0) * -1

        # dx is the same, but dy and dz are swapped
        instance_bbox[3] = pseudo_label[1][:, 0].max(
            axis=0) - pseudo_label[1][:, 0].min(axis=0)
        instance_bbox[4] = pseudo_label[1][:, 2].max(
            axis=0) - pseudo_label[1][:, 2].min(axis=0)
        instance_bbox[5] = pseudo_label[1][:, 1].max(
            axis=0) - pseudo_label[1][:, 1].min(axis=0)

        # instance_bbox[3:6] = pseudo_label[1].max(
        #     axis=0) - pseudo_label[1].min(axis=0)
        instance_bbox[6] = pseudo_label[0]
        return instance_bbox

    def instance_box_to_pseudo_label(self, instance_box):
        # instance_box is a numpy array of shape (7, )
        # The first element is the x coordinate of the center of the bounding box.
        # The second element is the y coordinate of the center of the bounding box.
        # The third element is the z coordinate of the center of the bounding box.
        # The fourth element is the x-axis length of the bounding box. (full length not half length)
        # The fifth element is the y-axis length of the bounding box. (full length not half length)
        # The sixth element is the z-axis length of the bounding box. (full length not half length)
        # The seventh element is the class index.

        # BB to PL: 0 -> 0, -2 -> 1, 1 -> 2
        pseudo_label = []
        pseudo_label.append(instance_box[6])
        pseudo_label.append(np.zeros((8, 3)))
        # pseudo_label[1][:, 0] is the x coordinates of the 8 corners of the bounding box. i.e., x0 and x1.
        pseudo_label[1][:, 0] = np.array([instance_box[0] - instance_box[3] / 2, instance_box[0] + instance_box[3] / 2, instance_box[0] + instance_box[3] / 2, instance_box[0] - instance_box[3] / 2,
                                          instance_box[0] - instance_box[3] / 2, instance_box[0] + instance_box[3] / 2, instance_box[0] + instance_box[3] / 2, instance_box[0] - instance_box[3] / 2])
        # swap instance_box[1] and instance_box[2], and time -1 to instance_box[2]

        pseudo_label[1][:, 1] = -1 * np.array([instance_box[2] - instance_box[5] / 2, instance_box[2] - instance_box[5] / 2, instance_box[2] + instance_box[5] / 2, instance_box[2] + instance_box[5] / 2,
                                               instance_box[2] - instance_box[5] / 2, instance_box[2] - instance_box[5] / 2, instance_box[2] + instance_box[5] / 2, instance_box[2] + instance_box[5] / 2])
        pseudo_label[1][:, 2] = np.array([(instance_box[1] - instance_box[4] / 2), instance_box[1] - instance_box[4] / 2, instance_box[1] - instance_box[4] / 2, instance_box[1] - instance_box[4] / 2,
                                          instance_box[1] + instance_box[4] / 2, instance_box[1] + instance_box[4] / 2, instance_box[1] + instance_box[4] / 2, instance_box[1] + instance_box[4] / 2])
        pseudo_label.append(1)  # probability of the class is 1 for gound truth

        return pseudo_label

    def set_ap_config_dict(self, ap_config_dict):
        self.ap_config_dict = ap_config_dict

    def convert_to_nms_in(self, pseudo_labels):
        # pseudo_labels is a list of length 1.
        # pseudo_labels[0] is a list of length K, where K is the number of pseudo labels.
        # pseudo_labels[0][i] is a tuple of length 3. The first element is the class index.
        # The second element is an array of (8, 3), which is the x,y,z coordinates of 8 corners of the bounding box.
        # The third element is the probality of the class.
        # We convert pseudo_labels to a numpy array of shape (K, 8)
        # where the first column is x1, the second column is y1, the third column is z1,
        # the fourth column is x2, the fifth column is y2, the sixth column is z2.
        # the seventh column is the probability of the class,
        # the eighth column is the class index.

        nms_in_pseudo_labels = np.zeros((len(pseudo_labels[0]), 8))
        # x1
        nms_in_pseudo_labels[:, 0] = [pseudo_labels[0][i][1][:, 0].min(
            axis=0) for i in range(len(pseudo_labels[0]))]
        # y1
        nms_in_pseudo_labels[:, 1] = [pseudo_labels[0][i][1][:, 1].min(
            axis=0) for i in range(len(pseudo_labels[0]))]
        # z1
        nms_in_pseudo_labels[:, 2] = [pseudo_labels[0][i][1][:, 2].min(
            axis=0) for i in range(len(pseudo_labels[0]))]
        # x2
        nms_in_pseudo_labels[:, 3] = [pseudo_labels[0][i][1][:, 0].max(
            axis=0) for i in range(len(pseudo_labels[0]))]
        # y2
        nms_in_pseudo_labels[:, 4] = [pseudo_labels[0][i][1][:, 1].max(
            axis=0) for i in range(len(pseudo_labels[0]))]
        # z2
        nms_in_pseudo_labels[:, 5] = [pseudo_labels[0][i][1][:, 2].max(
            axis=0) for i in range(len(pseudo_labels[0]))]
        # score
        nms_in_pseudo_labels[:, 6] = [pseudo_labels[0][i][2]
                                      for i in range(len(pseudo_labels[0]))]
        # class
        nms_in_pseudo_labels[:, 7] = [pseudo_labels[0][i][0]
                                      for i in range(len(pseudo_labels[0]))]

        return nms_in_pseudo_labels

    def set_cls_threshold(self):
        self.static_base_pseudo_thresholds_list = SCANNET_BASE_PSEUDO_THRESHOLDS[
            self.dataset_config.num_base_class]
        self.dynamic_base_pseudo_thresholds_list = SCANNET_BASE_PSEUDO_THRESHOLDS[
            self.dataset_config.num_base_class]
        
    def update_dynamic_base_pseudo_thresholds_list(self, p_class):
        self.dynamic_base_pseudo_thresholds_list = p_class

    # def update_dynamic_base_pseudo_thresholds_list(self, tau=1, phi=None):
    #     # tau is a float between 0 and 1.
    #     # phi is a list of length self.dataset_config.num_base_class.
    #     # phi[i] is a float between 0 and 1.
    #     # The new threshold for class i is tau * self.dynamic_base_pseudo_thresholds_list[i] + (1 - tau) * phi[i]*self.dynamic_base_pseudo_thresholds_list[i]

    #     if phi is None:
    #         print('phi is None. No update is performed.')
    #         return
    #     else:
    #         for i in range(self.dataset_config.num_base_class):
    #             self.dynamic_base_pseudo_thresholds_list[i] = tau * \
    #                 self.dynamic_base_pseudo_thresholds_list[i] + \
    #                 (1 - tau) * phi[i] * \
    #                 self.dynamic_base_pseudo_thresholds_list[i]
    #         # crop the threshold to be between 1/num_base_class and 0.95
    #         self.dynamic_base_pseudo_thresholds_list = [
    #             min(0.95, max(1 / self.dataset_config.num_base_class, x)) for x in self.dynamic_base_pseudo_thresholds_list]

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        mesh_vertices = np.load(os.path.join(
            self.data_path, scan_name) + "_vert.npy")
        instance_bboxes = np.load(os.path.join(
            self.data_path, scan_name) + "_bbox.npy")

        # Filter instance_bboxes and keep only those with classes that are in the dataset_config
        # print(f'{idx}, N instance_bboxes before filtering: ', len(instance_bboxes))
        instance_bboxes = instance_bboxes[
            np.isin(instance_bboxes[:, -1], self.dataset_config.nyu40ids_novel)
        ]
        # print the number of instance_bboxes before and after filtering
        # print(f'{idx}, N instance_bboxes after filtering: ', len(instance_bboxes))

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:]

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate(
                [point_cloud, np.expand_dims(height, 1)], 1)

        # ------------------------------- LABELS ------------------------------
        MAX_NUM_OBJ = self.dataset_config.max_num_obj
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.float32)
        angle_classes = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)
        angle_residuals = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        raw_sizes = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        raw_angles = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)

        if self.augment and self.use_random_cuboid:
            (
                point_cloud,
                instance_bboxes,
                per_point_labels,
            ) = self.random_cuboid_augmentor(
                point_cloud, instance_bboxes,
            )

        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )

        # repeat random sampling to get ema_point_cloud
        ema_point_cloud = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=False
        )

        # uncomment to use semantic labels

        pcl_color = pcl_color[choices]

        # save instance_bboxes without class labels
        # skip this step when nms is used on instance_bboxes (GT novel class)
        # target_bboxes[0: instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

        # to obtain pseudo labels
        pseudo_labels = self.generate_pseudo_labels(
            point_cloud,
            point_cloud.min(axis=0)[:3],
            point_cloud.max(axis=0)[:3],
            self.base_detector,
            threshold_list=self.static_base_pseudo_thresholds_list
        )

        arr_of_cls_dynamic = -1 * np.ones(MAX_NUM_OBJ)
        arr_of_prob_dynamic = -1 * np.ones((MAX_NUM_OBJ, self.dataset_config.num_base_class))
        # if use_ema_pseudo_label, get pseudo labels from ema detector and append to pseudo_labels
        if self.use_ema_pseudo_label:
            ema_pseudo_labels = self.generate_pseudo_labels(
                ema_point_cloud,
                ema_point_cloud.min(axis=0)[:3],
                ema_point_cloud.max(axis=0)[:3],
                self.ema_detector,
                threshold_list=self.dynamic_base_pseudo_thresholds_list,
                return_all_cls_probs = True
            )

            # update list_of_cls_probs
            for i in range(len(ema_pseudo_labels[0])):
                if i < arr_of_cls_dynamic.shape[0]:
                    arr_of_cls_dynamic[i] = ema_pseudo_labels[0][i][0]
                    arr_of_prob_dynamic[i] = ema_pseudo_labels[0][i][3][:self.dataset_config.num_base_class]
            # map class using nyu40id2class in instance_bboxes
            for i in range(len(instance_bboxes)):
                instance_bboxes[i][6] = self.dataset_config.nyu40id2class[instance_bboxes[i][6]]
            # convert gt instance_bboxes to pseudo label format then to nms_in format
            gt_labels = []
            for i in range(len(instance_bboxes)):
                gt_labels.append(
                    self.instance_box_to_pseudo_label(instance_bboxes[i]))

            # add a dimension to gt_labels
            gt_labels = [gt_labels]
            nms_in_gt_labels = self.convert_to_nms_in(gt_labels)

            # # convert pseudo_labels to nms_in format
            nms_in_pseudo_labels = self.convert_to_nms_in(pseudo_labels)

            # convert ema_pseudo_labels to nms_in format
            nms_in_ema_pseudo_labels = self.convert_to_nms_in(ema_pseudo_labels)

            # combine all 3 labels and pseudo labels
            nms_in_all_labels = np.concatenate(
                (nms_in_gt_labels, nms_in_pseudo_labels, nms_in_ema_pseudo_labels), axis=0)

            # # do nms for pseudo_labels and ground truth instance_bboxes using nms_3d_faster_base_novel
            # remove base pseudo labels if they overlap with novel pseudo labels, otherwise keep them.
            nms_out_all_labels_idx = nms_3d_faster_base_novel(
                boxes=nms_in_all_labels, overlap_threshold=self.nms_threshold_base_novel, num_base_cls=self.dataset_config.num_base_class, old_type=False)
            # filter nms_in_all_labels by nms_out_all_labels
            nms_in_all_labels_filtered = nms_in_all_labels[nms_out_all_labels_idx]

            # print the number of pseudo labels before and after NMS
            # print(f'{idx} labels before all NMS: ', len(nms_in_all_labels))
            # print(f'{idx} labels after all NMS: ', len(nms_in_all_labels_filtered))

            # nms_in_all_labels_filtered can be used for class wise NMS in the following code.
            # use nms_3d_faster_samecls to get the final pseudo labels
            nms_in_all_labels_filtered_after_cls_nms_idx = nms_3d_faster_samecls(
                nms_in_all_labels_filtered, self.nms_threshold, old_type=False)
            # filter nms_in_pseudo_labels_combined by nms_out_pseudo_labels
            nms_in_all_labels_filtered_after_cls_nms = nms_in_all_labels_filtered[
                nms_in_all_labels_filtered_after_cls_nms_idx]

            # convert nms_in_pseudo_labels_combined_filtered to pseudo_labels' format
            all_labels_filtered_after_cls_nms = []
            for i in range(len(nms_in_all_labels_filtered_after_cls_nms)):
                all_labels_filtered_after_cls_nms.append((int(nms_in_all_labels_filtered_after_cls_nms[i][7]), np.array([[nms_in_all_labels_filtered_after_cls_nms[i][0], nms_in_all_labels_filtered_after_cls_nms[i][1], nms_in_all_labels_filtered_after_cls_nms[i][2]],
                                                                                                                         [nms_in_all_labels_filtered_after_cls_nms[i][3], nms_in_all_labels_filtered_after_cls_nms[i][4], nms_in_all_labels_filtered_after_cls_nms[i][5]]], dtype=np.float32), nms_in_all_labels_filtered_after_cls_nms[i][6]))
            # print the number of pseudo labels before and after NMS
            # print(f'{idx} N pseudo labels before NMS: ', len(nms_in_pseudo_labels_combined))
            # print(f'{idx} N pseudo labels after NMS: ', len(nms_in_pseudo_labels_combined_filtered))
            # let pseudo_labels be pseudo_labels_filtered
            # pseudo_labels is a list of length 1
            pseudo_labels = [all_labels_filtered_after_cls_nms]
            # print(f'{idx} labels after cls NMS: ', len(all_labels_filtered_after_cls_nms))
            # # convert nms_in_all_labels_filtered to pseudo_labels' format1
            # all_labels_filtered = []
            # for i in range(len(nms_in_all_labels_filtered)):
            #     all_labels_filtered.append((int(nms_in_all_labels_filtered[i][7]), np.array([[nms_in_all_labels_filtered[i][0], nms_in_all_labels_filtered[i][1], nms_in_all_labels_filtered[i][2]],
            #                                                                                   [nms_in_all_labels_filtered[i][3], nms_in_all_labels_filtered[i][4], nms_in_all_labels_filtered[i][5]]], dtype=np.float32), nms_in_all_labels_filtered[i][6]))

            no_base_obj = 0
            no_obj = 0

            # NMS
            # if len(ema_pseudo_labels[0]) > 0:
            #     # select only base classes from ema_pseudo_labels, i.e., not novel classes
            #     ema_pseudo_labels[0] = [ema_pseudo_labels[0][i] for i in range(len(
            #         ema_pseudo_labels[0])) if ema_pseudo_labels[0][i][0] < self.dataset_config.num_base_class]
            #     if len(ema_pseudo_labels[0]) > 0:
            #         if self.nms_threshold >= 0.999:
            #             # append ema_pseudo_labels to pseudo_labels without filtering
            #             pseudo_labels[0].extend(ema_pseudo_labels[0])
            #         else:
            #             # convert ema_pseudo_labels to nms_in format
            #             nms_in_ema_pseudo_labels = self.convert_to_nms_in(ema_pseudo_labels)
            #             nms_in_pseudo_labels = self.convert_to_nms_in(pseudo_labels)
            #             # combine nms_in_pseudo_labels and nms_in_ema_pseudo_labels
            #             nms_in_pseudo_labels_combined = np.concatenate((nms_in_pseudo_labels, nms_in_ema_pseudo_labels), axis=0)
            #             # use nms_3d_faster_samecls to get the final pseudo labels
            #             nms_out_pseudo_labels = nms_3d_faster_samecls(nms_in_pseudo_labels_combined, self.nms_threshold, old_type=False)
            #             # filter nms_in_pseudo_labels_combined by nms_out_pseudo_labels
            #             nms_in_pseudo_labels_combined_filtered = nms_in_pseudo_labels_combined[nms_out_pseudo_labels]

            #             # convert nms_in_pseudo_labels_combined_filtered to pseudo_labels' format
            #             pseudo_labels_filtered = []
            #             for i in range(len(nms_in_pseudo_labels_combined_filtered)):
            #                 pseudo_labels_filtered.append((int(nms_in_pseudo_labels_combined_filtered[i][7]), np.array([[nms_in_pseudo_labels_combined_filtered[i][0], nms_in_pseudo_labels_combined_filtered[i][1], nms_in_pseudo_labels_combined_filtered[i][2]],
            #                                                                                                             [nms_in_pseudo_labels_combined_filtered[i][3], nms_in_pseudo_labels_combined_filtered[i][4], nms_in_pseudo_labels_combined_filtered[i][5]]], dtype=np.float32), nms_in_pseudo_labels_combined_filtered[i][6]))
            #             # print the number of pseudo labels before and after NMS
            #             # print(f'{idx} N pseudo labels before NMS: ', len(nms_in_pseudo_labels_combined))
            #             # print(f'{idx} N pseudo labels after NMS: ', len(nms_in_pseudo_labels_combined_filtered))
            #             # let pseudo_labels be pseudo_labels_filtered
            #             pseudo_labels = [pseudo_labels_filtered] # pseudo_labels is a list of length 1
            #     else:
            #         # print(f'{scan_name} ema no base cls obj')
            #         no_base_obj = 1
            # else:
            #     # print(f'{scan_name} ema no obj')
            #     no_obj = 1

        # pseudo_labels.append(ema_pseudo_labels[0])

        # Convert the pseudo labels to the format of instance_bboxes
        converted_instance_bboxes = []
        for pseudo_label in pseudo_labels[0]:
            converted_instance_bboxes.append(
                self.pseudo_label_to_instance_bbox(pseudo_label))

        # make sure converted_instance_bboxes is no more than MAX_NUM_OBJ - the number of instance_bboxes

        # when gt is not NMSed.
        # max_converted_instance_bboxes_num = MAX_NUM_OBJ - instance_bboxes.shape[0]
        # when gt is NMSed.
        max_converted_instance_bboxes_num = MAX_NUM_OBJ
        if len(converted_instance_bboxes) > max_converted_instance_bboxes_num:
            print(
                'Warning: converted_instance_bboxes is more than MAX_NUM_OBJ - the number of instance_bboxes')
            converted_instance_bboxes = converted_instance_bboxes[0:
                                                                  max_converted_instance_bboxes_num]

        # concat converted_instance_bboxes and instance_bboxes
        converted_instance_bboxes_no_cls = [
            converted_instance_bbox[0:6] for converted_instance_bbox in converted_instance_bboxes]

        # pseudo labels are added to the front of the instance_bboxes
        # if converted_instance_bboxes is empty, skip this step
        if len(converted_instance_bboxes) > 0:
            target_bboxes = np.concatenate(
                (converted_instance_bboxes_no_cls, target_bboxes), axis=0)
            # if gt is NMSed,
            # if len(converted_instance_bboxes) + instance_bboxes.shape[0] > MAX_NUM_OBJ:
            # if gt is not NMSed, converted_instance_bboxes already includes instance_bboxes
            if len(converted_instance_bboxes) > MAX_NUM_OBJ:
                target_bboxes_mask[:] = 1
            else:
                # target_bboxes_mask[0: len(converted_instance_bboxes) +
                #                    instance_bboxes.shape[0]] = 1
                target_bboxes_mask[0: len(converted_instance_bboxes)] = 1
        else:
            # no pseudo labels, only instance labels
            # assuming instance_bboxes is no more than MAX_NUM_OBJ
            # target_bboxes_mask[0: instance_bboxes.shape[0]] = 1
            # converted instance bboxes include instance bboxes but is empty.
            print('Warning: no pseudo labels nor instance labels')

        # make sure target_bboxes is no more than MAX_NUM_OBJ
        # this line will always trigger because target_bboxes is already MAX_NUM_OBJ
        # but the truncated values are all zeros
        target_bboxes = target_bboxes[0: MAX_NUM_OBJ]

        flip_x_axis = 0
        flip_y_axis = 0
        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:

            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                flip_x_axis = 1
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                flip_y_axis = 1
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

            # Rotation along up-axis/Z-axis
            # debug rot_angle, set to 0 TODO remove the debug code
            # rot_angle = 0.
            rot_angle = (np.random.random() * np.pi / 18) - \
                np.pi / 36  # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(
                point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = self.dataset_config.rotate_aligned_boxes(
                target_bboxes, rot_mat
            )

        raw_sizes = target_bboxes[:, 3:6]
        point_cloud_dims_min = point_cloud.min(axis=0)[:3]
        point_cloud_dims_max = point_cloud.max(axis=0)[:3]

        # for ema
        ema_point_cloud_dims_min = ema_point_cloud.min(axis=0)[:3]
        ema_point_cloud_dims_max = ema_point_cloud.max(axis=0)[:3]

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * \
            target_bboxes_mask[..., None]
        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(
            np.float32
        )
        ret_dict["gt_angle_class_label"] = angle_classes.astype(np.int64)
        ret_dict["gt_angle_residual_label"] = angle_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))

        # When gt is not NMSed.
        # # Do it differently for pseudo labels and instance labels.
        # # No need to convert pseudo label classes because they are already mapped.
        # # For instance labels, we need to map them to the new class ids.
        # target_bboxes_semcls[0: len(converted_instance_bboxes)
        #                      ] = [converted_instance_bbox[-1] for converted_instance_bbox in converted_instance_bboxes]
        # target_bboxes_semcls[len(converted_instance_bboxes): len(converted_instance_bboxes) + instance_bboxes.shape[0]] = [
        #     self.dataset_config.nyu40id2class[int(x)]
        #     for x in instance_bboxes[:, -1][0: instance_bboxes.shape[0]]
        # ]

        # target_bboxes_semcls[0: instance_bboxes.shape[0]] = [
        #     self.dataset_config.nyu40id2class[int(x)]
        #     for x in instance_bboxes[:, -1][0: instance_bboxes.shape[0]]
        # ]

        # When gt is NMSed, treat them the same way as pseudo labels.
        target_bboxes_semcls[0: len(converted_instance_bboxes)
                             ] = [converted_instance_bbox[-1] for converted_instance_bbox in converted_instance_bboxes]

        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        # ret_dict['scan_name'] = scan_name
        # ret_dict['num_pseudo_label'] = len(converted_instance_bboxes)
        ret_dict["pcl_color"] = pcl_color
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        # ret_dict['pseudo_labels'] = pseudo_labels
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(
            np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min.astype(
            np.float32)
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max.astype(
            np.float32)
        # ema
        ret_dict["ema_point_clouds"] = ema_point_cloud.astype(np.float32)
        ret_dict["ema_point_cloud_dims_min"] = ema_point_cloud_dims_min.astype(
            np.float32)
        ret_dict["ema_point_cloud_dims_max"] = ema_point_cloud_dims_max.astype(
            np.float32)

        ret_dict['flip_x_axis'] = np.array(flip_x_axis).astype(np.int64)
        ret_dict['flip_y_axis'] = np.array(flip_y_axis).astype(np.int64)
        ret_dict['rot_mat'] = rot_mat.astype(np.float32)
        # np.array(no_base_obj).astype(np.int64)
        ret_dict['no_base_obj'] = no_base_obj
        ret_dict['no_obj'] = no_obj  # np.array(no_obj).astype(np.int64)
        # ret_dict['scan_name'] = scan_name
        ret_dict['arr_of_cls_dynamic'] = arr_of_cls_dynamic
        ret_dict['arr_of_prob_dynamic'] = arr_of_prob_dynamic

        return ret_dict # 

# TODO add testing code below
