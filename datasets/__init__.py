# Copyright (c) Facebook, Inc. and its affiliates.
from .scannet import ScannetDetectionDataset, ScannetDatasetConfig
from .scannet_base import ScannetDetectionDataset_base, ScannetDatasetConfig_base
from .scannet_incremental import ScannetDetectionDataset_incremental, ScannetDatasetConfig_incremental

from .sunrgbd import SunrgbdDetectionDataset, SunrgbdDatasetConfig

# TODO implement scannet_incremental

DATASET_FUNCTIONS = {
    "scannet": [ScannetDetectionDataset, ScannetDatasetConfig],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
}

DATASET_FUNCTIONS_BASE = {
    "scannet": [ScannetDetectionDataset_base, ScannetDatasetConfig_base],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
}

DATASET_FUNCTIONS_INCREMENTAL = {
    "scannet": [ScannetDetectionDataset_incremental, ScannetDatasetConfig_incremental],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
}

def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()

    dataset_dict = {
        "train": dataset_builder(
            dataset_config,
            split_set="train",
            root_dir=args.dataset_root_dir,
            meta_data_dir=args.meta_data_dir,
            use_color=args.use_color,
            augment=True
        ),
        "test": dataset_builder(
            dataset_config,
            split_set="val",
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            augment=False,
        ),
    }
    return dataset_dict, dataset_config

def build_dataset_base(args):
    dataset_builder = DATASET_FUNCTIONS_BASE[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS_BASE[args.dataset_name][1](num_base_class = args.num_base_class)

    dataset_dict = {
        "train": dataset_builder(
            dataset_config,
            split_set="train",
            root_dir=args.dataset_root_dir,
            meta_data_dir=args.meta_data_dir,
            use_color=args.use_color,
            augment=True
        ),
        "test": dataset_builder(
            dataset_config,
            split_set="val",
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            augment=False,
        ),
    }
    return dataset_dict, dataset_config

def build_dataset_incremental(args):
    dataset_builder = DATASET_FUNCTIONS_INCREMENTAL[args.dataset_name][0]
    dataset_config_train = DATASET_FUNCTIONS_INCREMENTAL[args.dataset_name][1](num_base_class = args.num_base_class, num_novel_class = args.num_novel_class)
    dataset_config_val = DATASET_FUNCTIONS_BASE[args.dataset_name][1](num_base_class = args.num_base_class + args.num_novel_class)
    # note that the val dataset covers both base and incremental classes

    dataset_dict = {
        "train": dataset_builder(
            dataset_config_train,
            split_set="train",
            root_dir=args.dataset_root_dir,
            meta_data_dir=args.meta_data_dir,
            use_color=args.use_color,
            augment=True
        ),
        "test": dataset_builder(
            dataset_config_val,
            split_set="val",
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            augment=False,
        ),
    }
    return dataset_dict, dataset_config_train, dataset_config_val
