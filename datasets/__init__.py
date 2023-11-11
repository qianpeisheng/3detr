# Copyright (c) Facebook, Inc. and its affiliates.
from .scannet import ScannetDetectionDataset, ScannetDatasetConfig
from .scannet_base import ScannetDetectionDataset_base, ScannetDatasetConfig_base
from .scannet_incremental import ScannetDetectionDataset_incremental, ScannetDatasetConfig_incremental
from .scannet_SDCoT import ScannetDetectionDataset_SDCoT, ScannetDatasetConfig_SDCoT
from .scannet_Pseudo_EMA import ScannetDetectionDataset_Pseudo_EMA, ScannetDatasetConfig_Pseudo_EMA
from .scannet_Pseudo_2_source_EMA import ScannetDetectionDataset_Pseudo_2_source_EMA, ScannetDatasetConfig_Pseudo_2_source_EMA
from .scannet_Pseudo_2_source_EMA_free_match import ScannetDetectionDataset_Pseudo_2_source_EMA_free_match, ScannetDatasetConfig_Pseudo_2_source_EMA_free_match
from .scannet_Pseudo_2_source_EMA_v2 import ScannetDetectionDataset_Pseudo_2_source_EMA_v2, ScannetDatasetConfig_Pseudo_2_source_EMA_v2

from .sunrgbd import SunrgbdDetectionDataset, SunrgbdDatasetConfig
from .sunrgbd_base import SunrgbdDetectionDataset_base, SunrgbdDatasetConfig_base

# TODO implement scannet_incremental

DATASET_FUNCTIONS = {
    "scannet": [ScannetDetectionDataset, ScannetDatasetConfig],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
}

DATASET_FUNCTIONS_BASE = {
    "scannet": [ScannetDetectionDataset_base, ScannetDatasetConfig_base],
    "sunrgbd": [SunrgbdDetectionDataset_base, SunrgbdDatasetConfig_base],
}

DATASET_FUNCTIONS_INCREMENTAL = {
    "scannet": [ScannetDetectionDataset_incremental, ScannetDatasetConfig_incremental],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
}

DATASET_FUNCTIONS_SDCoT = {
    "scannet": [ScannetDetectionDataset_SDCoT, ScannetDatasetConfig_SDCoT],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
}

DATASET_FUNCTIONS_Pseudo_EMA = {
    "scannet": [ScannetDetectionDataset_Pseudo_EMA, ScannetDatasetConfig_Pseudo_EMA],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
}

DATASET_FUNCTIONS_Pseudo_2_source_EMA = {
    "scannet": [ScannetDetectionDataset_Pseudo_2_source_EMA, ScannetDatasetConfig_Pseudo_2_source_EMA],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
}

DATASET_FUNCTIONS_Pseudo_2_source_EMA_v2 = {
    "scannet": [ScannetDetectionDataset_Pseudo_2_source_EMA_v2, ScannetDatasetConfig_Pseudo_2_source_EMA_v2],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
}

DATASET_FUNCTIONS_Pseudo_2_source_EMA_free_match = {
    "scannet": [ScannetDetectionDataset_Pseudo_2_source_EMA_free_match, ScannetDatasetConfig_Pseudo_2_source_EMA_free_match],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
}


def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()
    try:
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
    except TypeError:
        # for sunrgbd there is no meta_data_dir
        dataset_dict = {
            "train": dataset_builder(
                dataset_config,
                split_set="train",
                root_dir=args.dataset_root_dir,
                # meta_data_dir=args.meta_data_dir,
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
    dataset_config = DATASET_FUNCTIONS_BASE[args.dataset_name][1](
        num_base_class=args.num_base_class)
    try:
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
    except TypeError:
        # for sunrgbd there is no meta_data_dir
        dataset_dict = {
            "train": dataset_builder(
                dataset_config,
                split_set="train",
                root_dir=args.dataset_root_dir,
                # meta_data_dir=args.meta_data_dir,
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
    dataset_config_train = DATASET_FUNCTIONS_INCREMENTAL[args.dataset_name][1](
        num_base_class=args.num_base_class, num_novel_class=args.num_novel_class)
    dataset_config_val = DATASET_FUNCTIONS_BASE[args.dataset_name][1](
        num_base_class=args.num_base_class + args.num_novel_class)
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


def build_dataset_SDCoT(args):
    # TODO The current implementation is not correct. the train dataset should not load base classes labels.
    dataset_builder_train = DATASET_FUNCTIONS_SDCoT[args.dataset_name][0]
    dataset_builder_test = DATASET_FUNCTIONS_BASE[args.dataset_name][0]
    dataset_config_train = DATASET_FUNCTIONS_SDCoT[args.dataset_name][1](
        num_base_class=args.num_base_class, num_novel_class=args.num_novel_class)
    dataset_config_base = DATASET_FUNCTIONS_BASE[args.dataset_name][1](
        num_base_class=args.num_base_class)
    dataset_config_val = DATASET_FUNCTIONS_BASE[args.dataset_name][1](
        num_base_class=args.num_base_class + args.num_novel_class)
    # note that the val dataset covers both base and incremental classes

    dataset_dict = {
        "train": dataset_builder_train(
            dataset_config_train,
            split_set="train",
            root_dir=args.dataset_root_dir,
            meta_data_dir=args.meta_data_dir,
            use_color=args.use_color,
            augment=True
        ),
        "test": dataset_builder_test(
            dataset_config_val,
            split_set="val",
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            augment=False,
        ),
    }
    return dataset_dict, dataset_config_train, dataset_config_val, dataset_config_base


def build_dataset_Pseudo_EMA(args):
    # TODO The current implementation is not correct. the train dataset should not load base classes labels.
    dataset_builder_train = DATASET_FUNCTIONS_Pseudo_EMA[args.dataset_name][0]
    dataset_builder_test = DATASET_FUNCTIONS_BASE[args.dataset_name][0]
    dataset_config_train = DATASET_FUNCTIONS_Pseudo_EMA[args.dataset_name][1](
        num_base_class=args.num_base_class, num_novel_class=args.num_novel_class)
    dataset_config_base = DATASET_FUNCTIONS_BASE[args.dataset_name][1](
        num_base_class=args.num_base_class)
    dataset_config_val = DATASET_FUNCTIONS_BASE[args.dataset_name][1](
        num_base_class=args.num_base_class + args.num_novel_class)
    # note that the val dataset covers both base and incremental classes

    dataset_dict = {
        "train": dataset_builder_train(
            dataset_config_train,
            split_set="train",
            root_dir=args.dataset_root_dir,
            meta_data_dir=args.meta_data_dir,
            use_color=args.use_color,
            augment=True,
            use_cls_threshold=args.use_cls_threshold,
        ),
        "test": dataset_builder_test(
            dataset_config_val,
            split_set="val",
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            augment=False,
        ),
    }
    return dataset_dict, dataset_config_train, dataset_config_val, dataset_config_base


def build_dataset_Pseudo_2_source_EMA(args):
    # TODO The current implementation is not correct. the train dataset should not load base classes labels.
    dataset_builder_train = DATASET_FUNCTIONS_Pseudo_2_source_EMA[args.dataset_name][0]
    dataset_builder_test = DATASET_FUNCTIONS_BASE[args.dataset_name][0]
    dataset_config_train = DATASET_FUNCTIONS_Pseudo_2_source_EMA[args.dataset_name][1](
        num_base_class=args.num_base_class, num_novel_class=args.num_novel_class)
    dataset_config_base = DATASET_FUNCTIONS_BASE[args.dataset_name][1](
        num_base_class=args.num_base_class)
    dataset_config_val = DATASET_FUNCTIONS_BASE[args.dataset_name][1](
        num_base_class=args.num_base_class + args.num_novel_class)
    # note that the val dataset covers both base and incremental classes

    dataset_dict = {
        "train": dataset_builder_train(
            dataset_config_train,
            split_set="train",
            root_dir=args.dataset_root_dir,
            meta_data_dir=args.meta_data_dir,
            use_color=args.use_color,
            augment=True,
            use_ema_pseudo_label=args.use_ema_pseudo_label,
            nms_threshold=args.ema_nms_threshold,
            use_cls_threshold=args.use_cls_threshold,
        ),
        "test": dataset_builder_test(
            dataset_config_val,
            split_set="val",
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            augment=False,
        ),
    }
    return dataset_dict, dataset_config_train, dataset_config_val, dataset_config_base


def build_dataset_Pseudo_2_source_EMA_v2(args):
    # TODO The current implementation is not correct. the train dataset should not load base classes labels.
    dataset_builder_train = DATASET_FUNCTIONS_Pseudo_2_source_EMA_v2[args.dataset_name][0]
    dataset_builder_test = DATASET_FUNCTIONS_BASE[args.dataset_name][0]
    dataset_config_train = DATASET_FUNCTIONS_Pseudo_2_source_EMA_v2[args.dataset_name][1](
        num_base_class=args.num_base_class, num_novel_class=args.num_novel_class)
    dataset_config_base = DATASET_FUNCTIONS_BASE[args.dataset_name][1](
        num_base_class=args.num_base_class)
    dataset_config_val = DATASET_FUNCTIONS_BASE[args.dataset_name][1](
        num_base_class=args.num_base_class + args.num_novel_class)
    # note that the val dataset covers both base and incremental classes

    dataset_dict = {
        "train": dataset_builder_train(
            dataset_config_train,
            split_set="train",
            root_dir=args.dataset_root_dir,
            meta_data_dir=args.meta_data_dir,
            use_color=args.use_color,
            augment=True,
            use_ema_pseudo_label=args.use_ema_pseudo_label,
            nms_threshold=args.ema_nms_threshold
        ),
        "test": dataset_builder_test(
            dataset_config_val,
            split_set="val",
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            augment=False,
        ),
    }
    return dataset_dict, dataset_config_train, dataset_config_val, dataset_config_base


def build_dataset_Pseudo_2_source_EMA_free_match(args):
    # TODO The current implementation is not correct. the train dataset should not load base classes labels.
    dataset_builder_train = DATASET_FUNCTIONS_Pseudo_2_source_EMA_free_match[
        args.dataset_name][0]
    dataset_builder_test = DATASET_FUNCTIONS_BASE[args.dataset_name][0]
    dataset_config_train = DATASET_FUNCTIONS_Pseudo_2_source_EMA_free_match[args.dataset_name][1](
        num_base_class=args.num_base_class, num_novel_class=args.num_novel_class)
    dataset_config_base = DATASET_FUNCTIONS_BASE[args.dataset_name][1](
        num_base_class=args.num_base_class)
    dataset_config_val = DATASET_FUNCTIONS_BASE[args.dataset_name][1](
        num_base_class=args.num_base_class + args.num_novel_class)
    # note that the val dataset covers both base and incremental classes

    dataset_dict = {
        "train": dataset_builder_train(
            dataset_config_train,
            split_set="train",
            root_dir=args.dataset_root_dir,
            meta_data_dir=args.meta_data_dir,
            use_color=args.use_color,
            augment=True,
            use_ema_pseudo_label=args.use_ema_pseudo_label,
            nms_threshold=args.ema_nms_threshold
        ),
        "test": dataset_builder_test(
            dataset_config_val,
            split_set="val",
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            augment=False,
        ),
    }
    return dataset_dict, dataset_config_train, dataset_config_val, dataset_config_base


def build_dataset_Pseudo_2_source_EMA_v2(args):
    # TODO The current implementation is not correct. the train dataset should not load base classes labels.
    dataset_builder_train = DATASET_FUNCTIONS_Pseudo_2_source_EMA_v2[args.dataset_name][0]
    dataset_builder_test = DATASET_FUNCTIONS_BASE[args.dataset_name][0]
    dataset_config_train = DATASET_FUNCTIONS_Pseudo_2_source_EMA_v2[args.dataset_name][1](
        num_base_class=args.num_base_class, num_novel_class=args.num_novel_class)
    dataset_config_base = DATASET_FUNCTIONS_BASE[args.dataset_name][1](
        num_base_class=args.num_base_class)
    dataset_config_val = DATASET_FUNCTIONS_BASE[args.dataset_name][1](
        num_base_class=args.num_base_class + args.num_novel_class)
    # note that the val dataset covers both base and incremental classes

    dataset_dict = {
        "train": dataset_builder_train(
            dataset_config_train,
            split_set="train",
            root_dir=args.dataset_root_dir,
            meta_data_dir=args.meta_data_dir,
            use_color=args.use_color,
            augment=True,
            use_ema_pseudo_label=args.use_ema_pseudo_label,
            nms_threshold=args.ema_nms_threshold
        ),
        "test": dataset_builder_test(
            dataset_config_val,
            split_set="val",
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            augment=False,
        ),
    }
    return dataset_dict, dataset_config_train, dataset_config_val, dataset_config_base


def build_dataset_Pseudo_2_source_EMA_free_match(args):
    # TODO The current implementation is not correct. the train dataset should not load base classes labels.
    dataset_builder_train = DATASET_FUNCTIONS_Pseudo_2_source_EMA_free_match[
        args.dataset_name][0]
    dataset_builder_test = DATASET_FUNCTIONS_BASE[args.dataset_name][0]
    dataset_config_train = DATASET_FUNCTIONS_Pseudo_2_source_EMA_free_match[args.dataset_name][1](
        num_base_class=args.num_base_class, num_novel_class=args.num_novel_class)
    dataset_config_base = DATASET_FUNCTIONS_BASE[args.dataset_name][1](
        num_base_class=args.num_base_class)
    dataset_config_val = DATASET_FUNCTIONS_BASE[args.dataset_name][1](
        num_base_class=args.num_base_class + args.num_novel_class)
    # note that the val dataset covers both base and incremental classes

    dataset_dict = {
        "train": dataset_builder_train(
            dataset_config_train,
            split_set="train",
            root_dir=args.dataset_root_dir,
            meta_data_dir=args.meta_data_dir,
            use_color=args.use_color,
            augment=True,
            use_ema_pseudo_label=args.use_ema_pseudo_label,
            nms_threshold=args.ema_nms_threshold
        ),
        "test": dataset_builder_test(
            dataset_config_val,
            split_set="val",
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            augment=False,
        ),
    }
    return dataset_dict, dataset_config_train, dataset_config_val, dataset_config_base
