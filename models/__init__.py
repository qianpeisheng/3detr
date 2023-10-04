# Copyright (c) Facebook, Inc. and its affiliates.
from .model_3detr import build_3detr
from .model_3detr_distill import build_3detr_distill

MODEL_FUNCS = {
    "3detr": build_3detr,
    "3detr_distill": build_3detr_distill,
}


def build_model(args, dataset_config, dataset_config_val=None):
    model, processor = MODEL_FUNCS[args.model_name](
        args, dataset_config, dataset_config_val)
    return model, processor
