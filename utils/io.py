# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import os
from utils.dist import is_primary


def save_checkpoint(
    checkpoint_dir,
    model_no_ddp,
    optimizer,
    epoch,
    args,
    best_val_metrics,
    filename=None,
):
    if not is_primary():
        return
    if filename is None:
        filename = f"checkpoint_{epoch:04d}.pth"
    checkpoint_name = os.path.join(checkpoint_dir, filename)

    sd = {
        "model": model_no_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": args,
        "best_val_metrics": best_val_metrics,
    }
    torch.save(sd, checkpoint_name)


def resume_if_possible(checkpoint_dir, model_no_ddp, optimizer, checkpoint_name="checkpoint_0300.pth"):
    """
    Resume if checkpoint is available.
    Return
    - epoch of loaded checkpoint.
    """
    epoch = -1
    best_val_metrics = {}
    if not os.path.isdir(checkpoint_dir):
        return epoch, best_val_metrics

    last_checkpoint = os.path.join(checkpoint_dir, checkpoint_name)
    if not os.path.isfile(last_checkpoint):
        return epoch, best_val_metrics
    print('last_checkpoint', last_checkpoint)
    print('resuming ............')
    sd = torch.load(last_checkpoint, map_location=torch.device("cpu"))
    epoch = sd["epoch"]
    best_val_metrics = sd["best_val_metrics"]
    print(f"Found checkpoint at {epoch}. Resuming.")

    model_no_ddp.load_state_dict(sd["model"])
    optimizer.load_state_dict(sd["optimizer"])
    print(
        f"Loaded model and optimizer state at {epoch}. Loaded best val metrics so far."
    )
    return epoch, best_val_metrics

def resume_if_possible_incremental(checkpoint_dir, model_no_ddp, optimizer, checkpoint_name="base_checkpoint.pth"):
    """
    Resume if checkpoint is available.
    Return
    - epoch of loaded checkpoint.
    """
    epoch = -1
    best_val_metrics = {}
    if not os.path.isdir(checkpoint_dir):
        return epoch, best_val_metrics

    last_checkpoint = os.path.join(checkpoint_dir, checkpoint_name)
    if not os.path.isfile(last_checkpoint):
        return epoch, best_val_metrics

    sd = torch.load(last_checkpoint, map_location=torch.device("cpu"))
    # epoch = sd["epoch"]
    sd['epoch'] = 0 # reset epoch to 0
    epoch = sd["epoch"]
    best_val_metrics = sd["best_val_metrics"]

    # set all values in best_val_metrics to 0, note that the best_val_metrics is a dict and could be nested
    def set_zero(d):
        for k, v in d.items():
            if isinstance(v, dict):
                set_zero(v)
            else:
                d[k] = 0

    print(f"Found checkpoint at {epoch}. Resuming.")


    model_no_ddp.load_state_dict(sd["model"])

    # do not load optimizer for incremental training as the optimizers are not the same when loading from different epochs of base training.
    # optimizer.load_state_dict(sd["optimizer"])
    print(
        # f"Loaded model and optimizer state at {epoch}. Loaded best val metrics so far."
        f"Loaded model at {epoch}. Loaded best val metrics so far. NOT loading the optimizer."
    )
    return epoch, best_val_metrics
