# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import copy
import os
import sys
import pickle

import numpy as np
import torch
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler

# 3DETR codebase specific imports
from datasets import build_dataset_Pseudo_2_source_EMA_free_match  # build_dataset_SDCoT
from engine_distill_ema_2_source_match import evaluate, train_one_epoch, evaluate_incremental
from models import build_model
from optimizer import build_optimizer
from criterion_distill_ema import build_criterion
from utils.dist import init_distributed, is_distributed, is_primary, get_rank, barrier
from utils.misc import my_worker_init_fn
from utils.io import save_checkpoint, resume_if_possible, resume_if_possible_SDCoT
from utils.logger import Logger
from utils.ap_calculator_match import get_ap_config_dict

torch.autograd.set_detect_anomaly(True)


def make_args_parser():
    parser = argparse.ArgumentParser(
        "3D Detection Using Transformers", add_help=False)

    ##### Optimizer #####
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--filter_biases_wd",
                        default=False, action="store_true")
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
    )

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="3detr",
        type=str,
        help="Name of the model",
        choices=["3detr", "3detr_distill"],
    )
    # Encoder
    parser.add_argument(
        "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)

    # Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    # MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    # Other model params
    parser.add_argument("--preenc_npoints", default=2048, type=int)
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=256, type=int)
    parser.add_argument("--use_color", default=False, action="store_true")

    # add if freeze all model parameters
    parser.add_argument("--freeze", default=False, action="store_true")

    ##### Set Loss #####
    # Matcher
    parser.add_argument("--matcher_giou_cost", default=2, type=float)
    parser.add_argument("--matcher_cls_cost", default=1, type=float)
    parser.add_argument("--matcher_center_cost", default=0, type=float)
    parser.add_argument("--matcher_objectness_cost", default=0, type=float)

    # Loss Weights
    parser.add_argument("--loss_giou_weight", default=0, type=float)
    parser.add_argument("--loss_sem_cls_weight", default=1, type=float)
    parser.add_argument(
        "--loss_no_object_weight", default=0.2, type=float
    )  # "no object" or "background" class for detection
    parser.add_argument("--loss_angle_cls_weight", default=0.1, type=float)
    parser.add_argument("--loss_angle_reg_weight", default=0.5, type=float)
    parser.add_argument("--loss_center_weight", default=5.0, type=float)
    parser.add_argument("--loss_size_weight", default=1.0, type=float)

    ##### Dataset #####
    parser.add_argument(
        "--dataset_name", required=True, type=str, choices=["scannet", "sunrgbd"]
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument(
        "--meta_data_dir",
        type=str,
        default=None,
        help="Root directory containing the metadata files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)

    # SDCoT specific args
    parser.add_argument("--num_base_class", default=9, type=int)
    parser.add_argument("--num_novel_class", default=0, type=int)
    parser.add_argument('--loss_distill_weight', type=float, default=1.0,
                        help='use distillation loss with given weight')
    parser.add_argument('--distillation_ramp_len', type=int,
                        default=100, help='length of the stabilization loss ramp-up')
    # EMA specific args

    parser.add_argument('--loss_center_consistency_weight', type=float, default=0.1,
                        help='use consistency loss with given weight')
    parser.add_argument('--loss_cls_consistency_weight', type=float, default=10.0,
                        help='use consistency loss with given weight')
    parser.add_argument('--loss_size_consistency_weight', type=float, default=1.0,
                        help='use consistency loss with given weight')
    parser.add_argument('--ema_decay', type=float,
                        default=0.999, help='ema variable decay rate')
    parser.add_argument('--consistency_ramp_len', type=int,
                        default=100, help='length of the consistency loss ramp-up')

    # ema pseudo labels
    parser.add_argument("--use_ema_pseudo_label", default=False, action="store_true")
    parser.add_argument("--ema_nms_threshold", type=float, default=0.5, help="nms threshold for ema pseudo labels")

    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=720, type=int)
    parser.add_argument("--eval_every_epoch", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)

    ##### Testing #####
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--test_ckpt", default=None, type=str)

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--checkpoint_name",
                        default="base_checkpoint.pth", type=str)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--log_metrics_every", default=20, type=int)
    parser.add_argument(
        "--save_separate_checkpoint_every_epoch", default=100, type=int)

    ##### Distributed Training #####
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--dist_url", default="tcp://localhost:12345", type=str)

    return parser


def do_train(
    args,
    model,
    model_no_ddp,
    ema_model,
    optimizer,
    criterion_train,
    criterion_val,
    dataset_config_train,
    dataset_config_val,
    dataloaders,
    best_val_metrics,
    static_teacher=None
):
    """
    Main training loop.
    This trains the model for `args.max_epoch` epochs and tests the model after every `args.eval_every_epoch`.
    We always evaluate the final checkpoint and report both the final AP and best AP on the val set.
    """

    num_iters_per_epoch = len(dataloaders["train"])
    num_iters_per_eval_epoch = len(dataloaders["test"])
    print(f"Model is {model}")
    print(
        f"Training started at epoch {args.start_epoch} until {args.max_epoch}.")
    print(f"One training epoch = {num_iters_per_epoch} iters.")
    print(f"One eval epoch = {num_iters_per_eval_epoch} iters.")

    final_eval = os.path.join(args.checkpoint_dir, "final_eval.txt")
    final_eval_pkl = os.path.join(args.checkpoint_dir, "final_eval.pkl")

    if os.path.isfile(final_eval):
        print(f"Found final eval file {final_eval}. Skipping training.")
        return

    logger = Logger(args.checkpoint_dir)

    for epoch in range(args.start_epoch, args.max_epoch):
        if is_distributed():
            dataloaders["train_sampler"].set_epoch(epoch)

        aps = train_one_epoch(
            args,
            epoch,
            model,
            ema_model,
            optimizer,
            criterion_train,
            dataset_config_train,
            dataloaders["train"],
            logger,
            static_teacher
        )

        # latest checkpoint is always stored in checkpoint.pth
        save_checkpoint(
            args.checkpoint_dir,
            model_no_ddp,
            optimizer,
            epoch,
            args,
            best_val_metrics,
            filename="checkpoint.pth",
        )

        metrics = aps.compute_metrics()
        metric_str = aps.metrics_to_str(metrics, per_class=False) # not using per class to save space
        metrics_dict = aps.metrics_to_dict(metrics)
        curr_iter = epoch * len(dataloaders["train"])
        if is_primary():
            print("==" * 10)
            print(f"Epoch [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
            print("==" * 10)
            logger.log_scalars(metrics_dict, curr_iter, prefix="Train/")

        if (
            epoch > 0
            and args.save_separate_checkpoint_every_epoch > 0
            and epoch % args.save_separate_checkpoint_every_epoch == 0
        ):
            # separate checkpoints are stored as checkpoint_{epoch}.pth
            save_checkpoint(
                args.checkpoint_dir,
                model_no_ddp,
                optimizer,
                epoch,
                args,
                best_val_metrics,
            )

            # save the ema model
            save_checkpoint(
                args.checkpoint_dir,
                ema_model,
                optimizer,
                epoch,
                args,
                best_val_metrics,
                filename=f"EMA_checkpoint_{epoch:04d}.pth",
            )

        if epoch % args.eval_every_epoch == 0 or epoch == (args.max_epoch - 1) or epoch == 1:
            # evaluate the model at epoch 1 for sanity check

            ap_calculator = evaluate_incremental(
                args=args,
                curr_epoch=epoch,
                model=model,
                criterion=None,  # do not compute loss for speed-up; Comment out to see test loss
                dataset_config=dataset_config_val,
                dataset_loader=dataloaders["test"],
                logger=logger,
                curr_train_iter=curr_iter,
                test_prefix="Student ",
            )
            metrics = ap_calculator.compute_metrics()
            ap25 = metrics[0.25]["mAP"]
            metric_str = ap_calculator.metrics_to_str(metrics, per_class=True)
            metrics_dict = ap_calculator.metrics_to_dict(metrics)
            if is_primary():
                print("==" * 10)
                print(
                    f"Evaluate Epoch [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
                print("==" * 10)
                logger.log_scalars(metrics_dict, curr_iter, prefix="Test/")

            if is_primary() and (
                len(
                    best_val_metrics) == 0 or best_val_metrics[0.25]["mAP"] < ap25 or epoch < 2
            ):
                print('Best val metrics updated.')
                # Note that the loaded best_val_metrics is for base classes, which is
                # not directly comparable to the current metrics for base + novel classes.
                # So we replace the loaded best_val_metrics with the current metrics.
                best_val_metrics = metrics
                filename = "checkpoint_best.pth"
                save_checkpoint(
                    args.checkpoint_dir,
                    model_no_ddp,
                    optimizer,
                    epoch,
                    args,
                    best_val_metrics,
                    filename=filename,
                )
                print(
                    f"Epoch [{epoch}/{args.max_epoch}] saved current best val checkpoint at {filename}; ap25 {ap25}"
                )

            # repeat the evaluation with the ema model
            ap_calculator_ema = evaluate_incremental(
                args=args,
                curr_epoch=epoch,
                model=ema_model,
                criterion=None,  # do not compute loss for speed-up; Comment out to see test loss
                dataset_config=dataset_config_val,
                dataset_loader=dataloaders["test"],
                logger=logger,
                curr_train_iter=curr_iter,
                test_prefix="EMA",
            )
            metrics_ema = ap_calculator_ema.compute_metrics()
            ap25_ema = metrics_ema[0.25]["mAP"]
            metric_str_ema = ap_calculator_ema.metrics_to_str(
                metrics_ema, per_class=True)
            metrics_dict_ema = ap_calculator_ema.metrics_to_dict(metrics_ema)
            if is_primary():
                print("==" * 10)
                print(
                    f"EMA Evaluate Epoch [{epoch}/{args.max_epoch}]; Metrics {metric_str_ema}")
                print("==" * 10)
                logger.log_scalars(
                    metrics_dict_ema, curr_iter, prefix="Test EMA/")
            if is_primary() and (
                len(
                    best_val_metrics) == 0 or best_val_metrics[0.25]["mAP"] < ap25_ema
            ):
                best_val_metrics = metrics_ema
                filename = "checkpoint_best.pth"
                save_checkpoint(
                    args.checkpoint_dir,
                    model_no_ddp,
                    optimizer,
                    epoch,
                    args,
                    best_val_metrics,
                    filename=filename,
                )
                print(
                    f"EMA Epoch [{epoch}/{args.max_epoch}] saved current best val checkpoint at {filename}; ap25 {ap25_ema}"
                )

    # always evaluate last checkpoint
    epoch = args.max_epoch - 1
    curr_iter = epoch * len(dataloaders["train"])
    ap_calculator = evaluate_incremental(
        args,
        epoch,
        model,
        # criterion_val,
        None,  # do not compute loss for speed-up; Comment out to see test loss
        dataset_config_val,
        dataloaders["test"],
        logger,
        curr_iter,
    )
    metrics = ap_calculator.compute_metrics()
    metric_str = ap_calculator.metrics_to_str(metrics)
    if is_primary():
        print("==" * 10)
        print(
            f"Evaluate Final [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
        print("==" * 10)

        with open(final_eval, "w") as fh:
            fh.write("Training Finished.\n")
            fh.write("==" * 10)
            fh.write("Final Eval Numbers.\n")
            fh.write(metric_str)
            fh.write("\n")
            fh.write("==" * 10)
            fh.write("Best Eval Numbers.\n")
            fh.write(ap_calculator.metrics_to_str(best_val_metrics))
            fh.write("\n")

        with open(final_eval_pkl, "wb") as fh:
            pickle.dump(metrics, fh)

    # no need to deroll_weights because we are done with training.

# When testing, the model checkpoint should contains the classifier weights for all classes.


def test_model(args, model, model_no_ddp, criterion_val, dataset_config_val, dataloaders):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))

    # we already loaded the base detection model weights to the model partially
    if args.test_only:
        model_no_ddp.load_state_dict(sd["model"])
    logger = Logger()
    criterion_val = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    ap_calculator = evaluate(
        args,
        epoch,
        model,
        criterion_val,
        dataset_config_val,
        dataloaders["test"],
        logger,
        curr_iter,
    )
    metrics = ap_calculator.compute_metrics()
    metric_str = ap_calculator.metrics_to_str(metrics)
    for iou in [0.25, 0.5]:
        metrics_25 = metrics[iou]
        # metrics_25 is an OrderedDict, select the first k items
        # average values in metrics_25_base is a dict with 9 keys
        # each key has a value of a float
        metrics_25_base = {k: metrics_25[k]
                           for k in list(metrics_25)[:args.num_base_class]}
        average_ap_base_25 = sum(
            metrics_25_base.values()) / len(metrics_25_base)
        print(f'Base mAP {iou}: ', average_ap_base_25)

        metrics_25_novel = {k: metrics_25[k] for k in list(
            metrics_25)[args.num_base_class:18]}
        average_ap_novel_25 = sum(
            metrics_25_novel.values()) / len(metrics_25_novel)
        print(f'novel mAP {iou}: ', average_ap_novel_25)
    if is_primary():
        print("==" * 10)
        print(f"Test model; Metrics {metric_str}")
        print("==" * 10)


def main(local_rank, args):
    if args.ngpus > 1:
        print(
            "Initializing Distributed Training. This is in BETA mode and hasn't been tested thoroughly. Use at your own risk :)"
        )
        print("To get the maximum speed-up consider reducing evaluations on val set by setting --eval_every_epoch to greater than 50")
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend="nccl",
        )

    print(f"Called with args: {args}")
    torch.cuda.set_device(local_rank)
    np.random.seed(args.seed + get_rank())
    torch.manual_seed(args.seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + get_rank())

    # For incremental learning, the train and test dataset are different,
    # The train dataset only contains NOVEL classes.
    # The test dataset contains both base and novel classes.

    datasets, dataset_config_train, dataset_config_val, dataset_config_base = build_dataset_Pseudo_2_source_EMA_free_match(
        args)

    # define the base detection model and load weights
    base_detection_model, _ = build_model(args, dataset_config_base)
    base_detection_model = base_detection_model.cuda(local_rank)  # TODO add ddp

    # set base detection model to eval mode
    base_detection_model.eval()
    # freeze all base detection model parameters
    for name, param in base_detection_model.named_parameters():
        param.requires_grad = False

    # load the base detection model
    if not args.test_only:
        resume_if_possible(
            checkpoint_dir=args.checkpoint_dir, model_no_ddp=base_detection_model, optimizer=None, checkpoint_name=args.checkpoint_name
        )

    # For the train set, set the base detector
    datasets['train'].set_base_detector(base_detection_model)
    ap_config_dict = get_ap_config_dict(
        dataset_config=dataset_config_train, remove_empty_box=True
    )
    # set set_ap_config_dict
    datasets['train'].set_ap_config_dict(ap_config_dict)

    model, _ = build_model(args, dataset_config_train)
    model = model.cuda(local_rank)
    model_no_ddp = model
    # load the base detection model weights to the model partially

    def load_except_classifier(model, base_detection_model):
        # set all weights in model to zero
        for name, param in model.named_parameters():
            param.data.zero_()

        _stat_dict = base_detection_model.state_dict()
        old_cls_size = _stat_dict['mlp_heads.sem_cls_head.layers.8.weight'].shape[0]
        new_cls_size = model.state_dict(
        )['mlp_heads.sem_cls_head.layers.8.weight'].shape[0]
        assert new_cls_size >= old_cls_size
        # copy the old classifier weights to the new classifier weights
        model.state_dict()['mlp_heads.sem_cls_head.layers.8.weight'][:old_cls_size,
                                                                     ...] = _stat_dict['mlp_heads.sem_cls_head.layers.8.weight']
        model.state_dict()['mlp_heads.sem_cls_head.layers.8.bias'][:old_cls_size,
                                                                   ...] = _stat_dict['mlp_heads.sem_cls_head.layers.8.bias']

        # swap the last and the old_cls_size-1 th class weights because the last class is the background class
        model.state_dict()['mlp_heads.sem_cls_head.layers.8.weight'][-1,
                                                                     ...] = _stat_dict['mlp_heads.sem_cls_head.layers.8.weight'][-1, ...]
        model.state_dict()['mlp_heads.sem_cls_head.layers.8.bias'][-1,
                                                                   ...] = _stat_dict['mlp_heads.sem_cls_head.layers.8.bias'][-1, ...]
        model.state_dict()['mlp_heads.sem_cls_head.layers.8.weight'][old_cls_size - 1,
                                                                     ...] *= 0.
        model.state_dict()['mlp_heads.sem_cls_head.layers.8.bias'][old_cls_size - 1,
                                                                   ...] *= 0.

        # remove mlp_heads.sem_cls_head.layers.8.weight and bias from the state dict
        _stat_dict.pop('mlp_heads.sem_cls_head.layers.8.weight')
        _stat_dict.pop('mlp_heads.sem_cls_head.layers.8.bias')
        # _state_dict does not contain the classifier weights
        model.load_state_dict(_stat_dict, strict=False)
        return model

    # if not test only load the base detection model weights to the model partially
    if not args.test_only:
        model = load_except_classifier(model, base_detection_model)
    # model.load_state_dict(base_detection_model.state_dict(), strict=False)

    if is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )

    # create an ema model
    ema_model = copy.deepcopy(model)
    ema_model.cuda(local_rank)
    for param in ema_model.parameters():
        param.detach_()

    # train dataset set the ema detector
    datasets['train'].set_ema_detector(ema_model)

    criterion_train = build_criterion(args, dataset_config_train)
    criterion_train = criterion_train.cuda(local_rank)

    criterion_val = build_criterion(args, dataset_config_val)
    criterion_val = criterion_val.cuda(local_rank)

    dataloaders = {}
    if args.test_only:
        dataset_splits = ["test"]
    else:
        dataset_splits = ["train", "test"]
    for split in dataset_splits:
        if split == "train":
            shuffle = True
        else:
            shuffle = False
        if is_distributed():
            sampler = DistributedSampler(datasets[split], shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(datasets[split])
        else:
            sampler = torch.utils.data.SequentialSampler(datasets[split])

        dataloaders[split] = DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=args.dataset_num_workers,
            worker_init_fn=my_worker_init_fn,
        )
        dataloaders[split + "_sampler"] = sampler

    if args.test_only:
        criterion_val = None  # faster evaluation
        test_model(args, model, model_no_ddp, criterion_val,
                   dataset_config_val, dataloaders)
    else:
        assert (
            args.checkpoint_dir is not None
        ), f"Please specify a checkpoint dir using --checkpoint_dir"
        if is_primary() and not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        optimizer = build_optimizer(args, model_no_ddp)
        loaded_epoch, best_val_metrics = resume_if_possible_SDCoT(
            args.checkpoint_dir, model_no_ddp, optimizer, checkpoint_name=args.checkpoint_name, num_cls_novel=args.num_novel_class,
            num_cls_base=args.num_base_class
        )

        # freeze is not used for SDCoT
        # freeze all model parameters except classifier weights
        # if args.freeze:
        #     for name, param in model_no_ddp.named_parameters():
        #         if 'mlp_heads' not in name:
        #             param.requires_grad = False
        #         else:
        #             print('not freezing ', name)

        args.start_epoch = loaded_epoch + 1
        do_train(
            args,
            model,
            model_no_ddp,
            ema_model,
            optimizer,
            criterion_train,
            criterion_val,
            dataset_config_train,
            dataset_config_val,
            dataloaders,
            best_val_metrics,
            static_teacher=base_detection_model
        )


def launch_distributed(args):
    world_size = args.ngpus
    if world_size == 1:
        main(local_rank=0, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=world_size, args=(args,))


if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    launch_distributed(args)
