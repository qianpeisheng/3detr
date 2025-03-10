# Copyright (c) Facebook, Inc. and its affiliates.
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from utils.pc_util import scale_points, shift_scale_points

from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine
from models.transformer import (MaskedTransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer)


class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(
            size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1:
            # special case for datasets with no rotation angle
            # we still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        # The following assert is not true for fine-tuning evaluation.

        # assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        return self.dataset_config.box_parametrization_to_corners(
            box_center_unnorm, box_size_unnorm, box_angle
        )


class Model3DETR(nn.Module):
    """
    Main 3DETR model. Consists of the following learnable sub-models
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
        self,
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        dataset_config_val,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        num_queries=256,
    ):
        super().__init__()
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.decoder = decoder
        self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

        self.num_queries = num_queries
        self.box_processor = BoxProcessor(dataset_config)

    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )
        # Semantic class of the box
        if hasattr(dataset_config, 'num_semcls_novel'):
            # TODO  Deprecated. Remove this branch.
            # semcls_head should output num_novel_class + 1 classes
            # but we load the weights and bias of the base model to the current model
            # and will deroll the weights and bias.
            semcls_head = mlp_func(
                output_dim=dataset_config.num_semcls_base + 1)
        else:
            # semcls_head should output num_semcls + 1 classes
            semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)
        # add 1 for background/not-an-object class
        # semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)

        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)
        angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = furthest_point_sample(encoder_xyz, self.num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds)
                     for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed, pos_embed

    def static_teacher_get_query_embeddings(self, pos_embed):
        # already have pos_embed
        query_embed = self.query_projection(pos_embed)
        return query_embed

    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(
            1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds, student_inds=None, interim_inds=None):
        xyz, features = self._break_up_pc(point_clouds)
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(
            xyz=xyz, features=features, inds=student_inds)
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(
            src=pre_enc_features, xyz=pre_enc_xyz, interim_inds=interim_inds
        )

        if student_inds is not None:
            # for static teacher
            if enc_inds is None:
                # encoder does not perform any downsampling, vallina 3DETR
                enc_inds = pre_enc_inds
            # else:
                # masked encoder
                # use gather here to ensure that it works for both FPS and random sampling
                # enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.type(torch.int64))
                # the line above works because enc_inds is a subset of pre_enc_inds

            # the returned pre_enc_inds and enc_inds from the static steacher are not used.
            return enc_xyz, enc_features, pre_enc_inds, enc_inds, pre_enc_features

        else:
            # for student
            if enc_inds is None:
                # encoder does not perform any downsampling, vanilla 3DETR
                enc_inds = pre_enc_inds
            # else:
                # use gather here to ensure that it works for both FPS and random sampling
                # enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.type(torch.int64))
            # Note that we do not gather because end_inds from masked student encoder
            # is used to index the pre_enc_inds from the static teacher.
            # This is different from original 3DETR.

            return enc_xyz, enc_features, pre_enc_inds, enc_inds, pre_enc_features

    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(
            num_layers * batch, channel, num_queries)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["sem_cls_head"](
            box_features).transpose(1, 2)
        center_offset = (
            self.mlp_heads["center_head"](
                box_features).sigmoid().transpose(1, 2) - 0.5
        )
        size_normalized = (
            self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        )
        angle_logits = self.mlp_heads["angle_cls_head"](
            box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](
            box_features
        ).transpose(1, 2)

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        center_offset = center_offset.reshape(
            num_layers, batch, num_queries, -1)
        size_normalized = size_normalized.reshape(
            num_layers, batch, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(
            num_layers, batch, num_queries, -1
        )
        angle_residual = angle_residual_normalized * (
            np.pi / angle_residual_normalized.shape[-1]
        )

        outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            (
                center_normalized,
                center_unnormalized,
            ) = self.box_processor.compute_predicted_center(
                center_offset[l], query_xyz, point_cloud_dims
            )
            angle_continuous = self.box_processor.compute_predicted_angle(
                angle_logits[l], angle_residual[l]
            )
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[l], point_cloud_dims
            )
            box_corners = self.box_processor.box_parametrization_to_corners(
                center_unnormalized, size_unnormalized, angle_continuous
            )

            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])

            box_prediction = {
                "sem_cls_logits": cls_logits[l],
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[l],
                "size_unnormalized": size_unnormalized,
                "angle_logits": angle_logits[l],
                "angle_residual": angle_residual[l],
                "angle_residual_normalized": angle_residual_normalized[l],
                "angle_continuous": angle_continuous,
                "objectness_prob": objectness_prob,
                "sem_cls_prob": semcls_prob,
                "box_corners": box_corners,
            }
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }

    def forward(self, inputs, query_xyz=None, pos_embed=None, encoder_only=False, student_inds=None, interim_inds=None):
        point_clouds = inputs["point_clouds"]

        is_static_teacher = pos_embed is not None
        if is_static_teacher:
            enc_xyz, enc_features, pre_enc_inds, interim_inds, pre_enc_features = self.run_encoder(
                point_clouds=point_clouds, student_inds=student_inds, interim_inds=interim_inds)
        else:
            enc_xyz, enc_features, pre_enc_inds, interim_inds, pre_enc_features = self.run_encoder(
                point_clouds=point_clouds)
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3

        if encoder_only:
            # return: batch x npoints x channels
            return enc_xyz, enc_features.transpose(0, 1)

        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]

        if not is_static_teacher:
            query_xyz, query_embed, pos_embed = self.get_query_embeddings(
                enc_xyz, point_cloud_dims)
        else:
            query_embed = self.static_teacher_get_query_embeddings(pos_embed)

        # query_embed: batch x channel x npoint
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder expects: npoints x batch x channel
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        box_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )[0]

        box_predictions = self.get_box_predictions(
            query_xyz, point_cloud_dims, box_features
        )
        return box_predictions, query_xyz, pos_embed, pre_enc_inds, interim_inds, enc_pos, query_embed, box_features, enc_features, enc_xyz, pre_enc_features

    def enroll_weights(self, base_weights, base_bias):
        '''
        Following SDCoT paper, we concat the weights and bias of the base model to the current model.
        The weights and bias of the base model are the weights and bias of the last layer of the base model.
        The weights and bias of the current model are the weights and bias of the last layer of the current model.
        Note that in SDCoT, this function just copy the weights and bias of the base model to the current model without any modification.
        While in our implementation, we concat the weights and bias of the base model to the current model.
        '''
        novel_weights = self.mlp_heads["sem_cls_head"].layers[-1].weight.detach().clone()
        novel_bias = self.mlp_heads["sem_cls_head"].layers[-1].bias.detach().clone()

        # remove the last dimension of base_weights and base_bias because they are for the background class
        base_weights = base_weights[:-1, ...]
        base_bias = base_bias[:-1]

        # define self.mlp_heads["sem_cls_head"].layers[-1] as a new nn.Conv1d layer
        self.mlp_heads["sem_cls_head"].layers[-1] = nn.Conv1d(in_channels=self.mlp_heads["sem_cls_head"].layers[-1].in_channels,
                                                              out_channels=novel_weights.shape[0] + base_weights.shape[0], kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)

        # stack base_weights and novel_weights along the first dimension
        self.mlp_heads["sem_cls_head"].layers[-1].weight = nn.Parameter(
            torch.cat((base_weights, novel_weights), dim=0))

        # stack base_bias and novel_bias along the first dimension
        self.mlp_heads["sem_cls_head"].layers[-1].bias = nn.Parameter(
            torch.cat((base_bias, novel_bias), dim=0))

        return novel_weights, novel_bias

    def deroll_weights(self, novel_weights, novel_bias):
        '''
        Enroll_weights is used for evaluation.
        After evaluation, we need to deroll_weights to restore the model to the original state.
        '''
        # define model.mlp_heads["sem_cls_head"].layers[-1] as a new nn.Conv1d layer
        self.mlp_heads["sem_cls_head"].layers[-1] = nn.Conv1d(in_channels=self.mlp_heads["sem_cls_head"].layers[-1].in_channels,
                                                              out_channels=novel_weights.shape[0], kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)

        # replace the weights and bias of the last layer of the current model with novel_weights and novel_bias
        self.mlp_heads["sem_cls_head"].layers[-1].weight = nn.Parameter(
            novel_weights)
        self.mlp_heads["sem_cls_head"].layers[-1].bias = nn.Parameter(
            novel_bias)

    def deroll_weights_init(self, out_dim):
        # define model.mlp_heads["sem_cls_head"].layers[-1] as a new nn.Conv1d layer
        self.mlp_heads["sem_cls_head"].layers[-1] = nn.Conv1d(in_channels=self.mlp_heads["sem_cls_head"].layers[-1].in_channels,
                                                              out_channels=out_dim, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.mlp_heads["sem_cls_head"].layers[-1].cuda()


def build_preencoder(args):
    mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
    # the preencoder is a Pointnet++ module, i,e. sa1
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,  # this is K in pointnet++ and is a hyperparameter
        npoint=args.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder


def build_encoder(args):
    if args.enc_type == "vanilla":
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=args.enc_nlayers
        )
    elif args.enc_type in ["masked"]:
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        interim_downsampling = PointnetSAModuleVotes(
            radius=0.4,
            nsample=32,
            npoint=args.preenc_npoints // 2,
            mlp=[args.enc_dim, 256, 256, args.enc_dim],
            normalize_xyz=True,
        )

        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
        )
    else:
        raise ValueError(f"Unknown encoder type {args.enc_type}")
    return encoder


def build_decoder(args):
    decoder_layer = TransformerDecoderLayer(
        d_model=args.dec_dim,
        nhead=args.dec_nhead,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
    )
    return decoder


def build_3detr_distill(args, dataset_config, dataset_config_val=None):
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)

    # For base model, dataset_config_val is None.
    # For incremental model, dataset_config_val is the dataset_config of all classes for evaluation.
    if dataset_config_val is None:
        dataset_config_val = dataset_config
    model = Model3DETR(
        pre_encoder,
        encoder,
        decoder,
        dataset_config=dataset_config,
        dataset_config_val=dataset_config_val,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
    )
    output_processor = BoxProcessor(dataset_config)
    return model, output_processor
