# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""Video models."""

import math
import random
from functools import partial

import memvit.utils.weight_init_helper as init_helper
import torch
import torch.nn as nn
from memvit.models.attention import MultiScaleBlock
from memvit.models.sgconv_helper import ConvMVITBlock
from memvit.models.batchnorm_helper import get_norm
from memvit.models.utils import round_width, validate_checkpoint_wrapper_import
from torch.nn.init import trunc_normal_

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY

import memvit.utils.logging as logging
logger = logging.get_logger(__name__)

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None

@MODEL_REGISTRY.register()
class MViT(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST
        self.use_online_memory = cfg.MEMVIT.ENABLE
        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        use_2d_patch = cfg.MVIT.PATCH_2D
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        self.enable_detection = cfg.DETECTION.ENABLE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        # Prepare output.
        if cfg.MODEL.NUM_CLASSES_LIST:
            num_classes = cfg.MODEL.NUM_CLASSES_LIST
        else:
            num_classes = cfg.MODEL.NUM_CLASSES

        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        self.box_depth = box_depth = cfg.MVIT.BOX_DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        box_drop_path_rate = cfg.MVIT.BOX_DROPPATH_RATE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        # Params for positional embedding
        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        self.rel_pos_spatial = cfg.MVIT.REL_POS_SPATIAL
        self.rel_pos_temporal = cfg.MVIT.REL_POS_TEMPORAL

        self.conv_q = cfg.MVIT.CONV_Q
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )
        if cfg.MODEL.ACT_CHECKPOINT:
            self.patch_embed = checkpoint_wrapper(self.patch_embed)

        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        box_dpr = [
            x.item() for x in torch.linspace(0, box_drop_path_rate, box_depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(1, self.patch_dims[1] * self.patch_dims[2], embed_dim)
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_dim, embed_dim))

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not [], initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[i][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        input_size = self.patch_dims
        self.x_shapes = [[] for i in range(cfg.MVIT.DEPTH + 1)]
        self.x_shapes[0] = input_size
        
        for i in range(cfg.MVIT.DEPTH):
            curr_kernel = pool_q[i] if len(pool_q) > 0 else [1,1,1]
            curr_stride = stride_q[i] if len(stride_q) > 0 else [1,1,1]
            curr_pad = [int(kernel // 2) for kernel in curr_kernel]
            
            curr_shape = self.x_shapes[i]
            
            next_length = ((curr_shape[0] + 2 * curr_pad[0] - curr_kernel[0]) // curr_stride[0]) + 1
            next_height = ((curr_shape[1] + 2 * curr_pad[1] - curr_kernel[1]) // curr_stride[1]) + 1
            next_width = ((curr_shape[2] + 2 * curr_pad[2] - curr_kernel[2]) // curr_stride[2]) + 1
            
            self.x_shapes[i+1] = [next_length, next_height, next_width]
            
        self.blocks = nn.ModuleList()

        if cfg.MODEL.ACT_CHECKPOINT:
            validate_checkpoint_wrapper_import(checkpoint_wrapper)

        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])

            if cfg.MVIT.DIM_MUL_IN_ATT:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
                
            if len(stride_q[i]) > 0:
                self.input_size_1 = [
                    size // stride for size, stride in zip(input_size, stride_q[i])
                ]
            
            attention_block = ConvMVITBlock(
                input_dim=embed_dim,
                output_dim=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
                drop_attn_rate=cfg.MVIT.DROP_ATTN_RATE,
                mlp_ratio = mlp_ratio,
                act_layer = nn.GELU,
                dim_mul_in_attn=cfg.MVIT.DIM_MUL_IN_ATT,
                part_length = 16 if self.input_size_1[0] > 16 else self.input_size_1[0] // 2,
                twh = self.input_size_1
            )
            #if cfg.MODEL.ACT_CHECKPOINT:
                #attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)

            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride for size, stride in zip(input_size, stride_q[i])
                ]
            
            #print("Input_size", input_size)
            #print("Check", self.x_shapes[i+1])
            embed_dim = dim_out

        self.box_blocks = nn.ModuleList()
        for i in range(box_depth):
            self.box_blocks.append(
                MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    input_size=None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=cfg.MVIT.BOX_DROPOUT_RATE,
                    drop_path=box_dpr[i],
                    norm_layer=norm_layer,
                    has_cls_embed=False,
                    pool_first=True,
                    rel_pos_spatial=False,
                    rel_pos_temporal=False,
                    use_online_memory=self.use_online_memory,
                    attn_max_len=cfg.MEMVIT.ATTN_MAX_LEN,
                    keep_max_len=(cfg.MEMVIT.ATTN_MAX_LEN - 1)
                    * int(cfg.MEMVIT.SAMPLER[-1])
                    + 1
                    if "gap" in cfg.MEMVIT.SAMPLER
                    else cfg.MEMVIT.ATTN_MAX_LEN,
                    is_box_attn=True,
                    drop_attn_rate=cfg.MVIT.BOX_DROP_ATTN_RATE,
                    drop_qkv_rate=cfg.MVIT.BOX_DROP_QKV_RATE,
                    cfg=cfg,
                    conv_q=min(
                        self.conv_q, 1
                    ),  # conv_q won't work for box, so use identity_q
                )
            )

        embed_dim = dim_out
        self.norm = norm_layer(embed_dim)

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[embed_dim],
                num_classes=num_classes,
                pool_size=[[temporal_size // self.patch_stride[0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.TransformerBasicHead(
                embed_dim,
                num_classes,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                frame_level=cfg.MVIT.FRAME_LEVEL,
            )
        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names.extend(
                        [
                            "pos_embed_spatial",
                            "pos_embed_temporal",
                            "pos_embed_class",
                        ]
                    )
                else:
                    names.append(["pos_embed"])
            if self.rel_pos_spatial:
                names.extend(["rel_pos_h", "rel_pos_w"])
            if self.rel_pos_temporal:
                names.extend(["rel_pos_t"])
            if self.cls_embed_on:
                names.append("cls_token")

        return names

    def forward(self, x, video_names=None, bboxes=None):
        x = x[0]
        H = x.shape[3] // self.patch_stride[1]

        x = self.patch_embed(x)

        T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        W = x.shape[1] // H // T
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x = x + pos_embed
            else:
                x = x + self.pos_embed

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        #logger.info(x.shape)
        for blk in self.blocks:
            x = blk(x)
        
        #logger.info(x.shape)
        thw = self.input_size_1

        x = self.norm(x)

        if self.enable_detection:
            if self.cls_embed_on:
                x = x[:, 1:]

            B, _, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, thw[0], thw[1], thw[2])
            if self.box_depth > 0:
                x = self.head([x], bboxes, do_cls=False)

                x = pad_features(B, x, bboxes)
                for blk in self.box_blocks:
                    x, thw = blk(x, thw, [], video_names)

                x = unpad_features(x, bboxes)
                x = self.head(x, _, do_pool=False)
            else:
                x = self.head([x], bboxes)
        else:
            if self.cfg.MVIT.FRAME_LEVEL:
                # Will pool spatially and do frame-level predictions in head.
                x = x[:, (1 if self.cls_embed_on else 0) :].reshape(
                    [x.shape[0]] + thw + [x.shape[-1]]
                )
            else:
                if self.cls_embed_on:
                    x = x[:, 0]
                else:
                    x = x.mean(1)

            x = self.head(x)
        return x
    
def pad_features(B, x, bboxes, num_pad=100):
    """
    Different videos can have different numbers of boxes.
    We thus pad zero features for videos with fewer boxes.
    """
    z = torch.zeros((B, num_pad, x.shape[1]), device=x.device)
    for ex_idx in sorted(set(bboxes[:, 0].cpu().numpy())):
        ex_idx = int(ex_idx)
        cur_boxes = x[bboxes[:, 0] == ex_idx, :, 0, 0]
        z[ex_idx, : cur_boxes.shape[0]] = cur_boxes
    return z


def unpad_features(x, bboxes):
    """
    Undo the pad_features fuction. Please see pad_features for
    more information.
    """
    out_boxes = []
    for ex_idx in range(x.shape[0]):
        ex_idx = int(ex_idx)
        num_boxes = (bboxes[:, 0] == ex_idx).sum()
        out_boxes.append(x[ex_idx, :num_boxes])
    return torch.cat(out_boxes, dim=0)[:, :, None, None]
