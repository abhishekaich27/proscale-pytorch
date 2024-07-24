# Copyright (c) NEC Laboratories America, Inc.. All Rights Reserved

from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import numpy as np
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
import warnings

from mask2former.modeling.pixel_decoder.msdeformattn import (
    MSDeformAttnTransformerEncoderLayer,
)
import fvcore.nn.weight_init as weight_init
from torch.nn import functional as F
from torch.cuda.amp import autocast

from .ops.modules import MSDeformAttn
from ..transformer_decoder.transformer import _get_indexed_layers
from torch.nn.init import normal_
from icecream import ic
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.layers.wrappers import check_if_dynamo_compiling
from ..transformer_decoder.position_encoding import PositionEmbeddingSine

class TRC(nn.Module):

    block: nn.Module

    def __init__(
        self,
        num_channels: int = 256,
        is_conv: bool = True,
        excite_temp: float = 1.0
    ) -> None:
        """
        Args:
            num_channels (int): Number of input channels.
            is_conv (bool): Whether we're operating with conv layer.
        """
        super().__init__()
        self.is_conv = is_conv
        if is_conv:
            ops = nn.Conv2d(num_channels, 1, kernel_size=1, bias=True)
        else:
            ops = nn.Linear(num_channels, 1, bias=True)

        self.ops = ops 
        self.sigmoid = nn.Sigmoid()
        self.excite_temp = excite_temp

    def forward(self, curr_feat: torch.Tensor, prev_feat: torch.Tensor, curr_shape, prev_shape) -> torch.Tensor:
        """
        Args:
            input_tensor: X, shape = (batch_size, num_channels, H, W).
            output tensor
        """
        B, L, C = curr_feat.shape

        if self.is_conv:
            prev_feat = prev_feat.view(-1, C, prev_shape[0], prev_shape[1])
            curr_feat = curr_feat.view(-1, C, curr_shape[0], curr_shape[1])

            resized_prev_feat = F.interpolate(prev_feat, size=[curr_shape[0], curr_shape[1]], mode="bilinear", align_corners=False)
            
            map_ = self.ops(resized_prev_feat)
            map_ = self.sigmoid(self.excite_temp * map_)
            output_tensor = torch.mul(curr_feat, map_)

            return output_tensor.reshape((B, L, C))
        else:
            prev_feat = prev_feat.view(-1, C, prev_shape[0], prev_shape[1])
            
            resized_prev_feat = F.interpolate(prev_feat, size=[curr_shape[0], curr_shape[1]], mode="bilinear", align_corners=False)
            resized_prev_feat = resized_prev_feat.reshape((B, L, C))
            
            map_ = self.ops(resized_prev_feat)
            map_ = self.sigmoid(self.excite_temp * map_)
            output_tensor = torch.mul(curr_feat, map_)

            return output_tensor
        
    
class LPE(nn.MaxPool2d):
    """
    A wrapper around :class:`torch.nn.MaxPool2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.MaxPool2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation


    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            # Dynamo doesn't support context managers yet
            is_dynamo_compiling = check_if_dynamo_compiling()
            if not is_dynamo_compiling:
                with warnings.catch_warnings(record=True):
                    if x.numel() == 0 and self.training:
                        # https://github.com/pytorch/pytorch/issues/12013
                        assert not isinstance(
                            self.norm, torch.nn.SyncBatchNorm
                        ), "SyncBatchNorm does not support empty inputs!"

        # x = F.conv2d(
        #     x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        # )
        x = F.max_pool2d(x, self.kernel_size, self.stride,
                            self.padding, self.dilation)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

# MSDeformAttn Transformer encoder in deformable detr
class PyramidMSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4, residual = False,
                 pyramid_num_layers = [1, 1, 1], is_excite_conv=False,
                 excite_temp=1.0
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        num_feature_levels_1_1 = 1
        encoder_layer_1_1 = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels_1_1,
                                                            nhead, enc_n_points)
        
        num_feature_levels_1_2 = 2
        encoder_layer_1_2 = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels_1_2,
                                                            nhead, enc_n_points)
        
        num_feature_levels_1_3 = 3
        encoder_layer_1_3 = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels_1_3,
                                                            nhead, enc_n_points)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        
        encoder_layers = [encoder_layer_1_1, encoder_layer_1_2, encoder_layer_1_3]
        encoder_layer_list = [encoder_layers[min(idx, len(encoder_layers)-1)] for idx in range(len(pyramid_num_layers)) for _ in range(pyramid_num_layers[idx])]

        self.encoder = PyramidMSDeformAttnTransformerEncoder(encoder_layer_list,\
                                                            residual, pyramid_num_layers,\
                                                            is_excite_conv=is_excite_conv,\
                                                            d_model=d_model,\
                                                            excite_temp=excite_temp)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        # normal_(self.level_embed_1_1)
        # normal_(self.level_embed_1_2)
        # normal_(self.level_embed_1_3)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            _, _, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index,\
                               valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return memory, spatial_shapes, level_start_index



class PyramidMSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer_list, 
                 residual = False, 
                 num_layers=[1,1,1],
                 is_excite_conv=False, 
                 excite_temp = 1.0,
                 d_model=256):
        super().__init__()
        self.layers = _get_indexed_layers(encoder_layer_list)
        self.num_layers = len(encoder_layer_list)
        self.residual = residual
        self.num_layers = num_layers
        if is_excite_conv is None:
            self.token_excite = nn.Identity()
        else:
            self.token_excite = TRC(num_channels=d_model, is_conv=is_excite_conv, excite_temp=excite_temp)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        # ic(len(src),spatial_shapes)
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        hw_res5 = level_start_index[1].item()
        pos_res5 = pos[:, :hw_res5, :]
        reference_points_res5 = reference_points[:, :hw_res5, :1, :]
        # output_res5 = output[:, :hw_res5, :]

        hw_res54 = level_start_index[-1].item()
        # ic(hw_res54)
        pos_res54 = pos[:, :hw_res54, :]
        reference_points_res54 = reference_points[:, :hw_res54, :2, :]

        num_feature_levels = len(spatial_shapes)
        split_size_or_sections = [None] * num_feature_levels
        for i in range(num_feature_levels):
            if i < num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = output.shape[1] - level_start_index[i]
        y = torch.split(output, split_size_or_sections, dim=1)
        
        output_res5, output_res4, output_res3 = y[0], y[1], y[2]
        if isinstance(self.token_excite, nn.Identity):
            output_res5 = output_res5
        else:
            output_res5 = self.token_excite(output_res5, output_res3, spatial_shapes[0], spatial_shapes[-1])
        
        # ic(self.num_layers)
        pred_res5 = output_res5.clone()
        for idx in range(self.num_layers[0]):
            pred_res5 = self.layers[idx](
                pred_res5,
                pos_res5,
                reference_points_res5,
                spatial_shapes[0].unsqueeze(0),
                level_start_index[0],
                padding_mask[:, :hw_res5])
            # input('layer 0 done')
        
            if self.residual:
                pred_res5 += output_res5
        
        if isinstance(self.token_excite, nn.Identity):
            output_res4 = output_res4
        else:
            output_res4 = self.token_excite(output_res4, output_res3, spatial_shapes[1], spatial_shapes[-1])
        mod_output_res54 = torch.cat([pred_res5, output_res4], dim=1)
            
        # ic(spatial_shapes[:2, :], level_start_index[:2])
        pred_res54 = mod_output_res54.clone()
        for idx in range(self.num_layers[1]):
            idx += self.num_layers[0]
            pred_res54 = self.layers[idx](
                    pred_res54,
                    pos_res54,
                    reference_points_res54,
                    spatial_shapes[:2, :],
                    level_start_index[:2],
                    padding_mask[:, :hw_res54])
            # input('layer 1 done')
            
            if self.residual:
                pred_res54 += torch.cat([output_res5, output_res4], dim=1)

        pred_res543 = torch.cat([pred_res54, output_res3], dim=1)
        for idx in range(self.num_layers[2]):
            idx += self.num_layers[0] + self.num_layers[1] 
            pred_res543 = self.layers[idx](
                pred_res543,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask)
            # input('layer 2 done')
        return pred_res543
    

@SEM_SEG_HEADS_REGISTRY.register()
class PROSCALE(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        common_stride: int,
        residual: bool,
        pyramid_num_layers: List[int],
        is_excite_conv: bool,
        mask_output_ops: str,
        excite_temp: float
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }

        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res3" to "res5"
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        
        # this is the input shape of transformer encoder (could use less features than pixel decoder
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # to decide extra FPN layers

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = PyramidMSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
            residual=residual,
            pyramid_num_layers=pyramid_num_layers,
            is_excite_conv=is_excite_conv,
            excite_temp = excite_temp

        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)
        
        self.maskformer_num_feature_levels = 3  # always use 3 scales
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            if mask_output_ops == 'conv':
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(output_conv)
            
            elif mask_output_ops == 'maxpool':
                output_conv = LPE(
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm=output_norm,
                    activation=F.relu,
                )

            weight_init.c2_xavier_fill(lateral_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        # ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["transformer_dim_feedforward"] = 1024  # use 1024 for deformable transformer encoder
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        
        ret["residual"] = cfg.MODEL.PROSCALE.RESIDUAL
        ret["pyramid_num_layers"] = cfg.MODEL.PROSCALE.PROSCALE_NUM_LAYERS
        ret["is_excite_conv"] = cfg.MODEL.PROSCALE.IS_EXCITE_CONV
        ret["excite_temp"] = cfg.MODEL.PROSCALE.EXCITE_TEMP
        ret["mask_output_ops"] = cfg.MODEL.PROSCALE.MASK_OUTPUT_OPS
        
        return ret

    @autocast(enabled=False)
    def forward_features(self, features):
        srcs = []
        pos = []
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            # [1, 256, 256, 512]
            y = output_conv(y)
            out.append(y)

        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        return self.mask_features(out[-1]), out[0], multi_scale_features
