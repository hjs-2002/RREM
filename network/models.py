import torch
import torch.nn as nn
from torch.nn import init
import timm
from transformers import CLIPModel
import clip



from dataclasses import dataclass
import logging
import math
from typing import Optional, Tuple, Union
from collections import OrderedDict
import re
from sklearn.metrics import pairwise

import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch import Tensor
import open_clip.utils.misc as misc
import argparse
from functools import partial
from open_clip.utils.env import checkpoint_pathmgr as pathmgr
from open_clip.hf_model import HFTextEncoder
# from .hf_model import HFTextEncoder
from open_clip.modified_resnet import ModifiedResNet
# from .modified_resnet import ModifiedResNet
from open_clip.timm_model import TimmModel
from open_clip.transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer, VisionTransformer_Mul
from open_clip.new_utils import to_2tuple

from open_clip.vp import (
    PadPrompter,
    RandomPatchPrompter,
    FixedPatchPrompter
)

from torch.autograd import Variable, grad
try:
    from .f3net import F3Net
    from .resnet_gram import get_GramNet
except:
    from f3net import F3Net
    from resnet_gram import get_GramNet


# fc layer weight init
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


# 当in_channel != 3 时，初始化模型的第一个Conv的weight， 把之前的通道copy input_chaneel/3 次
def init_imagenet_weight(_conv_stem_weight, input_channel=3):
    for i in range(input_channel // 3):
        if i == 0:
            _conv_stem_weight_new = _conv_stem_weight
        else:
            _conv_stem_weight_new = torch.cat([_conv_stem_weight_new, _conv_stem_weight], axis=1)

    return torch.nn.Parameter(_conv_stem_weight_new)


# TODO 加载训练后的权值时需要设置strict=False
class CLIPVisual(nn.Module):
    def __init__(self, model_name, num_classes=2, freeze_extractor=False):
        super(CLIPVisual, self).__init__()
        model = CLIPModel.from_pretrained(model_name)
        self.visual_model = model.vision_model
        if freeze_extractor:
            self.freeze(self.visual_model)
        self.fc = nn.Linear(in_features=model.vision_embed_dim, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.visual_model(x)
        x = self.fc(x[1])

        return x

    # 冻结网络层
    @staticmethod
    def freeze(model):
        for param in model.parameters():
            param.requires_grad = False


class CLIPModelV2(nn.Module):
    # 定义不同 CLIP 模型对应的输出通道数
    CHANNELS = {
        "RN50": 1024,  # ResNet50 模型的输出通道数
        "ViT-B/32": 512,  # Vision Transformer Base 32 模型的输出通道数
        "ViT-L/14": 768  # Vision Transformer Large 14 模型的输出通道数
    }

    def __init__(self, name='clip-RN50', num_classes=2, freeze_extractor=False):
        super(CLIPModelV2, self).__init__()
        name = name.replace('clip-', '').replace('L-', 'L/').replace('B-', 'B/')
        self.model, self.preprocess = clip.load(name, device="cpu")
        # 若 freeze_extractor 为 True，则冻结特征提取器的参数
        if freeze_extractor:
            self.freeze(self.model)
            print(f'Freezing the feature extractors!')

        # 定义全连接层，输入维度为对应 CLIP 模型的输出通道数，输出维度为分类的类别数量
        self.fc = nn.Linear(self.CHANNELS[name], num_classes)

    def forward(self, x, return_feature=False):

        features = self.model.encode_image(x)
        if return_feature:
            return features
        return self.fc(features)

    # 冻结网络层
    @staticmethod
    def freeze(model):
        """
        冻结给定模型的所有参数，使其在训练过程中不更新。

        :param model: 要冻结参数的模型。
        """
        for param in model.parameters():
            param.requires_grad = False


class ContrastiveModels(nn.Module):
    def __init__(self, model_name, num_classes=2, pretrained=True, embedding_size=1024,
                 freeze_extractor=False):
        super(ContrastiveModels, self).__init__()
        self.model_name = model_name
        self.embedding_size = embedding_size

        self.model = get_models(model_name=model_name, pretrained=pretrained, num_classes=embedding_size,
                                freeze_extractor=freeze_extractor)
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x, return_feature=False):
        feature = self.model(x)
        y_pred = self.fc(feature)
        if return_feature:
            return y_pred, feature

        return y_pred

    def extract_feature(self, x):
        feature = self.model(x)

        return feature


def get_efficientnet_ns(model_name='tf_efficientnet_b3_ns', pretrained=True, num_classes=2, start_down=True):
    net = timm.create_model(model_name, pretrained=pretrained)
    if not start_down:
        net.conv_stem.stride = (1, 1)
    n_features = net.classifier.in_features
    net.classifier = nn.Linear(n_features, num_classes)

    return net


def get_swin_transformers(model_name='swin_base_patch4_window7_224', pretrained=True, num_classes=2):
    """
    :param model_name: swin_base_patch4_window12_384   swin_base_patch4_window7_224 swin_base_patch4_window7_224_in22k
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.head.in_features
    net.head = nn.Linear(n_features, num_classes)

    return net


def get_convnext(model_name='convnext_base_in22k', pretrained=True, num_classes=2, in_channel=3):
    """
    :param model_name: convnext_base_384_in22ft1k, convnext_base_in22k
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.head.fc.in_features
    net.head.fc = nn.Linear(n_features, num_classes)

    if in_channel != 3:
        first_conv_weight = net.stem[0].weight
        first_out_channels = net.stem[0].out_channels
        first_conv = nn.Conv2d(in_channel, first_out_channels, kernel_size=4, stride=4)
        first_conv.weight = init_imagenet_weight(first_conv_weight, input_channel=in_channel)
        net.stem[0] = first_conv

    return net


def get_resnet(model_name='resnet200d', pretrained=True, num_classes=2):
    """
    :param model_name: resnet200d, input_size=512, resnet50
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.fc.in_features
    net.fc = nn.Linear(n_features, num_classes)

    return net


def get_clip_visual_model(model_name="openai/clip-vit-base-patch32", num_classes=2, pretrained=True,
                          freeze_extractor=False):
    if 'openai/clip' in model_name:
        model = CLIPVisual(model_name=model_name, num_classes=num_classes)
    else:
        # 'clip-' + 'name', clip-RN50, clip-ViT-L/14
        model = CLIPModelV2(name=model_name, num_classes=num_classes, freeze_extractor=freeze_extractor)

    return model

vitl14={"embed_dim": 768,
    "vision_cfg": {
        "image_size": 224,
        "layers": 24,
        "width": 1024,
        "patch_size": 14
    },
    "text_cfg": {
        "context_length": 77,
        "vocab_size": 49408,
        "width": 768,
        # "width": 1024,#noproj
        "heads": 12,
        # "heads": 16,#noproj
        "layers": 12
    }}

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype
def get_models(model_name='tf_efficientnet_b3_ns', pretrained=True, num_classes=2,
               in_channel=3, freeze_extractor=False, embedding_size=None):
 

    cast_dtype = get_cast_dtype('fp32')

    if embedding_size is not None and isinstance(embedding_size, int) and embedding_size < 0:
        model = ContrastiveModels(model_name, num_classes, pretrained, embedding_size, freeze_extractor)
        print('Using the Contrastive Model: {}'.format(model_name))
    
    elif model_name == 'RREM':
        model = RREM(
    embed_dim=vitl14['embed_dim'],
    vision_cfg=vitl14['vision_cfg'],
    text_cfg=vitl14['text_cfg'],
    quick_gelu=False,
    cast_dtype=cast_dtype)
        print('Using the RREM Model: {}'.format(model_name)) 
    else:
        raise NotImplementedError(model_name)

    return model


if __name__ == '__main__':
    import time
    image_size = 224
    # model = get_models(model_name='clip-ViT-L-14', num_classes=2, pretrained=False,
    #                    embedding_size=512)  # clip-ViT-L-14
    # print(model)
    model = get_models(model_name='InCTRL', num_classes=2, pretrained=False,
                       embedding_size=512)
    # print(model.default_cfg)
    model = model.to(torch.device('cpu'))
    img = torch.randn(1, 3, image_size, image_size)  # your high resolution picture
    start = time.time()
    times = 1
    for _ in range(times):
        out = model(img)
        if isinstance(out, tuple):
            print([o.shape for o in out])
        else:
            print(out.shape)
    print((time.time()-start)/times)

    # from torchsummary import summary
    # input_s = (3, image_size, image_size)
    # print(summary(model, input_s, device='cpu'))
    pass





PROMPT_TYPES = {
    "padding": PadPrompter,
    "random_patch": RandomPatchPrompter,
    "fixed_patch": FixedPatchPrompter
}


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = True

    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth

@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return visual

def _build_vision_tower_Mul(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )

    #进入多模态

    else:
        # 对于多模态，使用多模态的Transformer，width=1024，head_width=64
        vision_heads = vision_cfg.width // vision_cfg.head_width
        #16
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer_Mul(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return visual


class BatchNormPoint(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.feat_size = feat_size
        self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        x = x.view(s1 * s2, self.feat_size)
        x = self.bn(x)
        return x.view(s1, s2, s3)

class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

class TransformerBasicHead(nn.Module):
    """
    Basic Transformer Head. No pool.
    
    这个类实现了一个基础的Transformer头部，包含三层全连接层和批归一化层，
    用于将输入特征映射到类别预测结果。
    
    Args:
        dim_in (int): 输入特征的维度
        num_classes (int): 输出类别数量
    """

    def __init__(
        self,
        dim_in,
        num_classes
    ):
        super(TransformerBasicHead, self).__init__()
        # 定义三层全连接层和对应的批归一化层
        self.projection1 = nn.Linear(dim_in, 128, bias=True)
        self.projection2 = nn.Linear(128, 64, bias=True)
        self.projection3 = nn.Linear(64, num_classes, bias=True)
        self.bn1 = nn.BatchNorm1d(dim_in)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        """
        前向传播函数
        
        输入特征经过三层全连接层处理，每层后面都跟有ReLU激活函数和批归一化，
        最后通过sigmoid函数得到概率输出。
        
        Args:
            x (Tensor): 输入张量，形状为(batch_size, dim_in)
            
        Returns:
            Tensor: 输出张量，形状为(batch_size, num_classes)，值域为[0,1]
        """
        # 第一层：线性变换 -> ReLU激活 -> 批归一化
        x = self.projection1(x)
        x = F.relu(x, inplace=True)
        x = self.bn2(x)
        # 第二层：线性变换 -> ReLU激活 -> 批归一化
        x = self.projection2(x)
        x = F.relu(x, inplace=True)
        x = self.bn3(x)
        # 第三层：线性变换 -> sigmoid激活
        x = self.projection3(x)
        # return torch.sigmoid(x)
        return x

# ... existing code ...
class Adapter(nn.Module):
    """Adapter模块用于特征变换，通过两个全连接层和ReLU激活函数实现。
    
    Args:
        c_in (int): 输入特征维度
        reduction (int, optional): 降维比例因子，默认为4
    """
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """前向传播函数
        
        Args:
            x (Tensor): 输入张量
            
        Returns:
            Tensor: 经过Adapter变换后的输出张量
        """
        x = self.fc(x)
        return x

class MPIMPatchExtractor(nn.Module):
    def __init__(self, num_layers=4, dim=64):
        super().__init__()
        self.num_layers = num_layers
        self.grid_size = 16 # 224 / 14 = 16
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(num_layers, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.ca_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Linear(dim, dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim, bias=False),
            nn.Sigmoid()
        )
        

        self.sa_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.output_conv = nn.Conv2d(dim, 1, kernel_size=1)
        
    def forward(self, Fp_list, Fp_list_n):
 
        B, L, N, D = Fp_list.shape
        H = W = self.grid_size
    
        Fp_norm = F.normalize(Fp_list, dim=-1)
        Fp_n_norm = F.normalize(Fp_list_n, dim=-1)
        sim = (Fp_norm * Fp_n_norm).sum(dim=-1)
        
        diff_map = 0.5 * (1 - sim) 
    
        diff_map_2d = diff_map.view(B, L, H, W)
        
        feat = self.fusion_conv(diff_map_2d)
        b, c, _, _ = feat.size()
        y = self.ca_pool(feat).view(b, c)
        y = self.ca_fc(y).view(b, c, 1, 1)
        feat = feat * y.expand_as(feat)
        
      
        avg_out = torch.mean(feat, dim=1, keepdim=True)
        max_out, _ = torch.max(feat, dim=1, keepdim=True)
        scale_input = torch.cat([avg_out, max_out], dim=1) 
        spatial_att = self.sa_conv(scale_input) 
        
        feat = feat * spatial_att 
        
    
        out_map = self.output_conv(feat) 
        out_map = out_map.view(B, N)     
        
       
        base_map = torch.mean(diff_map, dim=1) # [B, N]
        final_map = base_map + 0.1 * out_map 
        
        return final_map


class RREM(nn.Module):
   

    def __init__(
            self,
           
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        
        self.total_steps = 0
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower_Mul(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.adapter = Adapter(1024, 4)
        self.diff_head = TransformerBasicHead(256, 1)
        
        self.diff_head_ref = TransformerBasicHead(1024, 1)
        self.patch_extractor = MPIMPatchExtractor(num_layers=4, dim=64)

        self.global_gate = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.Sigmoid() 
        )
        
       
        self.score_fusion_head = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
           
        )
        
        self.input_cls_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
        self.recon_cls_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

        for p in self.visual.parameters():
            p.requires_grad = False

 

    def encode_image(self, image, out_layers: list = [6, 12, 18,24], normalize: bool = False):
       
        features = self.visual.forward(image, out_layers)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
       
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype) 

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x) 
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, tokenizer, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None, normal_list = None,return_feature=False):
    
        if normal_list == None:
            
            img = image[0].cuda(non_blocking=True)
            normal_image = image[1:]
   
            normal_image = torch.stack(normal_image)
            shot, b, _, _, _ = normal_image.shape
            normal_image = normal_image.reshape(-1, 3,224, 224).cuda(non_blocking=True)
        else:
            
            img = image[0].cuda(non_blocking=True)
            normal_image = normal_list
            normal_image = torch.stack(normal_image)
            normal_image = normal_image.unsqueeze(1)
            b = len(img)
            normal_image = normal_image.repeat(1, b, 1, 1, 1)
            shot, _, _, _, _ = normal_image.shape
            normal_image = normal_image.reshape(-1, 3, 224, 224).cuda(non_blocking=True)
        
        token, Fp_list, Fp = self.encode_image(img, normalize=False)
        token_n, Fp_list_n, Fp_n = self.encode_image(normal_image, normalize=False)
        
        
        Fp_list = torch.stack(Fp_list)
        Fp_list_n = torch.stack(Fp_list_n)

        Fp_list = Fp_list[:, :, 1:, :]
        Fp_list_n = Fp_list_n[:, :, 1:, :]

        
        Fp_list = Fp_list.reshape(b, 4, 256, -1)
        Fp_list_n = Fp_list_n.reshape(b, 4, 256 * shot, -1)
        token_n = token_n.reshape(b, shot, -1)
        
        token_n = torch.mean(token_n, dim=1)
        token_ref = token_n - token
        token_ref = self.adapter.forward(token_ref)

       
        input_score = self.input_cls_head(token).squeeze(1)     
        recon_score = self.recon_cls_head(token_n).squeeze(1) 
       
        patch_ref_map = []

        patch_ref_map = self.patch_extractor(Fp_list, Fp_list_n)  
        fg_score = patch_ref_map.max(dim=1).values
       
        global_context = self.global_gate(token_ref) 
        holistic_map = patch_ref_map * global_context + patch_ref_map
  
        hl_score = self.diff_head(holistic_map) 
        hl_score = hl_score.squeeze(1)          
  
        img_ref_score = self.diff_head_ref(token_ref) 
        img_ref_score = img_ref_score.squeeze(1)      
   
        scores_concat = torch.stack([hl_score, fg_score, img_ref_score], dim=1) 

        final_score = self.score_fusion_head(scores_concat) 
        final_score = final_score.squeeze(1) 


    
  
        if return_feature:

            return final_score, img_ref_score, token_ref, input_score, recon_score
       
        return final_score, img_ref_score
         
class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed

