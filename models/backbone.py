"""ResNet-vd backbone (PResNet), copied and adapted from RT-DETR.

Source: https://github.com/lyuwenyu/RT-DETR (rtdetr_pytorch/src/nn/backbone/presnet.py
and common.py), commit at clone time 2026-07-01. License: Apache License 2.0
(see D:\\Projects\\_external\\RT-DETR\\LICENSE for the full text; this file
retains the original authorship notice "by lyuwenyu" per the license's
attribution requirement).

Changes vs. upstream:
- Dropped the `@register`/`src.core` YAML-config-registry integration (not
  used in this project; config is read directly from configs/base.yaml).
- Inlined `common.py` (ConvNormLayer, FrozenBatchNorm2d, get_activation) into
  this file instead of a separate module.
- Added `DualStreamBackbone`, a project-specific wrapper (not present
  upstream) holding two independent (non-weight-sharing) PResNet instances
  for the RGB and IR streams of FA-PromptDETR.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PResNet", "DualStreamBackbone"]


def get_activation(act: str, inpace: bool = True):
    act = act.lower()
    if act == "silu":
        m = nn.SiLU()
    elif act == "relu":
        m = nn.ReLU()
    elif act == "leaky_relu":
        m = nn.LeakyReLU()
    elif act == "gelu":
        m = nn.GELU()
    elif act is None:
        m = nn.Identity()
    elif isinstance(act, nn.Module):
        m = act
    else:
        raise RuntimeError(f"Unsupported activation: {act}")

    if hasattr(m, "inplace"):
        m.inplace = inpace
    return m


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with fixed batch statistics and affine parameters.

    Copy-paste from torchvision.misc.ops with added eps before rsqrt (via
    facebookresearch/detr), without which non-torchvision ResNet variants
    can produce NaNs.
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}"


ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
}

donwload_url = {
    18: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth",
    34: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth",
    50: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth",
    101: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth",
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="b"):
        super().__init__()
        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            ("conv", ConvNormLayer(ch_in, ch_out, 1, 1)),
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        short = x if self.shortcut else self.short(x)
        out = out + short
        return self.act(out)


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="b"):
        super().__init__()
        if variant == "a":
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out
        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            ("conv", ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1)),
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)
        short = x if self.shortcut else self.short(x)
        out = out + short
        return self.act(out)


class Blocks(nn.Module):
    def __init__(self, block, ch_in, ch_out, count, stage_num, act="relu", variant="b"):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    ch_in,
                    ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=False if i == 0 else True,
                    variant=variant,
                    act=act,
                )
            )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class PResNet(nn.Module):
    def __init__(
        self,
        depth,
        variant="d",
        num_stages=4,
        return_idx=[0, 1, 2, 3],
        act="relu",
        freeze_at=-1,
        freeze_norm=True,
        pretrained=False,
    ):
        super().__init__()

        block_nums = ResNet_cfg[depth]
        ch_in = 64
        if variant in ["c", "d"]:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(
            OrderedDict(
                [(_name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, _name in conv_def]
            )
        )

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(
                Blocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act, variant=variant)
            )
            ch_in = _out_channels[i]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            state = torch.hub.load_state_dict_from_url(donwload_url[depth])
            self.load_state_dict(state)
            print(f"Load PResNet{depth} state_dict")

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


class DualStreamBackbone(nn.Module):
    """Two independent PResNet backbones (no weight sharing) for RGB and IR streams.

    # DESIGN DECISION: return_idx=[0,1,2,3] pulls all four PResNet stages
    # (P2/P3/P4/P5, strides 4/8/16/32, channels 256/512/1024/2048) out of the
    # backbone. The decoder (Giai doan D, Buoc 5) only consumes P2/P3/P4 and
    # drops P5 -- P5 is still computed here (cheap relative to the rest of the
    # backbone) so this class stays a faithful, unmodified-shape copy of
    # upstream PResNet's forward; the P5 discard happens downstream.

    # DESIGN DECISION: ir_img is expected to already be a 3-channel tensor
    # (grayscale replicated to 3 channels by the data pipeline, NOT by this
    # module) so that backbone_ir can reuse the same ImageNet-pretrained
    # PResNet weights/stem as backbone_rgb. See models/backbone.py module
    # docstring and configs/base.yaml for the corresponding data-pipeline note.
    """

    def __init__(self, depth: int = 50, variant: str = "d", pretrained: bool = True):
        super().__init__()
        self.backbone_rgb = PResNet(
            depth=depth, variant=variant, return_idx=[0, 1, 2, 3], pretrained=pretrained
        )
        self.backbone_ir = PResNet(
            depth=depth, variant=variant, return_idx=[0, 1, 2, 3], pretrained=pretrained
        )
        self.out_channels = self.backbone_rgb.out_channels
        self.out_strides = self.backbone_rgb.out_strides

    def forward(self, rgb_img: torch.Tensor, ir_img: torch.Tensor) -> tuple[list, list]:
        """rgb_img: [B,3,H,W]. ir_img: [B,3,H,W] (already 3-channel, see class docstring).

        Returns (feats_rgb, feats_ir), each a list of 4 tensors [P2, P3, P4, P5].
        """
        feats_rgb = self.backbone_rgb(rgb_img)
        feats_ir = self.backbone_ir(ir_img)
        return feats_rgb, feats_ir
