import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES
from ..utils.yunet_layer import Conv4layerBlock, Conv_head


@BACKBONES.register_module()
class YuNetBackbone(nn.Module):

    def __init__(self,
                 stage_channels,
                 downsample_idx,
                 out_idx,
                 se_stages=None,          # <-- NEW: tuple/list[bool], len == len(stage_channels)
                 se_reduction: int = 8):  # <-- NEW
        super().__init__()
        self.layer_num = len(stage_channels)
        self.downsample_idx = downsample_idx
        self.out_idx = out_idx

        # default: no SE anywhere
        if se_stages is None:
            se_stages = [False] * self.layer_num
        assert len(se_stages) == self.layer_num, \
            f"se_stages must have length {self.layer_num}, got {len(se_stages)}"
        self.se_stages = list(map(bool, se_stages))
        self.se_reduction = se_reduction

        # stage 0 uses Conv_head(in, mid, out)
        self.model0 = Conv_head(*stage_channels[0],
                                use_se=self.se_stages[0],
                                se_reduction=self.se_reduction)

        # stages 1..N-1 use Conv4layerBlock(in, out [, withBNRelu=True])
        for i in range(1, self.layer_num):
            self.add_module(
                f'model{i}',
                Conv4layerBlock(*stage_channels[i],
                                use_se=self.se_stages[i],
                                se_reduction=self.se_reduction)
            )

        self.init_weights()

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = []
        for i in range(self.layer_num):
            x = self.__getattr__(f'model{i}')(x)
            if i in self.out_idx:
                out.append(x)
            if i in self.downsample_idx:
                x = F.max_pool2d(x, 2)
        return out
