import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['ResNet']


def conv_with_padding(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """1D convolution with padding to keep size constant when stride=1"""
    effective_kernel = (kernel_size - 1)*dilation + 1
    padding = (effective_kernel - 1)//2 + ((effective_kernel-1) % 2 > 0)
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)


def resize_conv_with_padding(in_planes: int, out_planes: int, kernel_size: int = 3, scale: int = 1, groups: int = 1,
                             dilation: int = 1) -> nn.Conv1d:
    """convolution with stride 1 and padding to keep size constant"""

    if scale == 1:
        out = conv_with_padding(in_planes, out_planes, kernel_size, groups=groups, dilation=dilation)
    else:
        out = nn.Sequential(
            Interpolate(scale_factor=scale),
            conv_with_padding(in_planes, out_planes, kernel_size, groups=groups, dilation=dilation)
        )

    return out


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def resize_conv3x3(in_planes: int, out_planes: int, scale: int = 1, groups: int = 1, dilation: int = 1) -> nn.Module:
    """up-sampled 3x3 convolution with padding"""
    if scale == 1:
        out = conv3x3(in_planes, out_planes, groups=groups, dilation=dilation)
    else:
        out = nn.Sequential(
            Interpolate(scale_factor=scale),
            conv3x3(in_planes, out_planes, groups=groups, dilation=dilation)
        )
    return out


def resize_conv1x1(in_planes: int, out_planes: int, scale: int = 1) -> nn.Module:
    """up-sampled 1x1 convolution with padding"""
    if scale == 1:
        out = conv1x1(in_planes, out_planes)
    else:
        out = nn.Sequential(
            Interpolate(scale_factor=scale),
            conv1x1(in_planes, out_planes)
        )
    return out


class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate"""

    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        kernel_size: int = 3
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        #if dilation > 1:
        #    raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv_with_padding(inplanes, planes, kernel_size, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_with_padding(planes, planes, kernel_size, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        kernel_size: int = 3
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if kernel_size != 3:
            raise NotImplementedError("Kernel size !=3 not implemented for Bottleneck")
        self.kernel_size = kernel_size
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        scale: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        kernel_size: int = 3
    ) -> None:
        super(DecoderBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('DecoderBlock only supports groups=1 and base_width=64')
        #if dilation > 1:
        #    raise NotImplementedError("Dilation > 1 not supported in DecoderBlock")

        self.conv1 = resize_conv_with_padding(inplanes, inplanes, kernel_size, dilation=dilation)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resize_conv_with_padding(inplanes, planes, kernel_size, scale, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class DecoderBottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        scale: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        kernel_size: int = 3
    ) -> None:
        super(DecoderBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.)) * groups

        if kernel_size != 3:
            raise NotImplementedError("Kernel size !=3 not implemented for Bottleneck")
        self.kernel_size = kernel_size

        self.conv1 = resize_conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = resize_conv3x3(width, width, scale, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        do_fc: bool = False,
        in_channels: int = 12,
        first_kernel: int = 7,
        inner_kernel: int = 3
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.do_fc = do_fc

        self.inplanes = 64
        self.dilation = 1
        self.first_kernel = first_kernel
        self.inner_kernel = inner_kernel
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        first_padding = (first_kernel - 1)//2 + ((first_kernel-1) % 2 > 0)
        self.conv1 = nn.Conv1d(in_channels, self.inplanes, kernel_size=first_kernel, stride=2, padding=first_padding,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        if self.do_fc:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, self.inner_kernel))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=self.inner_kernel))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # 256, 12, 384
        x = self.conv1(x)  # 256, 64, 192
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 256, 64, 96

        x = self.layer1(x)  # 256, 256, 96
        x = self.layer2(x)  # 256, 512, 48
        x = self.layer3(x)  # 256, 1024, 24
        x = self.layer4(x)  # 256, 2048, 12

        x = self.avgpool(x)  # 256, 2048, 1
        x = torch.flatten(x, 1)  # 256, 2048

        if self.do_fc:
            x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ResNetDecoder(nn.Module):

    def __init__(
        self,
        block: Type[Union[DecoderBlock, DecoderBottleneck]],
        layers: List[int],
        latent_dim: int = 16,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        out_channels: int = 12,
        conv1_scale: int = 1,
        conv2_scale: int = 4,
        initial_size: int = 16,
    ) -> None:
        super(ResNetDecoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.expansion = block.expansion
        self.inplanes = 512 * block.expansion
        self.dilation = 1
        self.conv1_scale = conv1_scale
        self.conv2_scale = conv2_scale
        self.initial_size = initial_size

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.linear = nn.Linear(latent_dim, self.inplanes * self.initial_size)
        self.conv1 = resize_conv1x1(self.inplanes, self.inplanes, scale=self.conv1_scale)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 256, layers[2], scale=2)
        self.layer4 = self._make_layer(block, 512, layers[3], scale=2)

        self.conv2 = resize_conv3x3(512*self.expansion, out_channels, self.conv2_scale)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DecoderBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, DecoderBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[DecoderBlock, DecoderBottleneck]], planes: int, blocks: int,
                    scale: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation

        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # 256, 32  # B, L
        x = self.linear(x)  # 256, 32768  # B, 24576  # B, self.inplanes * 12
        x = x.view(x.size(0), 512 * self.expansion, self.initial_size)  # 256, 2048, 16  # B, 2048, 12  # B, self.inplanes, 12
        x = self.conv1(x)  # 256, 2048, 128  # B, 2048, 12  #

        x = self.layer1(x)  # 256, 256, 128  # B, 256, 12  #
        x = self.layer2(x)  # 256, 512, 256  # B, 512, 24  #
        x = self.layer3(x)  # 256, 1024, 512  # B, 1024, 48  #
        x = self.layer4(x)  # 256, 2048, 1024  # B, 2048, 96  #

        x = self.conv2(x)  # 256, 12, 8192 or 256, 12, 1024  # B, 12, 768  #

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


