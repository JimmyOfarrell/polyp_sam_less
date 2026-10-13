import torch
import torch.nn as nn
from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate
from packaging import version
from typing import Optional


class PointwiseCNN(nn.Module):
    """
    Lightweight aggregator: 1x1 conv + bilinear upsample to (H, W).

    Produces raw logits (no activation). Used as the "aggregator" head
    in Phase 2 to map dense embeddings → segmentation mask.

    Parameters
    ----------
    in_channels : int
        Input channels (usually 256 from the prompt learner).
    out_channels : int
        Output channels (1 for binary polyp, 2 for MS-style).
    output_size : tuple of int
        Target spatial size (H, W) for the output mask.
        Should match ``Idim`` (typically 1024 for SAM ViT-B).
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 1, output_size: tuple = (1024, 1024)):
        super().__init__()
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.output_size = tuple(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pointwise_conv(x)
        if out.shape[-2:] != self.output_size: out = interpolate(out, size=self.output_size, mode='bilinear', align_corners=False)
        return out  # raw logits

class ConvBlock(nn.Module):
    """2 convolution 3x3 + BatchNorm + ReLU ."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners=align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
    
        
class TransposeConvUpsample(nn.Module):
    """
    Learnable-upsampling aggregator for Phase 2.

    At each step, reduces channels by 2 (via a 1x1 conv) and
    upsamples by ``factor`` using ``ConvTranspose2d`` (learnable).
    A final 1x1 conv maps to ``out_channels``.

    Compared to ``ConvUpsample`` (which uses bilinear upsampling),
    this aggregator has more parameters but can learn the
    upsampling operator.

    .. warning::
        ``ConvTranspose2d`` with ``kernel_size == stride`` may produce
        checkerboard artifacts. Consider using ``ConvUpsample`` if
        this becomes an issue.

    Parameters
    ----------
    in_channels : int, default 256
        Number of input channels (output of SmallDecoder).
    out_channels : int, default 1
        Number of output channels (1 for binary polyp, 2 for MS).
    upsample_factors : list of int, optional
        Multiplier at each upsampling step. Default [2, 2].
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 1, upsample_factors: Optional[list] = None) -> None:
        super().__init__()
        if upsample_factors is None: upsample_factors = [2, 2]

        layers = []
        current_channels = in_channels

        for factor in upsample_factors:
            # 1x1 conv: halve channels
            layers.append(nn.Conv2d(current_channels, current_channels // 2, kernel_size=1))
            # Learnable transpose-conv upsampling
            layers.append(nn.ConvTranspose2d(current_channels // 2, current_channels // 2, kernel_size=factor, stride=factor))
            current_channels = current_channels // 2

        # Final 1x1 conv: map to out_channels    
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        self.upsample_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Dense embeddings, shape (B, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Raw logits, shape (B, out_channels, H*prod(factors),
            W*prod(factors)).
        """
        
        return self.upsample_net(x)

class ConvUpsample(nn.Module):
    """
    Progressive-conv aggregator for Phase 2.

    At each step, reduces the channel dimension by 2 (via a 1x1 conv)
    and doubles the spatial resolution (via bilinear upsampling).
    A final 1x1 conv maps to ``out_channels``.

    Matches the aggregator described in the MS-SAM-LESS paper:
    "two pointwise (1x1) convolution layers, each followed by
    bilinear upsampling, progressively halving the channel dimension
    and doubling the spatial resolution."

    Parameters
    ----------
    in_channels : int, default 256
        Number of input channels (output of SmallDecoder).
    out_channels : int, default 1
        Number of output channels (1 for binary polyp, 2 for MS).
    upsample_factors : list of int, optional
        Multiplier at each upsampling step. Default [2, 2].
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 1,upsample_factors: Optional[list] = None) -> None:
        super().__init__()
        if upsample_factors is None: upsample_factors = [2, 2]

        layers = []
        current_channels = in_channels

        for factor in upsample_factors:
            layers.append(nn.Conv2d(current_channels, current_channels // 2, kernel_size=1))
            layers.append(Upsample(scale_factor=factor, mode='bilinear', align_corners=False))
            current_channels = current_channels // 2

        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        self.upsample_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Dense embeddings, shape (B, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Raw logits, shape (B, out_channels, H*prod(factors),
            W*prod(factors)).
        """

        return self.upsample_net(x)
    
class ConvTwoUpsample(nn.Module):
    def __init__(self, in_channels=256, out_channels=2, upsample_factors=[2, 2], relu=False):
        super(ConvTwoUpsample, self).__init__()
        layers = []
        current_channels = in_channels
        for factor in upsample_factors:
            layers.append(nn.Conv2d(current_channels, current_channels // 2, kernel_size=1))
            if relu:
                layers.append(nn.ReLU())
            layers.append(nn.Conv2d(current_channels // 2, current_channels // 2, kernel_size=1))
            layers.append(Upsample(scale_factor=factor, mode='bilinear', align_corners=True))
            current_channels = current_channels // 2
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        self.upsample_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.upsample_net(x)
    
class DeepConvUpsample(nn.Module):
    def __init__(self, in_channels=256, out_channels=2, upsample_factors=[2, 2]):
        super(DeepConvUpsample, self).__init__()
        layers = []
        current_channels = in_channels
        for factor in upsample_factors:
            layers.append(nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(current_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(current_channels, current_channels // 2, kernel_size=1))
            layers.append(nn.BatchNorm2d(current_channels//2))
            layers.append(nn.ReLU())
            layers.append(Upsample(scale_factor=factor, mode='bilinear', align_corners=True))
            current_channels = current_channels // 2
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        self.upsample_net = nn.Sequential(*layers)

    def forward(self,x):
        return self.upsample_net(x)
    


def get_aggregator(args):
    if args['aggregator'] == 'pointwise_cnn':
        aggregator = PointwiseCNN(in_channels=256, out_channels=args['out_channels'])
    elif args['aggregator'] == 'conv_upsample':
        aggregator = ConvUpsample(in_channels=256, out_channels=args['out_channels'], upsample_factors=args['upsample_factors'])
    elif args['aggregator'] == 'conv_two_upsample':
        aggregator = ConvTwoUpsample(in_channels=256, out_channels=args['out_channels'], upsample_factors=args['upsample_factors'])
    elif args['aggregator'] == 'conv_two_upsample_relu':
        aggregator = ConvTwoUpsample(in_channels=256, out_channels=args['out_channels'], upsample_factors=args['upsample_factors'], relu=True)
    elif args['aggregator'] == 'deep_conv_upsample':
        aggregator = DeepConvUpsample(in_channels=256, out_channels=args['out_channels'], upsample_factors=args['upsample_factors'])
    elif args['aggregator'] == 'transpose_conv_upsample':
        aggregator = TransposeConvUpsample(in_channels=256, out_channels=args['out_channels'], upsample_factors=args['upsample_factors'])
    else:
        raise ValueError('Aggregator not implemented')
        
    if args['ckpt_aggregator_path'] is not None:
        torch_version = version.parse(torch.__version__)
        if torch_version >= version.parse('1.13.0'):
            ckpt_model = torch.load(args['ckpt_aggregator_path'], map_location=args['device'], weights_only=False)
        else:
            ckpt_model = torch.load(args['ckpt_aggregator_path'], map_location=args['device'])
        if 'load_ckpt_single_dict' in args.keys():
            strict = False if args['load_ckpt_single_dict'] else True
            if not strict:
                if not isinstance(ckpt_model, dict):
                    ckpt_model = ckpt_model.state_dict()
                ckpt_model = {key.replace('segmentor.', ''): value
                                for key, value in ckpt_model.items() if 'segmentor' in key}
                
        else:
            strict=True
        if not isinstance(ckpt_model, dict):
            layers_not_loaded = aggregator.load_state_dict(ckpt_model.state_dict(), strict=strict)
            print(f'Aggregator loaded with strict {strict}')
        else:
            layers_not_loaded = aggregator.load_state_dict(ckpt_model, strict=strict)
            print(f'Aggregator loaded with strict {strict}')
    
    return aggregator
