from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from miniSegmentor.aggregator import PointwiseCNN, ConvUpsample, TransposeConvUpsample


class CNNBlock(nn.Module):
    """
    Two-layer 3x3 conv block with BatchNorm and ReLU.

    Used in SmallDecoder to reshape feature maps to the target
    channel count. The output has no activation (raw logits),
    matching downstream usage (e.g. tanh applied externally).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int, default 3
        Convolution kernel size.
    drop : float, default 0
        Dropout rate (0 disables dropout).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, drop: float = 0) -> None:
        super().__init__()
        P = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.conv2_drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.conv1_drop(self.conv1(x_in))
        x = F.relu(self.BN1(x))
        x_out = self.conv2(x)
        return x_out

class UpBlockSkip(nn.Module):
    """
    Upsample + skip-connection block for the decoder.

    Steps:
      1. Upsample ``x_in`` by 2x (bilinear).
      2. Concatenate with ``x_up`` (skip from the encoder).
      3. Two 3x3 convs with BatchNorm and an optional activation.

    Parameters
    ----------
    in_channels : int
        Number of channels of the concatenated input (x_up + upsampled x_in).
    out_channels : int
        Number of output channels.
    kernel_size : int, default 3
        Convolution kernel size.
    func : str or None, default None
        Activation function: 'relu', 'tanh', 'sigmoid', or None.
    drop : float, default 0
        Dropout rate. Use 0 to disable dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, func: Optional[str] = None, drop: float = 0) -> None:
        super().__init__()
        P = kernel_size // 2
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2_drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.BN = nn.BatchNorm2d(out_channels)
        self.func = func

    def forward(self, x_in: torch.Tensor, x_up: torch.Tensor) -> torch.Tensor:
        x = self.Upsample(x_in)
        x_cat = torch.cat((x, x_up), dim=1)
        x1 = self.conv2_drop(self.conv2(self.conv1_drop(self.conv1(x_cat))))
        if self.func == 'tanh': return torch.tanh(self.BN(x1))
        elif self.func == 'relu': return F.leaky_relu(self.BN(x1))
        elif self.func == 'sigmoid': return torch.sigmoid(self.BN(x1))
        else: return x1
        
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0, func=None):
        super(UpBlock, self).__init__()
        d = drop
        P = int((kernel_size - 1) / 2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(d)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2_drop = nn.Dropout2d(d)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.func = func

    def forward(self, x_in):
        x = self.Upsample(x_in)
        x = self.conv1_drop(self.conv1(x))
        x = F.relu(self.BN1(x))
        x = self.conv2_drop(self.conv2(x))
        if self.func == 'None':
            return x
        elif self.func == 'tanh':
            return torch.tanh(self.BN2(x))#F.tanh(self.BN2(x))
        elif self.func == 'relu':
            return F.relu(self.BN2(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0):
        super(DownBlock, self).__init__()
        P = int((kernel_size -1 ) /2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.pool = nn.MaxPool2d((2, 2))
        self.conv1_drop = nn.Dropout2d(drop)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN = nn.BatchNorm2d(out_channels)

    def forward(self, x_in):
        x1 = self.conv2_drop(self.conv2(self.conv1_drop(self.conv1(x_in))))
        x1_pool = F.relu(self.BN(self.pool(x1)))
        return x1, x1_pool


class Encoder(nn.Module):
    def __init__(self, AEdim, drop=0):
        super(Encoder, self).__init__()
        self.full_features = [AEdim, AEdim*2, AEdim*4, AEdim*8, AEdim*8]
        self.down1 = DownBlock(3, AEdim, drop=drop)
        self.down2 = DownBlock(AEdim, AEdim*2, drop=drop)
        self.down3 = DownBlock(AEdim*2, AEdim*4, drop=drop)
        self.down4 = DownBlock(AEdim*4, AEdim*8, drop=drop)

    def forward(self, x_in):
        x1, x1_pool = self.down1(x_in)
        x2, x2_pool = self.down2(x1_pool)
        x3, x3_pool = self.down3(x2_pool)
        x4, x4_pool = self.down4(x3_pool)
        return x1, x2, x3, x4, x4_pool


class MMDecoder(nn.Module):
    def __init__(self, full_features, out_channel, z_size, out_size):
        super(MMDecoder, self).__init__()
        self.bottleneck = BottleneckBlock(full_features[4], z_size)
        self.up0 = UpBlock(z_size, full_features[3],
                           func='relu', drop=0).cuda()
        self.up1 = UpBlock(full_features[3], out_channel,
                           func='None', drop=0).cuda()
        self.out_size = out_size

    def forward(self, z, z_text):
        zz = self.bottleneck(z)
        zz_norm = zz / zz.norm(dim=1).unsqueeze(dim=1)
        attn_map = (zz_norm * z_text.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdims=True)
        zz = zz * attn_map
        zz = self.up0(zz)
        zz = self.up1(zz)
        zz = F.interpolate(zz, size=self.out_size, mode="bilinear", align_corners=True)
        return F.sigmoid(zz)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv1_drop = nn.Dropout2d(drop)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x_in):
        x = self.conv1_drop(self.conv1(x_in))
        x = F.relu(self.BN1(x))
        x = self.conv2_drop(self.conv2(x))
        x = F.relu(self.BN2(x))
        return x
    


class SegmentorNet(nn.Module):
    """Base class for segmentor networks."""
    is_segmentor = True
    is_3d = None

    def __init__(self, args, out_channels):
        super().__init__()
        self.args = args
        self.out_channels = out_channels
        self.segmentor = None

    def init_segmentor_head(self, segmentor_type):
        if segmentor_type == 'PointwiseConv': self.segmentor = PointwiseCNN(in_channels=256, out_channels=self.out_channels)
        elif segmentor_type == 'ConvUpsample': self.segmentor = ConvUpsample(in_channels=256, out_channels=self.out_channels, upsample_factors=self.args['upsample_factor_list'])
        #elif segmentor_type == 'UnetLike': self.segmentor = UnetLikeSegmentor(in_channels=256, decoder_channels=self.args['decoder_channels'], num_classes=self.out_channels)
        elif segmentor_type == 'TransposeConvUpsample': self.segmentor = TransposeConvUpsample(in_channels=256, out_channels=self.out_channels, upsample_factors=self.args['upsample_factor_list'])
        else: raise ValueError('Segmentor not implemented')

class Segmentor2DNet(SegmentorNet):
    is_3d = False

          
    def preprocess(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected 2D input (B,C,H,W), got {x.shape}")
        return x

class ModelEmbNet(nn.Module):
    """
    Base class for model embedding (prompt learner) networks.

    Subclasses must set ``is_3d`` to True or False and implement the
    forward pass. The base class provides common attributes used by
    the training loop (``is_segmentor``, ``is_model_emb``) and stores
    ``args`` and ``out`` for convenience.

    Attributes
    ----------
    is_segmentor : bool
        Always False for embedding networks (used to distinguish from
        segmentation heads in the training loop).
    is_model_emb : bool
        Always True for embedding networks.
    is_3d : bool or None
        Set to True/False by subclasses. ``None`` indicates the base class
        itself should not be instantiated directly.
    args : dict
        Configuration dictionary (access from subclasses).
    out : int
        Output channel count. Must be 256 for SAM compatibility.
    """

    is_segmentor = False
    is_model_emb = True
    is_3d = None   # subclasses override this

    def __init__(self, args: dict, out: int = 256) -> None:
        """
        Parameters
        ----------
        args : dict
            Configuration dictionary.
        out : int, default 256
            Number of output channels. Must match SAM's dense embeddings
            (256 for ViT-B).
        """
        super().__init__()

        if out != 256:
            raise ValueError(
                f"out_ch must be 256 to match SAM's dense embeddings, got {out}"
            )

        self.args = args
        self.out = out

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Default: no preprocessing. Subclasses may override."""
        return x

class ModelEmb2DNet(ModelEmbNet):
    """
    2D variant of ``ModelEmbNet``.

    Used by ``ModelEmb`` for 2D image segmentation tasks such as polyp
    detection. Inputs are expected in ``(B, C, H, W)`` format.

    Attributes
    ----------
    is_3d : bool
        Always False for this class (2D mode).
    """

    is_3d = False

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Validate that the input tensor is 4D (B, C, H, W).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            The same tensor, unchanged.

        Raises
        ------
        ValueError
            If the input is not 4D.
        """
        if x.ndim != 4:
            raise ValueError(
                f"ModelEmb2DNet expects 4D input (B, C, H, W), "
                f"got shape {tuple(x.shape)}"
            )
        return x
