import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """
    Flatten all dimensions except the batch dimension.

    Input : (B, C, H, W) or (B, C, 1, 1)
    Output: (B, C * H * W) or (B, C)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class ConvLayer(nn.Sequential):
    """
    Standard conv block: Conv2d → BatchNorm2d → ReLU6.

    Used both in the HarDNet stem and inside HarDBlocks (as the
    pointwise component of CombConvLayer).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel : int, default 3
        Kernel size (padding is set to ``kernel // 2`` to preserve
        spatial resolution when ``stride == 1``).
    stride : int, default 1
        Convolution stride.
    bias : bool, default False
        Whether to include a bias term in the conv. Should remain False
        when BatchNorm follows (BN provides its own bias).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 3,
        stride: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.add_module(
            'conv',
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel, stride=stride,
                padding=kernel // 2, groups=1, bias=bias,
            ),
        )
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU6(inplace=True))


class DWConvLayer(nn.Sequential):
    """
    Depthwise convolution block: DWConv2d → BatchNorm2d.

    Uses ``groups = in_channels`` so that each input channel is convolved
    with its own 3x3 kernel, drastically reducing parameters and FLOPs
    compared to a standard 3x3 convolution.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int, optional
        Must equal ``in_channels`` (kept for API compatibility).
    stride : int, default 1
        Convolution stride (use 2 for downsampling).
    bias : bool, default False
        Whether to include a bias term in the conv. Should remain False
        when followed by BatchNorm.

    Notes
    -----
    No activation is applied inside this block. When used inside
    ``CombConvLayer``, the preceding ``ConvLayer`` (1x1) already provides
    a ReLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        stride: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if out_channels != in_channels:
            raise ValueError(
                f"DWConvLayer requires in_channels == out_channels "
                f"(got {in_channels} vs {out_channels})."
            )
        self.add_module(
            'dwconv',
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, stride=stride, padding=1,
                groups=in_channels, bias=bias,
            ),
        )
        self.add_module('norm', nn.BatchNorm2d(in_channels))


class CombConvLayer(nn.Sequential):
    """
    Depthwise-separable convolution block:
        ConvLayer (1x1, pointwise) → DWConvLayer (3x3, depthwise)

    Used inside HarDBlocks when ``dwconv=True``. Roughly 8x fewer
    parameters than a standard 3x3 conv, at a small accuracy cost.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (after the 1x1 conv).
    kernel : int, default 1
        Kernel size for the pointwise conv (usually 1).
    stride : int, default 1
        Stride for the depthwise conv (use 2 for downsampling).
    bias : bool, default False
        Whether to include bias terms. Usually False because both
        sub-layers have BatchNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 1,
        stride: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.add_module(
            'layer1',
            ConvLayer(in_channels, out_channels, kernel, bias=bias),
        )
        self.add_module(
            'layer2',
            DWConvLayer(out_channels, out_channels, stride=stride, bias=bias),
        )


class HarDBlock(nn.Module):
    """
    A dense-connection block with HarDNet's link pattern.

    Each internal layer i reads from layers {i-1, i-2, i-4, ...} and
    produces ``outch`` channels. Only a subset of layers (even indices
    and the last layer) contribute to the block's final output — this
    matches the selection rule in ``forward``.
    """

    def get_link(
        self,
        layer: int,
        base_ch: int,
        growth_rate: int,
        grmul: float,
    ) -> Tuple[int, int, List[int]]:
        """
        Compute (out_channels, in_channels, link) for a given layer index.

        Uses the HarDNet link rule: layer L connects to layers L-1, L-2,
        L-4, L-8, ... (i.e., subtract powers of 2). The number of output
        channels is::

            out_channels = growth_rate * grmul ** (number of extra links)

        Parameters
        ----------
        layer : int
            Layer index (0 = input, 1-based within the block).
        base_ch : int
            Number of channels of the block input.
        growth_rate : int
            Channels added by each new layer.
        grmul : float
            Growth-rate multiplier for layers with multiple incoming links.

        Returns
        -------
        (out_channels, in_channels, link) : tuple
            - out_channels : int   — output channels for this layer
            - in_channels  : int   — sum of output channels of linked layers
            - link         : list  — indices of layers this layer reads from
        """
        # Base case: the "input" layer has no predecessors
        if layer == 0:
            return base_ch, 0, []

        out_channels = growth_rate
        link: List[int] = []

        # Find links L-1, L-2, L-4, L-8, ...
        idx, dv = 0, 1
        while dv <= layer:
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if idx > 0:
                    out_channels *= grmul
            idx += 1
            dv *= 2

        # Round to nearest even number
        out_channels = int(int(out_channels + 1) / 2) * 2

        # Sum in_channels from all linked layers (recursive)
        in_channels = 0
        for k in link:
            ch, _, _ = self.get_link(k, base_ch, growth_rate, grmul)
            in_channels += ch

        return out_channels, in_channels, link

    def get_out_ch(self) -> int:
        """Return the number of output channels for this HarDBlock."""
        return self.out_channels

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        grmul: float,
        n_layers: int,
        keepBase: bool = False,
        residual_out: bool = False,
        dwconv: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Input channel count (also used as ``base_ch`` in link computation).
        growth_rate : int
            Channels added by each new layer.
        grmul : float
            Growth-rate multiplier for layers with multiple incoming links.
        n_layers : int
            Number of internal Conv layers.
        keepBase : bool, default False
            If True, the block's input tensor is concatenated to the output.
        residual_out : bool, default False
            Reserved for residual output (currently unused, kept for API
            compatibility with the original HarDNet).
        dwconv : bool, default False
            If True, use depthwise-separable convs (CombConvLayer);
            otherwise use standard convs (ConvLayer).
        """
        super().__init__()

        self.keepBase = keepBase
        self.links: List[List[int]] = []
        self.out_channels = 0
        layers_ = []

        for i in range(n_layers):
            outch, inch, link = self.get_link(
                i + 1, in_channels, growth_rate, grmul
            )
            self.links.append(link)

            if dwconv:
                layers_.append(CombConvLayer(inch, outch))
            else:
                layers_.append(ConvLayer(inch, outch))

            # Only even-indexed layers and the last one contribute to
            # the block's final output (see `forward`).
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch

        self.layers = nn.ModuleList(layers_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the block: apply all internal layers with dense links,
        then concatenate a subset of intermediate features.

        Output selection rule (must match ``out_channels`` in __init__):
          - layers_[0]    : input (only if ``keepBase=True``)
          - layers_[-1]   : output of the last layer
          - layers_[odd]  : outputs of even-indexed layers
        """
        # --- Step 1: run all layers ---
        layers_ = [x]
        for layer, link in enumerate(self.links):
            inputs = [layers_[i] for i in link]
            x_cat = torch.cat(inputs, dim=1) if len(inputs) > 1 else inputs[0]
            layers_.append(self.layers[layer](x_cat))

        # --- Step 2: select and concatenate outputs ---
        t = len(layers_)
        keep = [
            layers_[i] for i in range(t)
            if (i == 0 and self.keepBase) or (i == t - 1) or (i % 2 == 1)
        ]
        return torch.cat(keep, dim=1)


class HarDNet(nn.Module):
    """
    HarDNet backbone (variants 39, 68, 85).

    After construction, the ImageNet classifier head is removed so the
    network acts as a multi-scale feature extractor. ``forward`` returns
    5 feature maps at different spatial scales.
    """

    # URL mapping: (arch, depth_wise) → checkpoint URL
    _PRETRAINED_URLS = {
        (68, False): 'https://ping-chao.com/hardnet/hardnet68-5d684880.pth',
        (85, False): 'https://ping-chao.com/hardnet/hardnet85-a28faa00.pth',
        (68, True):  'https://ping-chao.com/hardnet/hardnet68ds-632474d2.pth',
        (39, True):  'https://ping-chao.com/hardnet/hardnet39ds-0e6c6fa9.pth',
    }

    def __init__(
        self,
        depth_wise: bool = False,
        arch: int = 68,
        pretrained: bool = True,
        weight_path: str = '',
        out: int = 1,
        args: Optional[dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        depth_wise : bool, default False
            Use depthwise-separable convolutions (lighter, less accurate).
        arch : int, default 68
            HarDNet variant: 39, 68, or 85.
        pretrained : bool, default True
            Load ImageNet pretrained weights.
        weight_path : str, default ''
            Prefix for the local weight file (used only when torch.hub
            is unavailable).
        out : int, default 1
            Kept for API compatibility with ``ModelEmb`` (unused here).
        args : dict or None
            Kept for API compatibility with ``ModelEmb`` (unused here).
        """
        super().__init__()

        if arch not in (39, 68, 85):
            raise ValueError(
                f"Unsupported HarDNet arch={arch}. Choose from 39, 68, 85."
            )

        # ============================================================
        # 1) Per-architecture configuration
        # ============================================================
        if arch == 68:
            first_ch = [32, 64]
            second_kernel = 3
            max_pool = True
            grmul = 1.7
            drop_rate = 0.1
            ch_list = [128, 256, 320, 640, 1024]
            gr = [14, 16, 20, 40, 160]
            n_layers = [8, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 1, 0]
            full_features = [64, 128, 320, 640, 1024]
            feature_indices = [1, 4, 9, 12, 15]
            truncate_at = 16

        elif arch == 85:
            first_ch = [48, 96]
            second_kernel = 3
            max_pool = True
            grmul = 1.7
            drop_rate = 0.2
            ch_list = [192, 256, 320, 480, 720, 1280]
            gr = [24, 24, 28, 36, 48, 256]
            n_layers = [8, 16, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 0, 1, 0]
            full_features = [96, 192, 320, 720, 1280]
            feature_indices = [1, 4, 9, 14, 18]
            truncate_at = 19

        else:  # arch == 39
            first_ch = [24, 48]
            second_kernel = 3
            max_pool = True
            grmul = 1.6
            drop_rate = 0.1
            ch_list = [96, 320, 640, 1024]
            gr = [16, 20, 64, 160]
            n_layers = [4, 16, 8, 4]
            downSamp = [1, 1, 1, 0]
            full_features = [48, 96, 320, 640, 1024]
            feature_indices = [1, 4, 7, 10, 13]
            truncate_at = 14

        # Depthwise overrides
        if depth_wise:
            second_kernel = 1
            max_pool = False
            drop_rate = 0.05

        # ---- Log 1: architecture summary (before building) ----
        print(f"[HarDNet] Building arch={arch}, depth_wise={depth_wise}, "
              f"pretrained={pretrained}")

        # ============================================================
        # 2) Build backbone
        # ============================================================
        blks = len(n_layers)
        self.base = nn.ModuleList([])

        # --- Stem: Layer 1 (Conv3x3 stride=2) ---
        self.base.append(
            ConvLayer(in_channels=3, out_channels=first_ch[0],
                      kernel=3, stride=2, bias=False)
        )

        # --- Stem: Layer 2 (Conv with configurable kernel) ---
        self.base.append(
            ConvLayer(first_ch[0], first_ch[1],
                      kernel=second_kernel, stride=1)
        )

        # --- Stem: Layer 3 (downsampling) ---
        if max_pool:
            self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.base.append(DWConvLayer(first_ch[1], first_ch[1], stride=2))

        # --- HarD blocks ---
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append(blk)

            # Extra dropout only for HarDNet-85's last block
            if i == blks - 1 and arch == 85:
                self.base.append(nn.Dropout(0.1))

            # 1x1 conv to reshape channels
            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]

            # Optional downsampling
            if downSamp[i] == 1:
                if max_pool:
                    self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    self.base.append(DWConvLayer(ch, ch, stride=2))

        # --- Temporary ImageNet classifier (needed for pretrained load) ---
        ch = ch_list[blks - 1]
        self.base.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(ch, 1000),
            )
        )

        # ============================================================
        # 3) Load pretrained weights (optional)
        # ============================================================
        if pretrained:
            self._load_pretrained(arch, depth_wise, weight_path)

        # ============================================================
        # 4) Truncate classifier + set architecture constants (ALWAYS)
        # ============================================================
        self.features = full_features[-1]
        self.full_features = full_features
        self.list = feature_indices
        self.base = self.base[:truncate_at]   # remove classifier head

        # ---- Log 2: final structure summary ----
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[HarDNet] features={self.features}, "
              f"full_features={self.full_features}, list={self.list}")
        print(f"[HarDNet] base_length={len(self.base)}, "
              f"params={n_params:,} ({n_params / 1e6:.2f}M)")

    # ------------------------------------------------------------
    # Internal: load pretrained weights
    # ------------------------------------------------------------
    def _load_pretrained(
        self, arch: int, depth_wise: bool, weight_path: str
    ) -> None:
        if hasattr(torch, 'hub'):
            key = (arch, depth_wise)
            if key not in self._PRETRAINED_URLS:
                raise ValueError(
                    f"No pretrained weights available for "
                    f"arch={arch}, depth_wise={depth_wise}."
                )
            checkpoint = self._PRETRAINED_URLS[key]
            try:
                state = torch.hub.load_state_dict_from_url(
                    checkpoint, progress=True, map_location='cpu'
                )
                self.load_state_dict(state)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download/load pretrained weights from "
                    f"{checkpoint}: {e}"
                ) from e
        else:
            postfix = 'ds' if depth_wise else ''
            weight_file = f'{weight_path}hardnet{arch}{postfix}.pth'
            if not os.path.isfile(weight_file):
                raise FileNotFoundError(
                    f"Weight file not found: {weight_file}\n"
                    f"Either set pretrained=False, or place the file at "
                    f"that path."
                )
            state = torch.load(weight_file, map_location='cpu',
                               weights_only=True)
            self.load_state_dict(state)

        postfix = 'DS' if depth_wise else ''
        print(f"ImageNet pretrained weights for HarDNet{arch}{postfix} is loaded")

    # ------------------------------------------------------------
    # Forward: return 5 multi-scale feature maps
    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the HarDNet backbone.

        Returns 5 feature maps at different spatial scales::

            x2  : ~1/4 of input resolution
            x4  : ~1/8
            x8  : ~1/16
            x16 : ~1/32
            x32 : ~1/64

        The indices into ``self.base`` where each feature is extracted
        are stored in ``self.list``.
        """
        x2 = x4 = x8 = x16 = x32 = None

        for inx, layer in enumerate(self.base):
            x = layer(x)
            if inx == self.list[0]:
                x2 = x
            elif inx == self.list[1]:
                x4 = x
            elif inx == self.list[2]:
                x8 = x
            elif inx == self.list[3]:
                x16 = x
            elif inx == self.list[4]:
                x32 = x

        features = (x2, x4, x8, x16, x32)

        # Safety check: ensure all 5 features were captured
        if any(f is None for f in features):
            missing = [i for i, f in enumerate(features) if f is None]
            raise RuntimeError(
                f"HarDNet forward: feature maps {missing} were not captured. "
                f"Check self.list = {self.list} and len(self.base) = "
                f"{len(self.base)}."
            )

        return features