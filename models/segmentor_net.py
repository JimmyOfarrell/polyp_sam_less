import torch

from models.hardnet import HarDNet
from models.base import Segmentor2DNet
from models.model_single import SmallDecoder


class HardNetSegmentor(Segmentor2DNet):
    """
    Complete segmentation model for Phase 2 (no SAM).

    Composed of:
      - HarDNet backbone (multi-scale feature extractor)
      - SmallDecoder (lightweight decoder)
      - Aggregator (PointwiseConv / ConvUpsample / ...)

    The aggregator replaces SAM's mask decoder, allowing
    fully automated inference without foundation models.

    Parameters
    ----------
    args : dict
        Configuration dictionary.
    segmentor : str, default 'PointwiseConv'
        Type of aggregator head.
    out : int, default 256
        Number of channels produced by the SmallDecoder
        (input to the aggregator).
    """

    def __init__(self, args: dict, segmentor: str = 'PointwiseConv', out: int = 256) -> None:
        super().__init__(args, out_channels=args['out_channels'])

        # --- Backbone: HarDNet ---
        self.backbone = HarDNet(depth_wise=bool(args['depth_wise']), arch=int(args['order']), args=args, pretrained=args['pretrained'])

        # --- Decoder: SmallDecoder ---
        d = self.backbone.full_features
        self.decoder = SmallDecoder(d, out=out)

        for param in self.backbone.parameters(): param.requires_grad = True

        # --- Aggregator head (replaces SAM's mask decoder) ---
        self.init_segmentor_head(segmentor_type=segmentor)

        print(f"[HardNetSegmentor] backbone={type(self.backbone).__name__}, "
              f"decoder={type(self.decoder).__name__}, "
              f"segmentor={type(self.segmentor).__name__}, "
              f"out_channels={self.out_channels}")
            
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        img : torch.Tensor
            Input image, shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
            Raw logits, shape (B, out_channels, H_out, W_out).
        """
        z = self.backbone(img)
        dec = self.decoder(z)
        out = self.segmentor(dec)
        return out
