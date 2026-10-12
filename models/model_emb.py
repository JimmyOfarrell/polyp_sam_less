import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hardnet import HarDNet
from models.base import ModelEmb2DNet
from models.model_single import SmallDecoder

class ModelEmb(ModelEmb2DNet):
    """
    Prompt learner for Phase 1 (SAM-guided training).

    Wraps a HarDNet backbone + SmallDecoder. Produces dense embeddings
    of shape (B, 256, 64, 64) that are fed directly to SAM's mask
    decoder as dense prompts.
    """

    def __init__(self, args: dict, out_ch: int = 256) -> None:
        """
        Parameters
        ----------
        args : dict
            Must contain 'depth_wise', 'order', 'pretrained'.
        out_ch : int, default 256
            Output channels. Must be 256 to match SAM's dense embeddings.
        """

        super().__init__(args)
        # --- Backbone: HarDNet ---
        self.backbone = HarDNet(depth_wise=bool(args['depth_wise']), arch=int(args['order']), args=args, pretrained=args['pretrained'])
        # --- Decoder: SmallDecoder produces dense embeddings ---
        d = self.backbone.full_features
        self.decoder = SmallDecoder(d, out=out_ch)

        for param in self.backbone.parameters(): param.requires_grad = True
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        img : torch.Tensor
        Input image, shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
        Dense embeddings, shape (B, out_ch, H/16, W/16).
        """
        self.backbone.to(img.device)
        z = self.backbone(img)
        dense_embeddings = self.decoder(z)

        # Match SAM's image embedding size (H/16, W/16)
        H = img.shape[-2] // 16
        W = img.shape[-1] // 16
        if dense_embeddings.shape[-2:] != (H, W): dense_embeddings = F.interpolate(dense_embeddings, (H, W), mode='bilinear', align_corners=False)
        return dense_embeddings