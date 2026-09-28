from models.hardnet import HarDNet
from models.base import ModelEmb2DNet
from models.model_single import SmallDecoder
import torch.nn.functional as F
import torch.nn as nn

class ModelEmb(ModelEmb2DNet):
    def __init__(self, args, out_ch=256):
        super().__init__(args)
        self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), args=args, pretrained=args['pretrained'])
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = SmallDecoder(d, out=out_ch)
        for param in self.backbone.parameters():
            param.requires_grad = True
        

    def forward(self, img, size=None):
        self.backbone.to(img.device)
        z = self.backbone(img)
        dense_embeddings = self.decoder(z)
        dense_embeddings = F.interpolate(dense_embeddings, (64, 64), mode='bilinear', align_corners=True)
        return dense_embeddings