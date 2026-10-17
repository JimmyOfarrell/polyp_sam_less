import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hardnet import HarDNet
from models.base import ModelEmb2DNet
from models.model_single import SmallDecoder

class ModelEmb(ModelEmb2DNet):
    def __init__(self, args, out_ch=256) -> None:
        super().__init__(args, out=out_ch)
        self.backbone = HarDNet(depth_wise=bool(args['depth_wise']), arch=int(args['order']), args=args, pretrained=args['pretrained'])
        for param in self.backbone.parameters(): param.requires_grad = True
        d = self.backbone.full_features
        self.decoder = SmallDecoder(d, out=out_ch, use_tanh=True)
        
    def forward(self, img):
        img = self.preprocess(img)
        z = self.backbone(img)
        dense_embeddings = self.decoder(z)
        dense_embeddings = F.interpolate(dense_embeddings, (64, 64), mode='bilinear', align_corners=False)

        return dense_embeddings