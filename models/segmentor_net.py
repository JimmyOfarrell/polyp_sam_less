
from models.hardnet import HarDNet
from models.base import Segmentor2DNet
from models.model_single import SmallDecoder


class HardNetSegmentor(Segmentor2DNet):
    def __init__(self, args, segmentor='PointwiseConv', out=256):
        super().__init__(args, out_channels=args['out_channels'])
        self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), args=args, pretrained=args['pretrained'])
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = SmallDecoder(d, out=out)
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.init_segmentor_head(segmentor_type=segmentor)
            
    def forward(self, img, size=None):
        z = self.backbone(img)
        dec = self.decoder(z)
        out = self.segmentor(dec)
        return out
