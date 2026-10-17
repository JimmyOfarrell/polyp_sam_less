from models.hardnet import HarDNet
from models.base import Segmentor2DNet
from models.model_single import SmallDecoder

class HardNetSegmentor(Segmentor2DNet):
    def __init__(self, args, segmentor='PointwiseConv', out=256):
        super().__init__(args, out_channels=args['out_channels'])

        self.backbone = HarDNet(depth_wise=bool(args['depth_wise']), arch=int(args['order']), args=args, pretrained=args['pretrained'])
        for param in self.backbone.parameters(): param.requires_grad = True
        d = self.backbone.full_features
        self.decoder = SmallDecoder(d, out=out, use_tanh=True)
        self.init_segmentor_head(segmentor_type=segmentor)
        print(f"[HardNetSegmentor] backbone={type(self.backbone).__name__}, "
              f"decoder={type(self.decoder).__name__}, "
              f"segmentor={type(self.segmentor).__name__}, "
              f"out_channels={self.out_channels}")
            
    def forward(self, img):
        img = self.preprocess(img)
        z = self.backbone(img)
        dec = self.decoder(z)
        out = self.segmentor(dec)
        return out
