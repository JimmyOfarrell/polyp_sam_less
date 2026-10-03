import torch.nn as nn
from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate
import torch
from packaging import version


class PointwiseCNN(nn.Module):
    def __init__(self, in_channels=256, out_channels=2, output_size=(512, 512)):
        super(PointwiseCNN, self).__init__()
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.output_size = output_size

    def forward(self, x):
        out = self.pointwise_conv(x)
        out = interpolate(out, self.output_size, mode='bilinear', align_corners=True)
        return out

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
    def __init__(self, in_channels=256, out_channels=2, upsample_factors=[2, 2]):
        super(TransposeConvUpsample, self).__init__()
        layers = []
        current_channels = in_channels
        for factor in upsample_factors:
            layers.append(nn.Conv2d(current_channels, current_channels // 2, kernel_size=1))
            layers.append(nn.ConvTranspose2d(current_channels // 2, current_channels // 2, kernel_size=factor, stride=factor))
            current_channels = current_channels // 2
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        self.upsample_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.upsample_net(x)

class ConvUpsample(nn.Module):
    def __init__(self, in_channels=256, out_channels=2, upsample_factors=[2, 2]):
        super(ConvUpsample, self).__init__()
        layers = []
        current_channels = in_channels
        for factor in upsample_factors:
            layers.append(nn.Conv2d(current_channels, current_channels // 2, kernel_size=1))
            layers.append(Upsample(scale_factor=factor, mode='bilinear', align_corners=True))
            current_channels = current_channels // 2
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        self.upsample_net = nn.Sequential(*layers)

    def forward(self, x):
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
