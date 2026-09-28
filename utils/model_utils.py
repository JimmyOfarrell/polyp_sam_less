import torch
from models.model_emb import ModelEmb
from models.segmentor_net import HardNetSegmentor
from packaging import version

def sam_call(batched_input, sam, dense_embeddings, args=None):
    with torch.no_grad():
        if args is not None:
            if args['task'] in ['mslesseg']:
                input_images = torch.stack([x["image"] for x in batched_input], dim=0)
        else:
            input_images = torch.stack([sam.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = sam.image_encoder(input_images)
        sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(points=None, boxes=None, masks=None)
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_masks

def get_model(args, sam, img_ch, test = False):
    if args['model_emb']=='HardNet':
        model = ModelEmb(args=args).to(args['device'])
    else:
        raise ValueError('Model embedding not implemented')
    if img_ch == 1:
        #model change layer
        if args['model_emb']=='HardNet':
            old_layer = model.backbone.base[0].conv
        else:
            raise ValueError('Model embedding not implemented')
        old_weights = old_layer.weight.clone()
        new_weights = old_weights.mean(dim=1, keepdim=True)
        new_layer = torch.nn.Conv2d(1, old_layer.out_channels, old_layer.kernel_size, old_layer.stride, old_layer.padding, bias=old_layer.bias)
        new_layer.weight.data = new_weights.data
        new_layer.weight.data = new_weights.data
        if args['model_emb']=='HardNet':
            model.backbone.base[0].conv = new_layer
        else:
            raise ValueError('Model embedding not implemented')
        if sam is not None:
            #sam change layer
            old_sam_pl = sam.image_encoder.patch_embed.proj
            old_sam__pl_weights = old_sam_pl.weight.clone()
            new_sam_pl_weights = old_sam__pl_weights.mean(dim=1, keepdim=True)
            if old_sam_pl.bias is not None:
                bias = True
            new_sam_pl = torch.nn.Conv2d(1, old_sam_pl.out_channels, old_sam_pl.kernel_size, old_sam_pl.stride, old_sam_pl.padding, bias=bias)
            new_sam_pl.weight.data = new_sam_pl_weights.data
            if bias:
                new_sam_pl.bias.data = old_sam_pl.bias.data
            sam.image_encoder.patch_embed.proj = new_sam_pl
        
        if args['ckpt_path'] is not None:
            torch_version = version.parse(torch.__version__)
            if torch_version >= version.parse('1.13.0'):
                ckpt_model = torch.load(args['ckpt_path'], map_location=args['device'], weights_only=False)
            else:
                ckpt_model = torch.load(args['ckpt_path'], map_location=args['device'])
            if 'load_ckpt_single_dict' in args.keys():
                strict = False if args['load_ckpt_single_dict'] else True
            else:
                strict=True
            if not isinstance(ckpt_model, dict):
                layers_not_loaded = model.load_state_dict(ckpt_model.state_dict(), strict=strict)
                print(f'Model loaded with strict {strict}')
            else:
                layers_not_loaded = model.load_state_dict(ckpt_model, strict=strict)
                print(f'Model loaded with strict {strict}')
            
        
    return model


def get_standard_model(args, img_ch=3):
    if args['net']=='HardNetSegmentor' and args['task'] in ['pancreas', 'spleen', 'prostate', 'mslesseg']:
        model = HardNetSegmentor(args, segmentor=args['net_segmentor']).to(args['device'])
    else:
        raise ValueError('Model not implemented')
    
    if img_ch == 1 and not args['net']=='Unet':
        #model change layer
        if args['net']=='HardNetSegmentor':
            old_layer = model.backbone.base[0].conv
        else:
            raise ValueError('Model not implemented')
        old_weights = old_layer.weight.clone()
        new_weights = old_weights.mean(dim=1, keepdim=True)
        new_layer = torch.nn.Conv2d(1, old_layer.out_channels, old_layer.kernel_size, old_layer.stride, old_layer.padding, bias=old_layer.bias)
        new_layer.weight.data = new_weights.data
        if args['net']=='HardNetSegmentor':
            model.backbone.base[0].conv = new_layer
        else:
            raise ValueError('Model not implemented')

    if args['ckpt_path'] is not None:
        torch_version = version.parse(torch.__version__)
        if torch_version >= version.parse('1.13.0'):
            ckpt_model = torch.load(args['ckpt_path'], weights_only=False)
        else:
            ckpt_model = torch.load(args['ckpt_path'])
        #ckpt_model = torch.load(args['ckpt_path'])
        if not args['use_sam'] and 'segmentor_finetune_backbone' in args.keys():
            strict = False #if args['segmentor_finetune_backbone'] else True
        else:
            strict=True
        
        if not isinstance(ckpt_model, dict):
            layers = model.load_state_dict(ckpt_model.state_dict(), strict=strict)
            print(f'Model loaded with strict {strict}')
        else:
            layers =  model.load_state_dict(ckpt_model, strict=strict)
            print(f'Model loaded with strict {strict}')
        if strict==False:
            print("************************************")
            print('Layers not loaded:')
            print(layers)
            print("************************************")
    return model