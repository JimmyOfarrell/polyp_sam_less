import torch
import torch.nn as nn
from models.model_emb import ModelEmb
from models.segmentor_net import HardNetSegmentor
from packaging import version

def sam_call(batched_input, sam, dense_embeddings, args=None):
    with torch.no_grad():
        input_images = torch.stack([x["image"] for x in batched_input], dim=0)
        image_embeddings = sam.image_encoder(input_images)
        sparse_embeddings_none, _ = sam.prompt_encoder(points=None, boxes=None, masks=None)

    low_res_masks, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False)
    return low_res_masks

def get_model(args):
    if args['model_emb'] != 'HardNet':
        raise ValueError(
            f"Unsupported model_emb='{args['model_emb']}'. "
            f"Only 'HardNet' is currently supported.")
    
    try: model = ModelEmb(args=args)
    except Exception as e: raise RuntimeError(f"Failed to build ModelEmb (model_emb={args['model_emb']}): {e}") from e
    print(f"[get_model] Model created on {args['device']}")

    if args.get('ckpt_path') is not None:
        ckpt_path = args['ckpt_path']

        try: ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except Exception as e: raise RuntimeError(f"Failed to load checkpoint '{ckpt_path}': {e}") from e

        if isinstance(ckpt, dict): state_dict = ckpt
        else: state_dict = ckpt.state_dict()

        strict = not args.get('load_ckpt_single_dict', False)

        try: result = model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as e: raise RuntimeError(f"Failed to load state_dict into model: {e}") from e

        print(f"[get_model] Checkpoint loaded from '{ckpt_path}' "
              f"(strict={strict})")

        if not strict:
            if result.missing_keys: print(f"[get_model]   Missing keys   : {len(result.missing_keys)}")
            if result.unexpected_keys: print(f"[get_model]   Unexpected keys: {len(result.unexpected_keys)}")

    return model

def get_standard_model(args):
    if args['net'] == 'HardNetSegmentor' and args['task'] == 'polyp':
        model = HardNetSegmentor(args, segmentor=args['net_segmentor'])
        print(f"[get_standard_model] Built HardNetSegmentor "
              f"(segmentor={args['net_segmentor']})")
    else:
        raise ValueError(
            f"Model not implemented for net='{args['net']}' and "
            f"task='{args['task']}'. "
            f"Supported: net='HardNetSegmentor' with task='polyp'.")
    
    if args.get('ckpt_path') is not None:
        ckpt_path = args['ckpt_path']
        print(f"[get_standard_model] Loading checkpoint: {ckpt_path}")

        try:
            torch_version = version.parse(torch.__version__)
            if torch_version >= version.parse('1.13.0'): ckpt_model = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            else: ckpt_model = torch.load(ckpt_path, map_location='cpu')
        except Exception as e: raise RuntimeError(f"Failed to load checkpoint '{ckpt_path}': {e}") from e
 
        if not args['use_sam'] and 'segmentor_finetune_backbone' in args.keys(): strict = False
        else: strict = True

        state_dict = ckpt_model if isinstance(ckpt_model, dict) else ckpt_model.state_dict()

        try: result = model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as e: raise RuntimeError(f"Failed to load state_dict into model: {e}") from e
        print(f"[get_standard_model] Checkpoint loaded (strict={strict})")

        if not strict:
            if result.missing_keys:
                print(f"[get_standard_model]   Missing keys   : {len(result.missing_keys)}")
                for key in result.missing_keys[:5]: print(f"   - {key}")
            if result.unexpected_keys:
                print(f"[get_standard_model]   Unexpected keys: {len(result.unexpected_keys)}")
                for key in result.unexpected_keys[:5]: print(f"   - {key}")

    return model