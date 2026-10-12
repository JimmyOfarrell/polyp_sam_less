import torch
import torch.nn as nn
from models.model_emb import ModelEmb
from models.segmentor_net import HardNetSegmentor
from packaging import version

def sam_call(batched_input, sam, dense_embeddings, args=None):
    with torch.no_grad():
        if args is not None:
            if args['task'] in ['polyp']:
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

def get_model(args: dict, sam, img_ch: int, test: bool = False):
    """
    Build the prompt learner (Phase 1 model) for polyp-SAM-LESS.

    Steps
    -----
    1. Build the prompt learner (ModelEmb) on the target device.
    2. If the input is grayscale (``img_ch == 1``), replace the first
       conv layer of the backbone (and SAM's patch embedding) with a
       1-channel version. Pretrained 3-channel weights are averaged
       over the RGB axis to preserve transferability.
    3. If ``args['ckpt_path']`` is provided, load its weights into the
       model. This step is **always executed**, regardless of ``img_ch``,
       because it is required for Phase 2 (aggregator training).

    Parameters
    ----------
    args : dict
        Configuration dictionary. Must contain:
            - 'model_emb'   : str   — prompt learner type ('HardNet')
            - 'device'      : torch.device
            - 'ckpt_path'   : str or None
            - 'load_ckpt_single_dict' : bool, optional
    sam : Sam or None
        SAM model (Phase 1). If None, SAM's patch embedding is not modified.
    img_ch : int
        Number of input channels (1 = grayscale, 3 = RGB).
    test : bool, default False
        Whether the model is being built for testing (currently unused).

    Returns
    -------
    ModelEmb
        The prompt learner, moved to ``args['device']``.
    """
    # ============================================================
    # 1) Build prompt learner (Phase 1)
    # ============================================================
    if args['model_emb'] != 'HardNet':
        raise ValueError(
            f"Unsupported model_emb='{args['model_emb']}'. "
            f"Only 'HardNet' is currently supported.")
    
    print(f"[get_model] Building prompt learner: {args['model_emb']}")
    try: model = ModelEmb(args=args).to(args['device'])
    except Exception as e: raise RuntimeError(f"Failed to build ModelEmb (model_emb={args['model_emb']}): {e}") from e
    print(f"[get_model] Model created on {args['device']}")

    # ============================================================
    # 2) Grayscale (1-channel) adaptations (only when img_ch == 1)
    #    For polyp (RGB, img_ch == 3), this block is skipped.
    # ============================================================
    if img_ch == 1:
        # ------------------------------------------------------------
        # 2a) HarDNet backbone: replace first conv layer
        # ------------------------------------------------------------
        old_layer = model.backbone.base[0].conv
        old_weights = old_layer.weight.clone()
        new_weights = old_weights.mean(dim=1, keepdim=True)   # (out, 1, k, k)

        has_bias = old_layer.bias is not None
        new_layer = nn.Conv2d(1, old_layer.out_channels, old_layer.kernel_size, old_layer.stride, old_layer.padding, bias=has_bias)
        new_layer.weight.data = new_weights.data
        if has_bias: new_layer.bias.data = old_layer.bias.data
        model.backbone.base[0].conv = new_layer
        print("[get_model] ModelEmb: first conv layer changed to 1-channel input")

        # ------------------------------------------------------------
        # 2b) SAM: replace patch embedding projection
        # ------------------------------------------------------------
        if sam is not None:
            old_sam_pl = sam.image_encoder.patch_embed.proj
            old_sam_weights = old_sam_pl.weight.clone()
            new_sam_weights = old_sam_weights.mean(dim=1, keepdim=True)

            has_bias = old_sam_pl.bias is not None
            new_sam_pl = nn.Conv2d(1, old_sam_pl.out_channels, old_sam_pl.kernel_size, old_sam_pl.stride, old_sam_pl.padding, bias=has_bias)
            new_sam_pl.weight.data = new_sam_weights.data
            if has_bias: new_sam_pl.bias.data = old_sam_pl.bias.data
            sam.image_encoder.patch_embed.proj = new_sam_pl
            print("[get_model] SAM: patch embedding changed to 1-channel input")
        # ============================================================
    
    # 3) Load checkpoint if provided — ALWAYS executed.
    #
    #    This block must run for BOTH img_ch == 1 and img_ch == 3.
    #    In Phase 2 (aggregator training), ckpt_path is set to the
    #    Phase 1 checkpoint; if this block is skipped, the model is
    #    randomly initialized and Phase 2 becomes meaningless.
    # ============================================================
    if args.get('ckpt_path') is not None:
        ckpt_path = args['ckpt_path']

        # --- Load checkpoint on CPU ---
        try: ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except Exception as e: raise RuntimeError(f"Failed to load checkpoint '{ckpt_path}': {e}") from e

        # --- Extract state_dict (model object or dict) ---
        if isinstance(ckpt, dict): state_dict = ckpt
        else: state_dict = ckpt.state_dict()

        # --- Determine strictness ---
        # Same logic as the original MS-SAM-LESS code:
        # load_ckpt_single_dict=True  → strict=False
        # load_ckpt_single_dict=False → strict=True
        strict = not args.get('load_ckpt_single_dict', False)

        # --- Load into model ---
        try: result = model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as e: raise RuntimeError(f"Failed to load state_dict into model: {e}") from e

        print(f"[get_model] Checkpoint loaded from '{ckpt_path}' "
              f"(strict={strict})")

        if not strict:
            if result.missing_keys: print(f"[get_model]   Missing keys   : {len(result.missing_keys)}")
            if result.unexpected_keys: print(f"[get_model]   Unexpected keys: {len(result.unexpected_keys)}")

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