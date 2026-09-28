import torch.utils.data
import torch
import os
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from utils.utils import get_input_dict, norm_batch
from monai.transforms import AsDiscrete, Activations
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
import sys
from utils.saver import Saver
from dataset.MSLesSeg import get_test_mslesseg_dataset
from utils.model_utils import get_model, sam_call, get_standard_model
from utils.utils import str2bool
from utils.metrics_utils import stratified_lesion_analysis, compute_lesionwise_metrics

@torch.no_grad()
def inference_ds_monai(ds, model, sam, transform, epoch, args, saver):
    print('Inference started...')
    pbar = tqdm(ds)
    model.eval()
    iou_list_true = []
    iou_list_false = []
    dice_list_true = []
    dice_list_false = []
    hausdorff_metric_list_true = []
    hausdorff_metric_list_false = []
    hausdorff_metric_list_true95 = []
    hausdorff_metric_list_false95 = []
    surface_distance_list_true = []
    surface_distance_list_false = []
    metrics_list_true = {}
    metrics_list_false = {}
    metrics_list_lesionwise = {}
    metrics_list_stratified = {}
    Idim = int(args['Idim'])
    diceMetric_true = DiceMetric(include_background=True, reduction='mean')
    diceMetric_false = DiceMetric(include_background=False, reduction='mean')
    iouMetric_true = MeanIoU(include_background=True, reduction='mean')
    iouMetric_false = MeanIoU(include_background=False, reduction='mean')
    hausdorff_metric_true = HausdorffDistanceMetric(include_background=True, reduction='mean')
    hausdorff_metric_false = HausdorffDistanceMetric(include_background=False, reduction='mean')
    hausdorff_metric_true95 = HausdorffDistanceMetric(include_background=True, reduction='mean', percentile=95)
    hausdorff_metric_false95 = HausdorffDistanceMetric(include_background=False, reduction='mean', percentile=95)
    surface_distance_metric_true = SurfaceDistanceMetric(include_background=True, reduction='mean', symmetric=True)
    surface_distance_metric_false = SurfaceDistanceMetric(include_background=False, reduction='mean', symmetric=True)
    metric_names = ('sensitivity', 'specificity', 'accuracy', 'precision', 'f1 score', 'false negative rate', 'false positive rate', 'threat score')
    metric_names_mappig = {'sensitivity': "Sens(Rec)", 
                           'specificity': 'Spec-TNR', 
                           'accuracy': 'Acc', 
                           'precision': 'Prec',
                           'f1 score': 'F1',
                           'false negative rate': 'FNR',
                           'false positive rate': 'FPR', 
                           'threat score': 'Jaccard'}
    conf_matr_metric_true = ConfusionMatrixMetric(include_background=True, metric_name = metric_names, compute_sample=True)
    conf_matr_metric_false = ConfusionMatrixMetric(include_background=False, metric_name = metric_names, compute_sample=True)
    
    for m in metric_names:
        metrics_list_true[m] = []
        metrics_list_false[m] = []

    apply_sigm = True if (getattr(model, "is_segmentor", False) or args['net']=='Unet') and args['use_standard_net'] else False
    
    #post-transformation of labels
    discretize_labels = AsDiscrete(threshold=args['theashold_discretize'])
    sigmoid = Activations(sigmoid=True)

    one_hot = AsDiscrete(to_onehot=2, dim=1)
    
    metrics_per_el = {}
    
    for ix,  sample in enumerate(pbar):
        assert len(sample[args['image_key']]) == 1
        imgs = sample[args['image_key']]
        gts = sample[args['mask_key']]
        original_sz_batch = torch.tensor(np.array([sample[args['image_key']][i].meta['spatial_shape'][:2] for i in range(len(sample[args['image_key']]))]))
        img_sz =torch.tensor(imgs.shape[2:]).repeat(len(sample[args['image_key']]), 1)
        orig_imgs = imgs.to(args['device'])
        
        if args['task'] in {'mslesseg'}:
            pix_dim = imgs.meta["pixdim"][1:4]
        else:
            raise ValueError(f'Unknown how to extract spacing for task {args["task"]}')
        spacing = tuple(float(s) for s in pix_dim)
        
        if args['task'] in ['mslesseg']:
            filename = imgs.meta['filename_or_obj'].split(os.path.sep)[-1].split('.')[0]
        else:
            raise ValueError(f'Unknown how to extract filename for task {args["task"]}')
        metrics_per_el[filename] = {'include_backgroud_true':{},
                                    'include_backgroud_false':{},
                                    'lesionwise':{},
                                    'stratified':{}}

        tensor_per_el = {}

        for idx in range(orig_imgs.shape[2]):
            slice_block = slice = orig_imgs[..., idx, :, :].to(args['device'])
            slice_small = F.interpolate(slice_block, size=(Idim, Idim), mode='bilinear', align_corners=False)

            with torch.no_grad():
                if sam is not None:
                    dense_embeddings = model(slice_small)
                    batched_input = get_input_dict(slice, original_sz_batch, img_sz)
                    pred = norm_batch(sam_call(batched_input, sam, dense_embeddings, args))
                    
                    if args['save_embeddings']:
                        if pred.shape[1] == 2:
                            save_pred = pred[:,1,:,:].unsqueeze(1)
                        elif pred.shape[1] == 1:
                            save_pred = pred
                        if apply_sigm:
                            save_pred = discretize_labels(sigmoid(save_pred))
                        else:
                            save_pred = discretize_labels(save_pred)
                        dense_embs_mean = dense_embeddings.mean(dim=1)
                        saver.save_batch(dense_embeddings.detach().cpu(), str(idx)+'embs', epoch, subfolder=filename)
                        saver.save_batch(slice.detach().cpu(), str(idx)+'slice', epoch, subfolder=filename, remove_batch_dim=False)
                        saver.save_batch(dense_embs_mean.detach().cpu(), str(idx)+'embs_mean', epoch, subfolder=filename, remove_batch_dim=False, normalize = True)
                        saver.save_batch(gts[:,:,idx].detach().cpu(), str(idx)+'gt', epoch, subfolder=filename, remove_batch_dim=False)
                        saver.save_batch(save_pred.detach().cpu(), str(idx)+'pred_discrete', epoch, subfolder=filename, remove_batch_dim=False)
                else:
                    pred = model(slice_block)
            
            if idx == 0:
                depth = orig_imgs.shape[2]
                masks = torch.zeros((pred.shape[0], pred.shape[1], depth, pred.shape[2], pred.shape[3]), device=args['device'])
                masks_resized = torch.zeros(torch.Size(pred.shape [0:2] + orig_imgs.shape[2:])).to(args['device'])
                gts_resized = torch.zeros((masks.shape[0], 1, depth, masks.shape[3], masks.shape[4]), device=args['device'])
                imgs_resized = torch.zeros((masks.shape[0], 1, depth, masks.shape[3], masks.shape[4]), device=args['device'])
                
            
            masks[:,:, idx] = pred #pred_res
            masks_resized[:,:, idx] = F.interpolate(pred, size=(slice.shape[-1], slice.shape[-2]), mode='bilinear', align_corners=True)
            gts_resized[:,:, idx] = F.interpolate(gts[:,:,idx], size=pred.shape[2:], mode='bilinear', align_corners=True)
            imgs_resized[:,:, idx] = F.interpolate(slice, size=pred.shape[2:], mode='bilinear', align_corners=True)
        
        onehot_gts=one_hot(discretize_labels(gts_resized))
        if masks.shape[1] == 1:
            onehot_masks = one_hot(discretize_labels(masks))
        elif masks.shape[1] == 2:
            if apply_sigm:
                onehot_masks = discretize_labels(sigmoid(masks))
            else:
                onehot_masks = discretize_labels(masks)

        if args['save_tensors']:
            tensor_per_el['masks'] = discretize_labels(masks_resized.detach())
            tensor_per_el['gts'] = gts.detach()
            tensor_per_el['imgs'] = orig_imgs.detach()
            saver.save_data(tensor_per_el, filename, subfolder='tensors')
        
        if args['save_test_images']:
            start_slice=0
            if imgs_resized.shape[-1] > 108:
                 print(f'file {filename} has more than 108 slices, saving only the last 108')
            saver.log_slices(imgs_resized[:,:, start_slice:].movedim(2, -1), onehot_masks[:, 1, start_slice: ].unsqueeze(1).detach().movedim(2, -1), gts_resized[:,:, start_slice:].detach().movedim(2, -1), epoch, filename, 'test')
        
        diceMetric_true(onehot_masks, onehot_gts)
        dice_true = diceMetric_true.aggregate().item()
        diceMetric_false(onehot_masks, onehot_gts)
        dice_false = diceMetric_false.aggregate().item()
        iouMetric_true(onehot_masks, onehot_gts)
        iouMetric_false(onehot_masks, onehot_gts)
        iou_true = iouMetric_true.aggregate().item()
        iou_false = iouMetric_false.aggregate().item()
        iou_list_true.append(iou_true)
        iou_list_false.append(iou_false)
        dice_list_true.append(dice_true)
        dice_list_false.append(dice_false)
        
        hausdorff_metric_true(onehot_masks, onehot_gts, spacing=spacing)
        hausdorff_metric_false(onehot_masks, onehot_gts, spacing=spacing)
        hausdorff_metric_true95(onehot_masks, onehot_gts, spacing=spacing)
        hausdorff_metric_false95(onehot_masks, onehot_gts, spacing=spacing)
        hausdorff_true= hausdorff_metric_true.aggregate().item()
        hausdorff_false = hausdorff_metric_false.aggregate().item()
        hausdorff_true95 = hausdorff_metric_true95.aggregate().item()
        hausdorff_false95 = hausdorff_metric_false95.aggregate().item()
        hausdorff_metric_list_true.append(hausdorff_true)
        hausdorff_metric_list_false.append(hausdorff_false)
        hausdorff_metric_list_true95.append(hausdorff_true95)
        hausdorff_metric_list_false95.append(hausdorff_false95)
        
        surface_distance_metric_true(onehot_masks, onehot_gts, spacing=spacing)
        surface_distance_metric_false(onehot_masks, onehot_gts, spacing=spacing)
        surface_distance_true = surface_distance_metric_true.aggregate().item()
        surface_distance_false = surface_distance_metric_false.aggregate().item()
        surface_distance_list_true.append(surface_distance_true)
        surface_distance_list_false.append(surface_distance_false)
        
        conf_matr_metric_true(onehot_masks, onehot_gts)
        conf_matr_metric_false(onehot_masks, onehot_gts)
        metrics_true = conf_matr_metric_true.aggregate()
        metrics_false = conf_matr_metric_false.aggregate()
        
        metrics_per_el[filename]['include_backgroud_true']['dice_score'] = dice_true
        metrics_per_el[filename]['include_backgroud_false']['dice_score'] = dice_false
        metrics_per_el[filename]['include_backgroud_true']['iou_score'] = iou_true
        metrics_per_el[filename]['include_backgroud_false']['iou_score'] = iou_false
        metrics_per_el[filename]['include_backgroud_true']['hausdorff_distance'] = hausdorff_true
        metrics_per_el[filename]['include_backgroud_false']['hausdorff_distance'] = hausdorff_false
        metrics_per_el[filename]['include_backgroud_true']['hausdorff_distance_95'] = hausdorff_true95
        metrics_per_el[filename]['include_backgroud_false']['hausdorff_distance_95'] = hausdorff_false95
        metrics_per_el[filename]['include_backgroud_true']['average_surface_distance'] = surface_distance_true
        metrics_per_el[filename]['include_backgroud_false']['average_surface_distance'] = surface_distance_false

        pred_bin = onehot_masks[:, 1, ...].cpu().numpy().squeeze()
        gt_bin = onehot_gts[:, 1, ...].cpu().numpy().squeeze()

        lesion_metrics = compute_lesionwise_metrics(pred_bin, gt_bin)
        for k, v in lesion_metrics.items():
            metrics_per_el[filename]['lesionwise'][k] = v
            if k not in metrics_list_lesionwise.keys():
                metrics_list_lesionwise[k] = []
            metrics_list_lesionwise[k].append(v)

        # --- Stratified lesion analysis ---
        strat_metrics = stratified_lesion_analysis(pred_bin, gt_bin) 
        for k, v in strat_metrics.items():
            metrics_per_el[filename]['stratified'][k] = v
            if k not in metrics_list_stratified.keys():
                metrics_list_stratified[k] = []
            metrics_list_stratified[k].append(v)
    
        for i,m in enumerate(metric_names):       
            metrics_per_el[filename]['include_backgroud_true'][m]= metrics_true[i].item()
            metrics_per_el[filename]['include_backgroud_false'][m]= metrics_false[i].item()
            metrics_list_true[m].append(metrics_true[i].item())
            metrics_list_false[m].append(metrics_false[i].item())
                
        pbar.set_description(
            '(Inference | {task}) Epoch {epoch} :: Dice True {dice_true:.4f} :: Dice False {dice_false:.4f} :: IoU True {iou_true:.4f} :: IoU False {iou_false:.4f}'.format(
                task=args['task'],
                epoch=epoch,
                dice_true=np.mean(dice_list_true),
                dice_false=np.mean(dice_list_false),
                iou_true=np.mean(iou_list_true), 
                iou_false=np.mean(iou_list_false)))
        
        iouMetric_true.reset()
        iouMetric_false.reset()
        diceMetric_true.reset()
        diceMetric_false.reset()
        hausdorff_metric_true.reset()
        hausdorff_metric_false.reset()
        hausdorff_metric_true95.reset()
        hausdorff_metric_false95.reset()
        conf_matr_metric_true.reset()
        conf_matr_metric_false.reset()
        surface_distance_metric_true.reset()
        surface_distance_metric_false.reset()
    
    if args['log_histograms']:
        saver.log_histogram('DiceTrue', dice_list_true, epoch)
        saver.log_histogram('DiceFalse', dice_list_false, epoch)
        saver.log_histogram('IoU True', iou_list_true, epoch)
        saver.log_histogram('IoU False', iou_list_false, epoch)
        saver.log_histogram('Hausdorff True', hausdorff_metric_list_true, epoch)
        saver.log_histogram('Hausdorff False', hausdorff_metric_list_false, epoch)
        saver.log_histogram('Hausdorff True 95', hausdorff_metric_list_true95, epoch)
        saver.log_histogram('Hausdorff False 95', hausdorff_metric_list_false95, epoch)
        saver.log_histogram('Average Surface Distance True', surface_distance_list_true, epoch)
        saver.log_histogram('Average Surface Distance False', surface_distance_list_false, epoch)
        for m in metric_names:
            saver.log_histogram(m+'_include_true', metrics_list_true[m], epoch)
            saver.log_histogram(m+'_include_false', metrics_list_false[m], epoch)
    
    saver.save_json(metrics_per_el, 'metrics_per_el')
    saver.log_loss('DiceTrue', np.mean(dice_list_true), epoch)
    saver.log_loss('DiceFalse', np.mean(dice_list_false), epoch)
    saver.log_loss('IoU True', np.mean(iou_list_true), epoch)
    saver.log_loss('IoU False', np.mean(iou_list_false), epoch)
    saver.log_loss('Hausdorff True', np.mean(hausdorff_metric_list_true), epoch)
    saver.log_loss('Hausdorff False', np.mean(hausdorff_metric_list_false), epoch)
    saver.log_loss('Hausdorff True 95', np.mean(hausdorff_metric_list_true95), epoch)
    saver.log_loss('Hausdorff False 95', np.mean(hausdorff_metric_list_false95), epoch)
    saver.log_loss('Average Surface Distance True', np.mean(surface_distance_list_true), epoch)
    saver.log_loss('Average Surface Distance False', np.mean(surface_distance_list_false), epoch)
    for m in metric_names:
        saver.log_loss(m+'_include_true', np.mean(metrics_list_true[m]), epoch)
        saver.log_loss(m+'_include_false', np.mean(metrics_list_false[m]), epoch)
    
    m_dict_log = {
        'Dice True': np.mean(dice_list_true),
        'Dice False': np.mean(dice_list_false),
        'IoU True': np.mean(iou_list_true),
        'IoU False': np.mean(iou_list_false),
        'HD True': np.mean(hausdorff_metric_list_true), #in millimeters
        'HD False': np.mean(hausdorff_metric_list_false), #in millimeters
        'HD95 True': np.mean(hausdorff_metric_list_true95), #in millimeters
        'HD95 False': np.mean(hausdorff_metric_list_false95), #in millimeters
        'ASD True': np.mean(surface_distance_list_true), #in millimeters
        'ASD False': np.mean(surface_distance_list_false), #in millimeters
    }
    for m in metric_names:
        m_dict_log[metric_names_mappig[m]+'_true'] = np.mean(metrics_list_true[m])
        m_dict_log[metric_names_mappig[m]+'_false'] = np.mean(metrics_list_false[m])

    for m in metrics_per_el[filename]['lesionwise'].keys():
        saver.log_loss(m+'_lesionwise', np.mean(metrics_list_lesionwise[m]), epoch)
        m_dict_log[m+'_lesionwise'] = np.mean(metrics_list_lesionwise[m])
    for m in metrics_per_el[filename]['stratified'].keys():
        saver.log_loss(m+'_stratified', np.mean(metrics_list_stratified[m]), epoch)
        m_dict_log[m+'_stratified'] = np.mean(metrics_list_stratified[m])

    m_dict_log['Exp_name'] = args['exp_name']
    m_dict_log['Step'] = epoch +1
    
    if saver.metrics_table is None:
        columns = list(m_dict_log.keys())
        saver.create_table(columns)
    saver.log_table_row(epoch +1, m_dict_log)
    
    return np.mean(dice_list_true),np.mean(dice_list_false), np.mean(iou_list_true), np.mean(iou_list_false)


def main(args=None, sam_args= None, saver=None):
    if args['device']=='cuda':
        args['device'] = torch.device("cuda:"+str(args['device_id']))
    else:
        args['device'] = torch.device("cpu")

    
    if args['use_sam']:
        sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
        sam.to(device=args['device'])
        img_dim = sam.image_encoder.img_size
        transform = ResizeLongestSide(img_dim)
    else:
        sam = None
        img_dim = args['Idim']
        transform = None
    
    if args['task'] == 'mslesseg':
        testset = get_test_mslesseg_dataset(args, sam_image_dim=img_dim)
        
    ds_test = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=int(args['nW_eval']), drop_last=False)
    
    img_ch = ds_test.dataset[0][args['image_key']].shape[0]
    
    if not args['use_standard_net']:
        model = get_model(args, sam, img_ch)
    else:
        model = get_standard_model(args, img_ch)
        
        
    with torch.no_grad():
        model.eval()
        inference_ds_monai(ds_test, model.eval(), sam, transform, 0, args, saver)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='monu', help='evaluation iteration', required=False)
    parser.add_argument('-depth_wise', '--depth_wise', default=False, type=str2bool, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85,type=int, help='image size', required=False)
    parser.add_argument('-folder', '--folder', default=34, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=512, type=int, help='image size', required=False)
    parser.add_argument('-rotate', '--rotate', default=22, help='image size', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='image size', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='image size', required=False)
    
    parser.add_argument('--wandb_mode', default='online', help='wandb mode', required=False)
    parser.add_argument('--device', default='cuda', help='device', required=False)
    parser.add_argument('--device_id', default=0, help='device id', required=False)
    parser.add_argument('--wandb_project', default='AutoSam_Pancreas', help='wandb project', required=False)
    parser.add_argument('--wandb_entity', default='fproietto', help='wandb entity', required=False)
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--data_dir', type=str, default='', help='Path to the dataset directory', required=False)
    parser.add_argument('--split_path', type=str, default='3DAutoSam_Pancreas/Dataset.json', help='Path to the split file')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=10, help='Save model every n epochs')
    parser.add_argument('--image_key', type=str, default='image',help='Modality to use for training')
    parser.add_argument('--mask_key', type=str, default='mask', help='Mask modality')
    parser.add_argument('--spatial_crop_size', type=int, default=224, help='Spatial crop size')
    parser.add_argument('--pos_rand_crop', type=float, default=0.75, help='Positive ratio for random crop')
    parser.add_argument('--neg_rand_crop', type=float, default=0.25, help='Negative ratio for random crop')
    parser.add_argument('--axcodes', type=str, default='RAS', help='Axial codes')
    parser.add_argument('--num_slices', type=int, default=1, help='Number of slices')
    parser.add_argument('--n_fold', type=int, default=0, help='Specify fold')
    parser.add_argument('--criterion', type=str, default='dice', choices=['dice', 'dice_ce'], help='Criterion')
    parser.add_argument('--include_background', type=str2bool, default=False, help='Include background in loss')
    parser.add_argument('--acc_steps', type=int, default=4, help='Number of accumulation steps')
    parser.add_argument('--cache_rate', type=float, default=.7, help='Cache rate')

    parser.add_argument('--save_test_images', type=str2bool, default=True, help='Save images')
    parser.add_argument('--theashold_discretize', type=float, default=0.01, help='Threshold for discretization')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Checkpoint path to continue training')
    parser.add_argument('--pretrained', type=str2bool, default=True, help='Pretrained model')
    parser.add_argument('--save_embeddings', type=str2bool, default=True, help='Save embeddings')
    parser.add_argument('--model_emb', type=str, default='HardNet', help='Model embedding')  
    parser.add_argument('--save_tensors', type=str2bool, default=False, help='If save the tensors of imgs, prediciton and gts of all the samples during test')
    parser.add_argument('--log_histograms', type=str2bool, default=False, help='Log histograms to wandb')
    
    parser.add_argument('--remove_maxpool', type=str2bool, default=False, help='Remove first maxpool when ResNet is used as embedding model')

    parser.add_argument('--use_sam', type=str2bool, default=True, help='Use SAM')
    parser.add_argument('--use_standard_net', type=str2bool, default=False, help='Use standard net')
    parser.add_argument('--net', type=str, default='HardNetSegmentor', help='Net')
    parser.add_argument('--net_segmentor', type=str, default='PointwiseConv', help='Net segmentor')
    parser.add_argument('--sam_zeroshot', type=str2bool, default=False, help='SAM zero shot')
    parser.add_argument('--upsample_factor_list', type=int, nargs='+', default=[2, 2], help='Upsample factor list for ConvUpsample segmentor')
    
    parser.add_argument('--sam_ckpt', type=str, default='sam_vit_b', choices = ['sam_vit_b', 'medsam_vit_b'], help='SAM checkpoint')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=['test'], help='Wandb tags')

    parser.add_argument('--out_channels', type=int, default=2, help='Output channels')
    parser.add_argument('--decoder_channels', type=int, nargs='+', default=[], help='Decoder channels for UnetLike segmentor/aggregator')
    
    args = vars(parser.parse_args())
    args['path_best'] = os.path.join('results',
                                     'gpu' + str(args['folder']),
                                     'net_best.pth')
    
    args['experiment_name']= args['exp_name']
    args['exp_folder'] = os.path.join('results', args['experiment_name'])
    
    saver = Saver(output_folder='results', experiment_name=args['experiment_name'], wandb_mode=args['wandb_mode'], wandb_project=args['wandb_project'], wandb_entity=args['wandb_entity'], args=args)
    args['path']= saver.path
    saver.log_hparams(args)
    cmd = str(sys.argv)
    saver.log_cmd(cmd)
    args['vis_folder'] = os.path.join(saver.path, 'vis')
    
    sam_ckpt = "cp/" + args['sam_ckpt'] + ".pth"
    
    sam_args = {
        'sam_checkpoint': sam_ckpt,
        'model_type': "vit_b",
        'generator_args': {
            'points_per_side': 8,
            'pred_iou_thresh': 0.95,
            'stability_score_thresh': 0.7,
            'crop_n_layers': 0,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 0,
            'point_grids': None,
            'box_nms_thresh': 0.7,
        },
        'gpu_id': args['device_id'],
    }


    main(args=args, sam_args=sam_args, saver=saver)
