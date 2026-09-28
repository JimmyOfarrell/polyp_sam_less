import numpy as np
from scipy.ndimage import label

def compute_lesionwise_metrics(pred_mask, gt_mask):
    """
    Compute lesion-wise metrics: detection sensitivity, precision, F1,
    false positives per volume, lesion count difference, and lesion-wise Dice.
    pred_mask, gt_mask: binary numpy arrays (H, W, D)
    """

    results = {}
    pred_labeled, num_pred = label(pred_mask)
    gt_labeled, num_gt = label(gt_mask)

    if num_gt == 0:
        # No lesions in GT
        V_pred = pred_mask.sum()
        V_true = gt_mask.sum()

        # LFPR: se non ci sono GT, tutti i predetti sono FP
        LFPR = 0.0 if num_pred == 0 else 1.0

        # AVD: se non ci sono lesioni vere, calcolo simbolico
        if V_true == 0:
            AVD = 0.0 if V_pred == 0 else np.inf
        else:
            AVD = abs(V_pred - V_true) / (V_true + 1e-8)
        results.update({
            'num_gt': 0,
            'num_pred': num_pred,
            'lesion_sensitivity (LTPR)': np.nan,
            'lesion_precision': np.nan,
            'lesion_f1': np.nan,
            'fp_per_volume': num_pred,
            'lesion_count_diff': num_pred,
            'lesion_dice_mean': np.nan,
            'LFPR': LFPR,
            'absolute_volume_difference': AVD
        })
        return results

    tp = 0
    matched_pred = set()
    dice_scores = []

    for i in range(1, num_gt + 1):
        gt_lesion = (gt_labeled == i)
        overlap_ids = np.unique(pred_labeled[gt_lesion])
        overlap_ids = overlap_ids[overlap_ids > 0]

        if len(overlap_ids) > 0:
            tp += 1
            matched_pred.update(overlap_ids.tolist())
            # Dice per questa lesione
            pred_union = np.zeros_like(pred_mask, dtype=bool)
            for oid in overlap_ids:
                pred_union |= (pred_labeled == oid)
            dice = 2 * (pred_union & gt_lesion).sum() / ((pred_union.sum() + gt_lesion.sum()) + 1e-8)
            dice_scores.append(dice)

    fp = num_pred - len(matched_pred)
    fn = num_gt - tp

    sens = tp / num_gt if num_gt > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * sens * prec / (sens + prec + 1e-8) if (sens + prec) > 0 else 0

    V_pred = pred_mask.sum()
    V_true = gt_mask.sum()
    AVD = abs(V_pred - V_true) / (V_true + 1e-8)

    results.update({
        'num_gt': num_gt,
        'num_pred': num_pred,
        'lesion_sensitivity (LTPR)': sens,
        'lesion_precision': prec,
        'LFPR': 1 - prec,
        'absolute_volume_difference (AVD)': AVD,
        'lesion_f1': f1,
        'fp_per_volume': fp,
        'lesion_count_diff': num_pred - num_gt,
        'lesion_dice_mean': np.mean(dice_scores) if dice_scores else np.nan
    })
    return results


def stratified_lesion_analysis(pred_mask, gt_mask):
    """
    Compute size-stratified Dice and sensitivity for small, medium, and large lesions.
    pred_mask, gt_mask: binary numpy arrays (H, W, D)
    """

    gt_labeled, num_gt = label(gt_mask)
    pred_labeled, num_pred = label(pred_mask)

    size_bins = {'small': [], 'medium': [], 'large': []}
    count_bins = {'small': 0, 'medium': 0, 'large': 0}
    sens_bins = {'small': 0, 'medium': 0, 'large': 0}

    if num_gt == 0:
        return {
            'dice_small': np.nan, 'dice_medium': np.nan, 'dice_large': np.nan,
            'sensitivity_small': np.nan, 'sensitivity_medium': np.nan, 'sensitivity_large': np.nan
        }

    for i in range(1, num_gt + 1):
        gt_lesion = (gt_labeled == i)
        lesion_size = gt_lesion.sum()

        if lesion_size < 10:
            bin_name = 'small'
        elif lesion_size < 100:
            bin_name = 'medium'
        else:
            bin_name = 'large'
        count_bins[bin_name] += 1

        overlap_ids = np.unique(pred_labeled[gt_lesion])
        overlap_ids = overlap_ids[overlap_ids > 0]

        if len(overlap_ids) > 0:
            sens_bins[bin_name] += 1
            pred_union = np.zeros_like(pred_mask, dtype=bool)
            for oid in overlap_ids:
                pred_union |= (pred_labeled == oid)
            dice = 2 * (pred_union & gt_lesion).sum() / ((pred_union.sum() + gt_lesion.sum()) + 1e-8)
            size_bins[bin_name].append(dice)

    strat_dice = {f'dice_{k}': np.mean(v) if len(v) > 0 else np.nan for k, v in size_bins.items()}
    strat_sens = {f'sensitivity_{k}': sens_bins[k] / count_bins[k] if count_bins[k] > 0 else np.nan for k in size_bins.keys()}

    results = {}
    results.update(strat_dice)
    results.update(strat_sens)
    return results


