import torch
import numpy as np

def dice_score(pred, target):
    """
    Compute Dice score between prediction and target. 
    Inputs allow for (B, C, H, W) or (B, H, W).
    """
    smooth = 1e-5
    
    # Flatten spatial dimensions
    if pred.dim() == 4:
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
    elif pred.dim() == 3:
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
    intersection = (pred * target).sum(-1)
    union = pred.sum(-1) + target.sum(-1)
    
    return (2. * intersection + smooth) / (union + smooth)  

def iou(pred, target):
    """
    Compute IoU between prediction and target.
    """
    smooth = 1e-5
    
    # Flatten spatial dimensions
    if pred.dim() == 4:
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
    elif pred.dim() == 3:
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
    intersection = (pred * target).sum(-1)
    union = pred.sum(-1) + target.sum(-1) - intersection
    
    return (intersection + smooth) / (union + smooth)

def generalized_energy_distance(preds, gts, nlabels=1, **kwargs):
    """
    Compute Generalized Energy Distance (GED).
    preds: (M, B, C, H, W) or (M, B, H, W) - M samples from model
    gts: (N, B, C, H, W) or (N, B, H, W) - N ground truth annotations
    """
    # Force to 5D (M, B, C, H, W)
    if preds.dim() == 4:
        preds = preds.unsqueeze(2)
    if gts.dim() == 4:
        gts = gts.unsqueeze(2)

    M = preds.size(0)
    N = gts.size(0)
    assert M == N, f"Expected same number of predictions and ground truths, but got {M} and {N}"
    
    def dist_fct(m1, m2):
        # Default to label 1 if nlabels is 1 (assumes binarization to foreground)
        val = kwargs.get('label_range', [1] if nlabels == 1 else range(nlabels))
        label_range = list(val)
        
        per_label_iou = []
        for lbl in label_range:
            m1_bin = (m1 == lbl).float()
            m2_bin = (m2 == lbl).float()
            
            sum1 = m1_bin.sum(dim=(-1, -2, -3))
            sum2 = m2_bin.sum(dim=(-1, -2, -3))
            
            both_zero = (sum1 == 0) & (sum2 == 0)
            one_zero = ((sum1 > 0) & (sum2 == 0)) | ((sum1 == 0) & (sum2 > 0))
            both_gt_zero = (sum1 > 0) & (sum2 > 0)
            
            res = torch.zeros_like(sum1)
            res[both_zero] = 1.0
            res[one_zero] = 0.0
            
            intersection = (m1_bin * m2_bin).sum(dim=(-1, -2, -3))
            union = sum1 + sum2 - intersection
            res[both_gt_zero] = intersection[both_gt_zero] / union[both_gt_zero]
            
            per_label_iou.append(res)
            
        per_label_iou = torch.stack(per_label_iou, dim=0)
        return 1.0 - (per_label_iou.sum(dim=0) / len(label_range))

    # Term 1: 2 * E[d(S, Y)] (Cross-term)
    term1 = 0.0
    for i in range(M):
        for j in range(M):
            term1 += dist_fct(preds[i], gts[j])
    term1 *= 2.0 / (M ** 2)

    # Term 2: E[d(S, S')] (Model diversity)
    term2 = 0.0
    for i in range(M):
        for j in range(M):
            term2 += dist_fct(preds[i], preds[j])
    term2 /= (M ** 2)

    # Term 3: E[d(Y, Y')] (GT diversity)
    term3 = 0.0
    for i in range(M):
        for j in range(M):
            term3 += dist_fct(gts[i], gts[j])
    term3 /= (M ** 2)

    ged_sq = term1 - term2 - term3
    return ged_sq

def max_dice(preds, gts):
    """
    Compute Max Dice metric.
    preds: (M, B, H, W) or (M, B, C, H, W)
    gts: (N, B, H, W) or (N, B, C, H, W)
    
    For each ground truth annotation, find the best matching prediction.
    Then average over all ground truths.
    """
    if preds.dim() == 4:
        preds = preds.unsqueeze(2)
    if gts.dim() == 4:
        gts = gts.unsqueeze(2)
        
    M = preds.size(0)
    N = gts.size(0)
    
    # Result: (B,)
    batch_size = preds.size(1)
    total_max_dice = torch.zeros(batch_size, device=preds.device)
    
    # For each GT, find max Dice among all preds
    for i in range(N):
        max_d = torch.zeros(batch_size, device=preds.device)
        for j in range(M):
             d = dice_score(preds[j], gts[i]) # (B, C)
             if d.dim() > 1: d = d.mean(1)    # (B,) assuming 1 class or mean
             max_d = torch.max(max_d, d)
        total_max_dice += max_d
        
    return total_max_dice / N


def combined_sensitivity(preds, gts):
    """
    Compute Combined Sensitivity (Sc).
    Sc = TP / (TP + FN) on union of masks.
    """
    epsilon = 1e-6
    # Combine (union) all predictions and ground truths across M and N dimension
    # preds: (M, B, C, H, W) -> Union over M -> (B, C, H, W)
    # Use max for union of binary masks (or soft max for probabilities)
    preds_union, _ = torch.max(preds, dim=0)
    gts_union, _ = torch.max(gts, dim=0)
    
    # Binarize if not already (assuming 0-1 inputs)
    # preds_union = (preds_union > 0.5).float()
    # gts_union = (gts_union > 0.5).float()
    
    # TP: Intersection of unions
    tp = (preds_union * gts_union).sum(dim=(-1, -2, -3)) # Sum over C, H, W -> (B,)
    
    # FN: GT union - Intersection (False Negatives are parts of GT missed by Pred)
    # P_union + G_union = TP + FP + TP + FN = 2TP + FP + FN
    # G_union = TP + FN
    fn = gts_union.sum(dim=(-1, -2, -3)) - tp
    
    # If union of both is empty, sensitivity is 1
    union_all = (preds_union + gts_union > 0).float().sum(dim=(-1, -2, -3))
    
    score = tp / (tp + fn + epsilon)
    score[union_all == 0] = 1.0
    
    return score

def diversity_agreement(preds, gts):
    """
    Compute Diversity Agreement (Da).
    Da = 1 - (delta_V_max + delta_V_min) / 2
    where delta_V = |Var_GT - Var_Pred|
    """
    # Calculate variances between all pairs for GT and Preds
    # Variance here seems to be defined as distance metric between raters? 
    # "matching maximum and minimum variance between two raters"
    # "calculate the variance between all pairs in ground truth distribution"
    # Usually variance implies statistical variance, but context suggests pairwise distance distribution?
    # Let's assume variance = pairwise IoU or 1-IoU? 
    # Paper say "variance between all pairs". 
    # Let's interpret "variance" as the dissimilarity (1-Dice or 1-IoU) between pairs.
    # Actually, simpler interpretation: The text says "take the minimum and maximum variance".
    # And "variance between all pairs".
    # A common proxy for diversity/variance in segmentation is 1 - Pairwise Dice.
    
    M = preds.size(0)
    N = gts.size(0)
    batch_size = preds.size(1)
    
    # Helper to get all pairwise distances (1 - Dice) for a set of masks
    def get_pairwise_distances(masks):
        num = masks.size(0)
        dists = []
        for i in range(num):
            for j in range(i+1, num):
                # Distance = 1 - Dice
                d = 1.0 - dice_score(masks[i], masks[j])
                if d.dim() > 1: d = d.mean(1) # (B,)
                dists.append(d)
        if not dists:
            return torch.zeros((batch_size,), device=masks.device), torch.zeros((batch_size,), device=masks.device)
        dists = torch.stack(dists, dim=0) # (K, B)
        return torch.min(dists, dim=0)[0], torch.max(dists, dim=0)[0]

    min_var_pred, max_var_pred = get_pairwise_distances(preds)
    min_var_gt, max_var_gt = get_pairwise_distances(gts)
    
    delta_v_min = torch.abs(min_var_gt - min_var_pred)
    delta_v_max = torch.abs(max_var_gt - max_var_pred)
    
    da = 1.0 - (delta_v_max + delta_v_min) / 2.0
    return da

def collective_insight(preds, gts):
    """
    Compute Collective Insight (CI) score.
    CI = 3 * Sc * Dmax * Da / (Sc + Dmax + Da)
    """
    sc = combined_sensitivity(preds, gts)
    dmax = max_dice(preds, gts)
    da = diversity_agreement(preds, gts)
    
    eps = 1e-8
    # Harmonic mean of the three metrics
    ci = 3 * sc * dmax * da / (sc * dmax + dmax * da + da * sc + eps)
    old_ci = 3 * sc * dmax * da / (sc + dmax + da + eps)
    
    return ci, sc, dmax, da, old_ci
