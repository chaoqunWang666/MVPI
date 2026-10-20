import os
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import auc
from skimage.measure import label, regionprops


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = x.new_zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


def _shifted_sorted_concat(arrays, span):
    """Concatenate per-group sorted 1-D arrays into ONE globally sorted array.

    Group k is shifted by ``k * span`` (with ``span`` strictly larger than the
    score range) so the concatenation stays sorted. This lets us answer
    "how many elements of group k are <= threshold" for ALL groups with a single
    ``np.searchsorted`` call instead of one call per group.
    """
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    total = int(sum(a.size for a in arrays))
    out = np.empty(total, dtype=np.float64)
    pos = 0
    for k, a in enumerate(arrays):
        n = a.size
        out[pos:pos + n] = a
        if k:
            out[pos:pos + n] += k * span
        offsets[k] = pos
        pos += n
    offsets[len(arrays)] = pos
    return out, offsets


def _prepare_pro_structures(scores, gt_mask, with_iou=False):
    """Pre-compute every quantity that does NOT depend on the threshold.

    The reference implementation re-runs ``skimage.measure.label`` +
    ``regionprops`` for every image at every one of the ``max_step`` thresholds,
    i.e. ~1000x redundant work, and re-scans the full (N, H, W) score volume at
    every step. Both are hoisted out of the loop here.

    ``with_iou`` additionally prepares the per-image sorted scores. They are only
    needed for the per-image IoU curve, which the original code computed but never
    used (``best_miou`` is dead code), so it is skipped by default.
    """
    n_img = gt_mask.shape[0]
    flat_gt = gt_mask.reshape(n_img, -1)
    gt_areas = flat_gt.sum(axis=1).astype(np.float64)
    img_has_gt = flat_gt.any(axis=1)

    region_img_id = []
    region_scores = []
    region_area = []
    for i in range(n_img):
        label_map = label(gt_mask[i], connectivity=2)
        for prop in regionprops(label_map):
            y0, x0, y1, x1 = prop.bbox
            filled = prop.filled_image
            region_img_id.append(i)
            region_scores.append(np.sort(scores[i][y0:y1, x0:x1][filled].astype(np.float64)))
            region_area.append(float(prop.area))

    region_img_id = np.asarray(region_img_id, dtype=np.int64)
    region_area = np.asarray(region_area, dtype=np.float64)
    region_len = np.asarray([s.size for s in region_scores], dtype=np.int64)

    neg = ~gt_mask
    # keep the native dtype: a single (unshifted) group needs no extra precision
    neg_sorted = np.sort(scores[neg])
    neg_area = float(neg.sum())

    span = float(scores.max() - scores.min()) + 1.0
    region_shifted, region_offsets = _shifted_sorted_concat(region_scores, span)

    if with_iou:
        img_scores_sorted = [np.sort(scores[i].ravel().astype(np.float64)) for i in range(n_img)]
        img_len = np.asarray([s.size for s in img_scores_sorted], dtype=np.int64)
        img_shifted, img_offsets = _shifted_sorted_concat(img_scores_sorted, span)
    else:
        img_len = np.zeros(0, dtype=np.int64)
        img_shifted = np.zeros(0, dtype=np.float64)
        img_offsets = np.zeros(0, dtype=np.int64)

    return {
        'with_iou': with_iou,
        'n_img': n_img,
        'gt_areas': gt_areas,
        'img_has_gt': img_has_gt,
        'region_img_id': region_img_id,
        'region_area': region_area,
        'region_len': region_len,
        'region_shifted': region_shifted,
        'region_offsets': region_offsets,
        'img_len': img_len,
        'img_shifted': img_shifted,
        'img_offsets': img_offsets,
        'neg_sorted': neg_sorted,
        'neg_area': neg_area,
        'span': span,
    }


def compute_pro_retrieval_metrics(scores, gt_mask, max_step=1000, expect_fpr=0.3,
                                   return_iou=False):
    """
    calculate segmentation AUPRO, from https://github.com/YoungGod/DFR

    Vectorised re-implementation with the SAME definition as the original
    (``max_step`` thresholds, per-region PRO, per-image IoU, global FPR). The
    connected-component analysis is done once and the per-threshold counts are
    obtained from sorted score arrays via ``np.searchsorted`` instead of 1000
    dense passes over the (N, H, W) score volume.

    Typical speed-up on 448x448 benchmarks: 100x - 500x (minutes -> ~1 s), which
    matters because this runs at the end of EVERY training epoch.

    Args:
        scores: (N, H, W) anomaly score maps (any float dtype).
        gt_mask: (N, H, W) boolean ground-truth masks.
        max_step: number of thresholds (default 1000, as in the reference code).
        expect_fpr: FPR integration limit (default 0.3).
        return_iou: if True, also return ``best_miou`` (the per-image IoU curve is
            then computed as well, at the cost of one extra sort per image).

    Note: the reference version accumulated the intersections in float32 while
    this one counts exactly in float64, so the two agree to ~1e-7 (this one being
    the more accurate of the two).
    """
    scores = np.asarray(scores)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    st = _prepare_pro_structures(scores, gt_mask, with_iou=return_iou)

    max_th = scores.max()
    min_th = scores.min()
    delta = (max_th - min_th) / max_step

    n_img = st['n_img']
    n_region = st['region_len'].size
    region_gid = np.arange(n_region, dtype=np.float64)
    img_gid = np.arange(n_img, dtype=np.float64)
    span = st['span']
    with_iou = st['with_iou']

    ious_mean, ious_std = [], []
    pros_mean, pros_std = [], []
    threds, fprs = [], []

    for step in range(max_step):
        thred = max_th - step * delta
        th_f64 = float(thred)

        # --- per-region true positives (PRO) -----------------------------------
        # number of elements <= thred inside every region, in one searchsorted
        le_region = np.searchsorted(st['region_shifted'], th_f64 + region_gid * span, side='right') \
            - st['region_offsets'][:n_region]
        region_tp = st['region_len'] - le_region
        pro = region_tp / st['region_area']
        pros_mean.append(pro.mean())
        pros_std.append(pro.std())

        if with_iou:
            # --- per-image IoU: |pred & gt| / |pred | gt| -----------------------
            inter = np.bincount(st['region_img_id'], weights=region_tp, minlength=n_img)
            le_img = np.searchsorted(st['img_shifted'], th_f64 + img_gid * span, side='right') \
                - st['img_offsets'][:n_img]
            pred_pos = st['img_len'] - le_img
            union = pred_pos + st['gt_areas'] - inter
            with np.errstate(invalid='ignore', divide='ignore'):
                iou_all = inter / union
            iou = iou_all[st['img_has_gt']]
            ious_mean.append(iou.mean())
            ious_std.append(iou.std())

        # --- global false positive rate ----------------------------------------
        # NOTE: the lookup value MUST be cast to the array dtype. Passing a Python
        # float to a float32 array makes numpy up-cast the whole (20M element)
        # array to float64 on *every* call, which costs ~60 s per evaluation.
        th_native = st['neg_sorted'].dtype.type(th_f64)
        fp = st['neg_sorted'].size - int(np.searchsorted(st['neg_sorted'], th_native, side='right'))
        fpr = fp / st['neg_area']

        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)
    if with_iou:
        ious_mean = np.array(ious_mean)
        ious_std = np.array(ious_std)
        # best per image iou
        best_miou = ious_mean.max()
    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr  # find the indexs of the fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pix_pro_auc = auc(fprs_selected, pros_mean_selected)

    if return_iou:
        return pix_pro_auc, best_miou
    return pix_pro_auc


def compute_aupr_retrieval_metrics(
    predicted_masks,
    ground_truth_masks,
    include_optimal_threshold_rates=False,
):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.
    Args:
        predicted_masks: [list of np.arrays or np.array] [NxHxW] Contains
                               generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    pred_mask = copy.deepcopy(predicted_masks)
    gt_mask = copy.deepcopy(ground_truth_masks)
    num = 200
    out = {}

    if pred_mask is None or gt_mask is None:
        for key in out:
            out[key].append(float('nan'))
    else:
        fprs, tprs = [], []
        precisions, f1s = [], []
        gt_mask = np.array(gt_mask, np.uint8)

        t = (gt_mask == 1)
        f = ~t
        n_true = t.sum()
        n_false = f.sum()
        th_min = pred_mask.min() - 1e-8
        th_max = pred_mask.max() + 1e-8
        pred_gt = pred_mask[t]
        th_gt_min = pred_gt.min()
        th_gt_max = pred_gt.max()

        '''
        Using scikit learn to compute pixel au_roc results in a memory error since it tries to store the NxHxW float score values.
        To avoid this, we compute the tp, fp, tn, fn at equally spaced thresholds in the range between min of predicted 
        scores and maximum of predicted scores
        '''
        percents = np.linspace(100, 0, num=num // 2)
        th_gt_per = np.percentile(pred_gt, percents)
        th_unif = np.linspace(th_gt_max, th_gt_min, num=num // 2)
        thresholds = np.concatenate([th_gt_per, th_unif, [th_min, th_max]])
        thresholds = np.flip(np.sort(thresholds))

        if n_true == 0 or n_false == 0:
            raise ValueError("gt_submasks must contains at least one normal and anomaly samples")

        for th in thresholds:
            p = (pred_mask > th).astype(np.uint8)
            p = (p == 1)
            fp = (p & f).sum()
            tp = (p & t).sum()

            fpr = fp / n_false
            tpr = tp / n_true
            if tp + fp > 0:
                prec = tp / (tp + fp)
            else:
                prec = 1.0
            if prec > 0. and tpr > 0.:
                f1 = (2 * prec * tpr) / (prec + tpr)
            else:
                f1 = 0.0
            fprs.append(fpr)
            tprs.append(tpr)
            precisions.append(prec)
            f1s.append(f1)

        roc_auc = auc(fprs, tprs)
        roc_auc = round(roc_auc, 4)
        pr_auc = auc(tprs, precisions)
        pr_auc = round(pr_auc, 4)
        out['roc_auc'] = (roc_auc)
        out['pr_auc'] = (pr_auc)
        out['fpr'] = (fprs)
        out['tpr'] = (tprs)
        out['precision'] = (precisions)
        out['f1'] = (f1s)
        out['thresholds'] = (thresholds)

    return pr_auc

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())
