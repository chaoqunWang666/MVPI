import os
import sys
import gc
import time
import json
import math
import resource
import argparse
import traceback
from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
# ensure repo root is on sys.path so imports like `trainer` and `datasets` work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import trainer as trainer_module
from trainer import Trainer
from datasets.mvtec import MVTEC_CLASS_NAMES
from utils.utils import compute_pro_retrieval_metrics as original_compute_pro_retrieval_metrics


# --------------------------------------------------------------------------- #
# multi-GPU sweep configurations
# --------------------------------------------------------------------------- #
# (config name, (n_views_y, n_views_x), aggregation rule)
AGG_CONFIGS = [
    ('single_view', (1, 1), 'midpoint'),
    ('disjoint_2x2', (2, 2), 'midpoint'),
    ('avg_16', (4, 4), 'avg'),
    ('midpoint_16', (4, 4), 'midpoint'),
]
# views sweep (always midpoint/disjoint aggregation)
VIEW_CONFIGS = {
    'v1': (1, 1),
    'v4': (2, 2),
    'v9': (3, 3),
    'v16': (4, 4),
    'v25': (5, 5),
}


def compute_axis_patch_split(dim_size, n_views_axis, patch_size_px=224, token_size=16):
    # compute starts in pixels per user's formula
    if n_views_axis == 1:
        starts_px = [0]
    else:
        starts_px = [token_size * math.floor((k * (dim_size - patch_size_px) / (token_size * (n_views_axis - 1))))
                     for k in range(n_views_axis)]
    patch_t = patch_size_px // token_size
    # convert to token units
    input_rois_tokens = [(s // token_size, s // token_size + patch_t) for s in starts_px]

    pred_rois = []
    for i in range(len(input_rois_tokens)):
        if i == 0:
            s = 0
        else:
            s = math.ceil((input_rois_tokens[i - 1][1] - input_rois_tokens[i][0]) / 2)
        if i == len(input_rois_tokens) - 1:
            e = patch_t
        else:
            e = patch_t - math.floor((input_rois_tokens[i][1] - input_rois_tokens[i + 1][0]) / 2)
        pred_rois.append((s, e))

    res_rois = [(0, pred_rois[0][1] - pred_rois[0][0])]
    for i in range(1, len(pred_rois)):
        s = res_rois[i - 1][1]
        res_rois.append((s, s + pred_rois[i][1] - pred_rois[i][0]))

    input_rois_pixels = [(s * token_size, e * token_size) for s, e in input_rois_tokens]
    return input_rois_pixels, pred_rois, res_rois


class EvalEncoder(nn.Module):
    """Frozen backbone wrapper that tiles the input into ``n_views`` crops.

    ``n_views`` / ``agg`` are constructor arguments (instead of monkey-patched
    bound methods) so no reference cycle is created around the encoder.
    """

    def __init__(self, raw_model, target_layers, inp_size=448, patch_size=224, token_size=16,
                 device='cpu', n_views=(1, 1), agg='midpoint'):
        super().__init__()
        self.model = raw_model
        self.target_layers = target_layers
        self.inp_size = inp_size
        self.patch_size = patch_size
        self.token_size = token_size
        self.patch_t = patch_size // token_size
        self.device = device
        self.n_views = tuple(n_views)
        self.agg = agg
        n_views_total = self.n_views[0] * self.n_views[1]
        self.view_layout = {'enabled': n_views_total > 1, 'n_views': n_views_total}
        # only request the blocks we use (ascending order) instead of all 12
        self._sorted_layers = sorted(set(target_layers))
        self._gather_order = [self._sorted_layers.index(i) for i in target_layers]

    def _backbone(self, x):
        with torch.no_grad():
            outputs = self.model.get_intermediate_layers(x.to(self.device), n=self._sorted_layers)
        return [outputs[i] for i in self._gather_order]

    def _reshape_outputs(self, outputs):
        maps = []
        for out in outputs:
            B, N, C = out.shape
            H = W = int(N ** 0.5)
            maps.append(out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous())
        return maps

    def forward(self, x, n_views=None, agg=None):
        n_views = self.n_views if n_views is None else n_views
        agg = self.agg if agg is None else agg
        # x: (B,3,H,W) expect B=1
        B, C, H, W = x.shape
        n_y, n_x = n_views

        # SINGLE VIEW == feed the WHOLE image to the backbone once (28x28 tokens).
        # Going through the tiling code with n_views=(1,1) would crop only the
        # top-left 224x224 patch and leave the remaining 3/4 of the token grid at
        # zero; those zero tokens normalise to zero, get reconstructed into a
        # non-zero vector, and score a constant 1 - cos = 1, which destroys the
        # ranking (I-AUC ~0.5, PRO ~0).
        if n_y == 1 and n_x == 1:
            return self._reshape_outputs(self._backbone(x))

        # compute axis splits
        in_y_px, pred_y, res_y = compute_axis_patch_split(H, n_y, self.patch_size, self.token_size)
        in_x_px, pred_x, res_x = compute_axis_patch_split(W, n_x, self.patch_size, self.token_size)

        # gather crops
        patches = []
        for (ys, ye) in in_y_px:
            for (xs, xe) in in_x_px:
                patches.append(x[:, :, ys:ye, xs:xe])
        patches = torch.cat(patches, dim=0) if len(patches) > 0 else \
            torch.zeros((0, C, self.patch_size, self.patch_size), device=x.device)

        # forward backbone on patches in a single batch
        # list per layer, each: (n_patches, C, th, tw)
        patch_maps = self._reshape_outputs(self._backbone(patches))

        tokens_h = H // self.token_size
        tokens_w = W // self.token_size
        merged = []
        for fmap in patch_maps:
            n_patches, ch, th, tw = fmap.shape
            # arrange fmap as (B, n_patches, C, th, tw)
            fmap = fmap.reshape(1, -1, ch, th, tw)
            full = torch.zeros((B, ch, tokens_h, tokens_w), device=fmap.device, dtype=fmap.dtype)
            if agg == 'avg':
                count = torch.zeros((1, 1, tokens_h, tokens_w), device=fmap.device, dtype=fmap.dtype)
                for i, ((ys, ye), (xs, xe)) in enumerate(product(in_y_px, in_x_px)):
                    s_y = ys // self.token_size
                    e_y = ye // self.token_size
                    s_x = xs // self.token_size
                    e_x = xe // self.token_size
                    full[:, :, s_y:e_y, s_x:e_x] += fmap[:, i]
                    count[:, :, s_y:e_y, s_x:e_x] += 1
                merged.append(full / count)
            else:
                # midpoint / disjoint placement: follow trainer._forward_with_overlap logic
                for i, (pidx, ridx) in enumerate(zip(product(pred_y, pred_x), product(res_y, res_x))):
                    py, px = pidx
                    ry, rx = ridx
                    # py,px are token ranges inside patch; ry,rx are token ranges in result
                    full[:, :, ry[0]:ry[1], rx[0]:rx[1]] = fmap[:, i, :, py[0]:py[1], px[0]:px[1]]
                merged.append(full)
        return merged


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def raise_fd_limit(target=8192):
    """Raise RLIMIT_NOFILE to the hard limit (best effort).

    The sweep used to die with EMFILE because DataLoader workers inherited an
    ever-growing set of file descriptors / shared memory handles.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard == resource.RLIM_INFINITY:
            new_soft = max(soft, target)
        else:
            new_soft = max(soft, min(target, hard))
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"[*] RLIMIT_NOFILE soft={soft} hard={hard}")
    except Exception as exc:  # pragma: no cover - purely defensive
        print(f"[WARN] could not raise RLIMIT_NOFILE: {exc}")


def parse_gpus(spec):
    """Resolve the GPU id list.

    ``auto`` (default) uses every GPU visible to this process.  Note that CUDA
    ordinals are re-numbered inside containers / by CUDA_VISIBLE_DEVICES, so
    "physical cards 1,2,3" usually show up as 0,1,2.
    """
    available = torch.cuda.device_count()
    if str(spec).strip().lower() == 'auto':
        return list(range(available))
    gpus = [int(g.strip()) for g in str(spec).split(',') if g.strip() != '']
    if not gpus:
        gpus = [0]
    invalid = [g for g in gpus if g < 0 or g >= available]
    if invalid:
        raise ValueError(
            'Invalid GPU id(s) {}: this process sees {} GPU(s) with ordinals {}. '
            'Use --gpus auto to take them all.'.format(invalid, available, list(range(available))))
    # de-duplicate while keeping order
    seen, uniq = set(), []
    for g in gpus:
        if g not in seen:
            seen.add(g)
            uniq.append(g)
    return uniq


def build_base_args(args, device):
    base_args = argparse.Namespace()
    base_args.backbone_arch = 'dinov3_vitb16'
    base_args.inp_size = args.inp_size
    base_args.target_layers = [2, 5, 8]
    base_args.feature_levels = 3
    base_args.n_heads = 8
    base_args.n_layers = 3
    base_args.jitter_scale = 0.0
    base_args.num_workers = args.num_workers
    base_args.pin_memory = args.pin_memory
    base_args.batch_size = 1
    # keep 1: this script reports per-image FPS and peak memory, so the batch size
    # must stay at 1 for those numbers to stay comparable across configs
    base_args.test_batch_size = 1
    base_args.prefetch_factor = args.prefetch_factor
    base_args.lr = 1e-4
    base_args.save_path = 'checkpoints'
    base_args.save_prefix = ''
    base_args.device = device
    base_args.fusion_weights = [0.4, 0.3, 0.3]
    base_args.img_pool = 'topk_mean'
    base_args.topk = 100
    base_args.select = 'best_pro'
    # dataset / preprocessing defaults expected by Dataset classes
    base_args.data_path = args.data_path
    base_args.img_size = (args.inp_size, args.inp_size)
    base_args.crop_size = (args.inp_size, args.inp_size)
    base_args.norm_mean = [0.485, 0.456, 0.406]
    base_args.norm_std = [0.229, 0.224, 0.225]
    base_args.use_mvpi = 1
    base_args.overlap_patch_size = 224
    base_args.overlap_size = 64
    base_args.mvpi_agg = 'disjoint'
    base_args.num_epochs = 1
    # eval-only: no train split, one backbone shared by all classes
    base_args.eval_only = 1
    base_args.reuse_backbone = 1
    return base_args


def attach_encoder(trainer, n_views, agg, device, inp_size):
    """Swap the trainer's encoder for the requested tiling configuration."""
    raw_model = trainer.encoder.model
    trainer.encoder = None
    gc.collect()
    enc = EvalEncoder(raw_model, target_layers=[2, 5, 8], inp_size=inp_size,
                      device=device, n_views=n_views, agg=agg)
    trainer.encoder = enc.to(device).eval()
    trainer.view_layout = enc.view_layout
    return trainer


def evaluate_subset(args, class_names, device, tag=''):
    """Evaluate every config on ``class_names`` using a single GPU/process."""
    pro_fallback_hits = {'count': 0}
    current_context = {'class_name': None, 'config_name': None}

    def safe_compute_pro_retrieval_metrics(scores, gt_mask):
        try:
            return original_compute_pro_retrieval_metrics(scores, gt_mask)
        except ValueError as exc:
            if 'At least 2 points are needed' in str(exc):
                pro_fallback_hits['count'] += 1
                print(
                    f"{tag} [WARN] PRO fallback triggered for class={current_context['class_name']} "
                    f"config={current_context['config_name']}"
                )
                return 0.0
            raise

    # must be patched inside every (spawned) worker process
    trainer_module.compute_pro_retrieval_metrics = safe_compute_pro_retrieval_metrics

    base_args = build_base_args(args, device)
    results = {
        'agg_rule': {name: {} for name, _, _ in AGG_CONFIGS},
        'views_sweep': {name: {} for name in VIEW_CONFIGS},
    }

    # ONE Trainer for the whole subset: backbone + reconstruction heads are class
    # independent, only the checkpoint and the test loader change.  Building a
    # Trainer per class used to fork 2 x num_workers processes 15 x 9 times.
    base_args.class_name = class_names[0]
    trainer = Trainer(base_args)
    t_start = time.time()
    try:
        for cfg_name, n_views, agg in AGG_CONFIGS:
            print(f"{tag} Evaluating agg_rule: {cfg_name}")
            attach_encoder(trainer, n_views, agg, device, args.inp_size)
            for cls in class_names:
                current_context['class_name'] = cls
                current_context['config_name'] = cfg_name
                gc.collect()
                torch.cuda.empty_cache()
                trainer.build_test_loader(cls)
                img_auc, pix_auc, pro_auc = trainer.test(vis=False, checkpoint_path=args.checkpoint_root)
                results['agg_rule'][cfg_name][cls] = {
                    'I-AUC': float(img_auc), 'P-AUC': float(pix_auc), 'PRO': float(pro_auc)}
                print(f"{tag}   {cfg_name}/{cls}: I-AUC={img_auc:.4f} P-AUC={pix_auc:.4f} PRO={pro_auc:.4f}")

        for cfg_name, n_views in VIEW_CONFIGS.items():
            print(f"{tag} Evaluating views sweep: {cfg_name}")
            attach_encoder(trainer, n_views, 'midpoint', device, args.inp_size)
            for cls in class_names:
                current_context['class_name'] = cls
                current_context['config_name'] = cfg_name
                gc.collect()
                torch.cuda.empty_cache()
                trainer.build_test_loader(cls)
                img_auc, pix_auc, pro_auc = trainer.test(vis=False, checkpoint_path=args.checkpoint_root)
                results['views_sweep'][cfg_name][cls] = {
                    'I-AUC': float(img_auc), 'P-AUC': float(pix_auc), 'PRO': float(pro_auc),
                    'FPS': float(trainer.timing.get('fps', 0.0)),
                    'Mem_MB': float(trainer.timing.get('peak_gpu_mem_MB', 0.0)),
                }
                print(f"{tag}   {cfg_name}/{cls}: I-AUC={img_auc:.4f} P-AUC={pix_auc:.4f} "
                      f"PRO={pro_auc:.4f} FPS={trainer.timing.get('fps', 0.0)}")
    finally:
        trainer.close()
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    if pro_fallback_hits['count']:
        print(f"{tag} [WARN] PRO fallback hits: {pro_fallback_hits['count']}")
    print(f"{tag} done in {(time.time() - t_start) / 60:.1f} min "
          f"({len(class_names)} classes x {len(AGG_CONFIGS) + len(VIEW_CONFIGS)} configs)")
    results['_pro_fallbacks'] = pro_fallback_hits['count']
    return results


def _worker_entry(gpu_id, class_names, args, queue):
    """Entry point of one GPU worker process (spawn start method)."""
    try:
        raise_fd_limit()
        torch.cuda.set_device(gpu_id)
        device = f'cuda:{gpu_id}'
        print(f"[cuda:{gpu_id}] classes: {class_names}", flush=True)
        res = evaluate_subset(args, class_names, device, tag=f"[cuda:{gpu_id}]")
        queue.put((gpu_id, None, res))
    except BaseException:  # noqa: BLE001 - report everything back to the parent
        queue.put((gpu_id, traceback.format_exc(), None))


def merge_results(parts):
    """Average the per-class records produced by every GPU worker."""
    per_class = {'agg_rule': {}, 'views_sweep': {}}
    fallback_total = 0
    for part in parts:
        fallback_total += part.get('_pro_fallbacks', 0)
        for section in ('agg_rule', 'views_sweep'):
            for cfg, records in part.get(section, {}).items():
                per_class[section].setdefault(cfg, {}).update(records)

    out = {'agg_rule': {}, 'views_sweep': {}, 'per_class': per_class}
    for section in ('agg_rule', 'views_sweep'):
        for cfg, records in per_class[section].items():
            if not records:
                continue
            keys = list(next(iter(records.values())).keys())
            out[section][cfg] = {k: float(np.mean([r[k] for r in records.values()])) for k in keys}
    out['_pro_fallbacks'] = fallback_total
    return out


def run_eval(args):
    raise_fd_limit()
    gpus = parse_gpus(args.gpus)
    if args.classes.strip().lower() != 'all':
        keep = [c.strip() for c in args.classes.split(',') if c.strip()]
        class_names = [c for c in keep if c in MVTEC_CLASS_NAMES]
        missing = [c for c in keep if c not in MVTEC_CLASS_NAMES]
        if missing:
            raise ValueError('Unknown MVTec classes: {}'.format(missing))
    else:
        class_names = list(MVTEC_CLASS_NAMES)

    n_gpus = min(len(gpus), len(class_names))
    gpus = gpus[:n_gpus]
    # round-robin split keeps the per-GPU load balanced across class sizes
    chunks = [class_names[i::n_gpus] for i in range(n_gpus)]
    print('[*] GPUs: {} | classes per GPU: {}'.format(
        gpus, {g: c for g, c in zip(gpus, chunks)}))

    if args.device:
        # explicit --device wins over --gpus (single process, no worker spawn)
        device = args.device
        if device.startswith('cuda:'):
            torch.cuda.set_device(int(device.split(':')[1]))
        parts = [evaluate_subset(args, class_names, device, tag=f"[{device}]")]
    elif n_gpus <= 1:
        device = f'cuda:{gpus[0]}'
        torch.cuda.set_device(gpus[0])
        parts = [evaluate_subset(args, class_names, device, tag=f"[cuda:{gpus[0]}]")]
    else:
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()
        procs = []
        for gpu, chunk in zip(gpus, chunks):
            p = ctx.Process(target=_worker_entry, args=(gpu, chunk, args, queue))
            p.start()
            procs.append(p)
        parts = []
        errors = []
        for _ in procs:
            gpu_id, err, res = queue.get()
            if err:
                errors.append((gpu_id, err))
            else:
                parts.append(res)
        for p in procs:
            p.join()
        if errors:
            for gpu_id, err in errors:
                print(f"[ERROR] worker cuda:{gpu_id} failed:\n{err}")
            raise RuntimeError('{} GPU worker(s) failed'.format(len(errors)))

    results = merge_results(parts)
    if results.get('_pro_fallbacks'):
        print(f"[WARN] Total PRO fallback hits: {results['_pro_fallbacks']}")

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved results to', args.output_json)

    for section in ('agg_rule', 'views_sweep'):
        print('---- {} ----'.format(section))
        for cfg, metrics in results[section].items():
            print(cfg, {k: round(v, 4) for k, v in metrics.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_root', type=str, default='checkpoints/MVTEC')
    parser.add_argument('--output_json', type=str, default='eval_results_mvtec.json')
    parser.add_argument('--inp_size', type=int, default=448)
    parser.add_argument('--data_path', type=str, default='/path/to/mvtec')
    parser.add_argument('--device', type=str, default=None,
                        help='run everything in one process on this device (e.g. "cuda:0"), '
                             'overrides --gpus')
    parser.add_argument('--gpus', type=str, default='auto',
                        help='"auto" = every visible GPU (default), or comma separated CUDA '
                             'ordinals, e.g. "0,1,2"')
    parser.add_argument('--classes', type=str, default='all',
                        help='comma separated MVTec class list, or "all"')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoader workers per GPU process (default: 2)')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='batches prefetched per worker (default: 2)')
    parser.add_argument('--pin_memory', type=int, default=1, choices=[0, 1],
                        help='0 disables pin_memory (removes shared-memory transfers)')
    args = parser.parse_args()
    # resolve repo-relative defaults (checkpoints/, ...) so the script works from
    # any current working directory
    if not os.path.isabs(args.checkpoint_root):
        args.checkpoint_root = os.path.join(ROOT, args.checkpoint_root)
    run_eval(args)
