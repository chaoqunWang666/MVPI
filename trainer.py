import os
import gc
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur
from math import ceil, floor
from itertools import product
from utils.utils import *
from utils.visualizer import Visualizer
from datasets.mvtec import MVTecDataset, MVTEC_CLASS_NAMES
from datasets.CAD import CADDataset, CAD_CLASS_NAMES
from datasets.btad import BTADDataset, BTAD_CLASS_NAMES
from models.model import MVPI
from models.dinov3_loader import load_dinov3_from_local
try:
    from thop import profile
except ImportError:
    pass


# process-wide cache of frozen DINOv3 backbones, keyed by (arch, device).
# Populated only when args.reuse_backbone is set (see Trainer._load_backbone).
_BACKBONE_CACHE = {}


class DinoV3Wrapper(nn.Module):
    """Frozen DINOv3 feature extractor with optional Multi-View Patch Inference (MVPI).

    agg='disjoint' : MVPI token placement. Each view only contributes its central
                     (non-overlapping) region; overlapping border tokens are discarded,
                     so every token in the fused map comes from exactly one view and
                     no defect can fall on a view boundary.
    agg='avg'      : plain overlapping-tiling baseline. Identical crops and identical
                     number of backbone forward passes, but ALL tokens of every view
                     are averaged over their coverage count (SAHI-style fusion).
    """

    def __init__(self, model, target_layers, fusion_groups=None,
                 overlap_patch_size=224, overlap_size=64, agg='disjoint'):
        super().__init__()
        self.model = model
        self.target_layers = target_layers
        # get_intermediate_layers accepts a list of block indices, so we can ask
        # only for the layers we actually use instead of materialising (and
        # norm-ing) all 12 of them. Output order is always ascending block index,
        # so keep a permutation back to the caller's order.
        self._sorted_layers = sorted(target_layers)
        self._gather_order = [self._sorted_layers.index(i) for i in target_layers]
        self.fusion_groups = fusion_groups
        self.patch_size = 16
        self.overlap_patch_size = overlap_patch_size
        self.overlap_size = overlap_size
        self.agg = agg

    def _backbone_forward(self, x):
        # `n` may be a list of block indices -> only those blocks are collected
        # and norm-ed (all blocks are still executed, they have to be).
        all_outputs = self.model.get_intermediate_layers(x, n=list(self._sorted_layers))
        return [all_outputs[i] for i in self._gather_order]

    def _reshape_outputs(self, outputs):
        maps = []
        for out in outputs:
            B, N, C = out.shape
            H = W = int(N ** 0.5)
            assert H * W == N, f"Invalid patch shape: N={N}"
            maps.append(out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous())
        return maps

    def _axis_patch_split(self, dim_size):
        patch_size = self.overlap_patch_size
        overlap = self.overlap_size
        assert dim_size >= patch_size
        assert overlap > 0 and patch_size > 2 * overlap
        assert patch_size % self.patch_size == 0 and overlap % self.patch_size == 0
        dim_t = dim_size // self.patch_size
        patch_t = patch_size // self.patch_size
        overlap_t = overlap // self.patch_size
        n_patches = ceil((dim_t - patch_t) / (patch_t - 2 * overlap_t)) + 1
        fac, div = dim_t - patch_t, max(1, n_patches - 1)
        input_rois = [((i * fac) // div, (i * fac) // div + patch_t) for i in range(n_patches)]
        pred_rois = []
        for i in range(n_patches):
            s = 0 if i == 0 else ceil((input_rois[i - 1][1] - input_rois[i][0]) / 2)
            e = patch_t if i == n_patches - 1 else patch_t - floor((input_rois[i][1] - input_rois[i + 1][0]) / 2)
            pred_rois.append((s, e))
        res_rois = [(0, pred_rois[0][1] - pred_rois[0][0])]
        for i in range(1, n_patches):
            s = res_rois[i - 1][1]
            res_rois.append((s, s + pred_rois[i][1] - pred_rois[i][0]))
        input_rois = [(s * self.patch_size, e * self.patch_size) for s, e in input_rois]
        return input_rois, pred_rois, res_rois

    def view_layout(self, h, w):
        """Concrete MVPI geometry for an h x w input (Reviewer 1-1)."""
        if not (self.overlap_patch_size and self.overlap_size):
            return {'enabled': False, 'n_views': 1}
        in_y, _, _ = self._axis_patch_split(h)
        in_x, _, _ = self._axis_patch_split(w)
        starts_y = [s for s, _ in in_y]
        starts_x = [s for s, _ in in_x]
        stride_y = (starts_y[1] - starts_y[0]) if len(starts_y) > 1 else h
        stride_x = (starts_x[1] - starts_x[0]) if len(starts_x) > 1 else w
        return {
            'enabled': True,
            'crop_size_px': self.overlap_patch_size,
            'n_views_per_axis': (len(in_y), len(in_x)),
            'n_views': len(in_y) * len(in_x),
            'crop_starts_x_px': starts_x,
            'crop_starts_y_px': starts_y,
            'stride_px': [stride_x, stride_y],
            'overlap_px': [self.overlap_patch_size - stride_x, self.overlap_patch_size - stride_y],
            'overlap_ratio': round(
                (self.overlap_patch_size - stride_x) / self.overlap_patch_size, 4),
            'agg': self.agg,
        }

    def _gather_crops(self, x):
        b, c, h, w = x.shape
        in_y, pred_y, res_y = self._axis_patch_split(h)
        in_x, pred_x, res_x = self._axis_patch_split(w)
        ph = pw = self.overlap_patch_size
        patches = torch.zeros((b, len(in_y) * len(in_x), c, ph, pw), device=x.device, dtype=x.dtype)
        for i, ((ys, ye), (xs, xe)) in enumerate(product(in_y, in_x)):
            patches[:, i] = x[:, :, ys:ye, xs:xe]
        outputs = self._backbone_forward(patches.reshape(-1, c, ph, pw))
        patch_maps = self._reshape_outputs(outputs)
        return patch_maps, (in_y, pred_y, res_y), (in_x, pred_x, res_x)

    def _forward_with_overlap(self, x):
        b, c, h, w = x.shape
        tokens_h, tokens_w = h // self.patch_size, w // self.patch_size
        patch_maps, (in_y, pred_y, res_y), (in_x, pred_x, res_x) = self._gather_crops(x)

        if self.agg == 'avg':
            # Plain overlapping-tiling baseline: average ALL tokens of every view
            # over their coverage count. Same crops, same #forward passes.
            merged = []
            for fmap in patch_maps:
                _, ch, th, tw = fmap.shape
                fmap = fmap.reshape(b, -1, ch, th, tw)
                full = torch.zeros((b, ch, tokens_h, tokens_w), device=fmap.device, dtype=fmap.dtype)
                count = torch.zeros((1, 1, tokens_h, tokens_w), device=fmap.device, dtype=fmap.dtype)
                for i, ((ys, ye), (xs, xe)) in enumerate(product(in_y, in_x)):
                    full[:, :, ys // self.patch_size:ye // self.patch_size,
                         xs // self.patch_size:xe // self.patch_size] += fmap[:, i]
                    count[:, :, ys // self.patch_size:ye // self.patch_size,
                          xs // self.patch_size:xe // self.patch_size] += 1
                merged.append(full / count)
            return merged

        # MVPI 'disjoint' placement: each view contributes only its central region
        merged = []
        for fmap in patch_maps:
            _, ch, th, tw = fmap.shape
            fmap = fmap.reshape(b, -1, ch, th, tw)
            full = torch.zeros((b, ch, tokens_h, tokens_w), device=fmap.device, dtype=fmap.dtype)
            for i, ((py, px), (ry, rx)) in enumerate(zip(product(pred_y, pred_x), product(res_y, res_x))):
                full[:, :, ry[0]:ry[1], rx[0]:rx[1]] = fmap[:, i, :, py[0]:py[1], px[0]:px[1]]
            merged.append(full)
        return merged

    def forward(self, x):
        use_overlap = self.overlap_patch_size and self.overlap_size and min(x.shape[-2:]) > self.overlap_patch_size
        reshaped_maps = self._forward_with_overlap(x) if use_overlap else self._reshape_outputs(self._backbone_forward(x))
        if self.fusion_groups:
            return [torch.stack([reshaped_maps[i] for i in group], dim=1).mean(dim=1) for group in self.fusion_groups]
        return reshaped_maps

    def feature_info_channels(self):
        if self.fusion_groups:
            return [768] * len(self.fusion_groups)
        return [768] * len(self.target_layers)


class Trainer(object):
    def __init__(self, args):
        # Note: do not force-overwrite `args.inp_size` here; keep caller's setting.
        self.args = args

        # getattr keeps this working with hand-built Namespaces (e.g. scripts/eval_mvtec.py)
        num_workers = getattr(args, 'num_workers', 4)
        self._loader_kwargs = {'num_workers': num_workers,
                               'pin_memory': bool(getattr(args, 'pin_memory', 1))}
        if num_workers > 0:
            # Without this, every epoch re-forks the workers twice (train -> test ->
            # train), which shows up as a long CPU-bound gap with an idle GPU.
            self._loader_kwargs['persistent_workers'] = True
            self._loader_kwargs['prefetch_factor'] = getattr(args, 'prefetch_factor', 4)

        # eval-only runs never touch the train split: building that loader too
        # doubles the number of worker processes (and their inherited fds).
        self.train_loader = None
        if not getattr(args, 'eval_only', 0):
            train_dataset = self._make_dataset(args, is_train=True)
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                **self._loader_kwargs)

        self.test_loader = None
        self.build_test_loader()

        self.timing = {}
        self.build_model()
        self.cos_criterion = nn.CosineSimilarity(dim=-1)

    @staticmethod
    def _make_dataset(args, is_train):
        if args.class_name in MVTEC_CLASS_NAMES:
            ds_cls = MVTecDataset
        elif args.class_name in CAD_CLASS_NAMES:
            ds_cls = CADDataset
        elif args.class_name in BTAD_CLASS_NAMES:
            ds_cls = BTADDataset
        else:
            raise ValueError('Invalid Class Name: {}'.format(args.class_name))
        return ds_cls(args, is_train=is_train)

    def build_test_loader(self, class_name=None):
        """(Re)build the test loader, optionally switching to another class.

        Lets a single ``Trainer`` be reused for every class (the backbone and the
        reconstruction heads are class independent - only the checkpoint and the
        test set change).  The previous loader is explicitly shut down first.
        """
        if class_name is not None:
            self.args.class_name = class_name
        # persistent_workers keeps the worker processes (and every fd / shared
        # memory handle they inherited) alive until the loader is destroyed; just
        # dropping the reference is not enough, so shut the iterator down here.
        self.close_loader('test_loader')
        test_dataset = self._make_dataset(self.args, is_train=False)
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=getattr(self.args, 'test_batch_size', 1),
            shuffle=False, drop_last=False, **self._loader_kwargs)
        return self.test_loader

    def close_loader(self, name):
        """Shut down a DataLoader's workers + pin-memory thread and drop it."""
        loader = getattr(self, name, None)
        if loader is None:
            return
        try:
            iterator = getattr(loader, '_iterator', None)
            if iterator is not None:
                iterator._shutdown_workers()
                loader._iterator = None
        except Exception:
            pass
        del loader
        setattr(self, name, None)

    def close(self):
        """Release every DataLoader worker (fd / shm leak guard)."""
        self.close_loader('train_loader')
        self.close_loader('test_loader')
        gc.collect()

    def _load_backbone(self):
        """Load the frozen DINOv3 backbone, optionally from a process-wide cache.

        ``torch.hub.load`` re-imports hubconf.py and re-reads the (hundreds of MB)
        checkpoint on *every* call, and each call leaves behind file descriptors
        and temp directories.  Doing that once per class per config is slow and is
        one of the reasons a long sweep runs into EMFILE ("Too many open files").
        """
        if not getattr(self.args, 'reuse_backbone', 0):
            print("Loading DINOv3 Encoder...")
            return load_dinov3_from_local(device=self.args.device)

        key = (getattr(self.args, 'backbone_arch', 'dinov3_vitb16'), str(self.args.device))
        if key in _BACKBONE_CACHE:
            print("Loading DINOv3 Encoder... (reusing cached backbone)")
            return _BACKBONE_CACHE[key]
        print("Loading DINOv3 Encoder...")
        _BACKBONE_CACHE[key] = load_dinov3_from_local(device=self.args.device)
        return _BACKBONE_CACHE[key]

    def build_model(self):
        raw_model = self._load_backbone()

        # multi-scale layer selection ({2, 5, 8} by default); single-scale ablation
        # (use_mrft=0) keeps only the deepest layer and is handled in main.py
        target_layers = self.args.target_layers
        assert max(target_layers) < len(raw_model.blocks), \
            'target_layers out of range for {} blocks'.format(len(raw_model.blocks))

        use_mvpi = bool(getattr(self.args, 'use_mvpi', 1))
        mvpi_agg = getattr(self.args, 'mvpi_agg', 'disjoint')
        overlap_patch_size = getattr(self.args, 'overlap_patch_size', 224) if use_mvpi else None
        overlap_size = getattr(self.args, 'overlap_size', 64) if use_mvpi else None
        print(f"MVPI enabled: {use_mvpi} | agg: {mvpi_agg} | patch: {overlap_patch_size} | overlap: {overlap_size}")
        self.encoder = DinoV3Wrapper(
            raw_model,
            target_layers=target_layers,
            fusion_groups=None,
            overlap_patch_size=overlap_patch_size,
            overlap_size=overlap_size,
            agg=mvpi_agg
        ).to(self.args.device).eval()

        self.view_layout = self.encoder.view_layout(self.args.inp_size, self.args.inp_size)
        print("View layout:", self.view_layout)

        feat_dims = self.encoder.feature_info_channels()
        print("Feature Dimensions:", feat_dims)

        models = []

        patch_H = self.args.inp_size // 16
        seq_len = patch_H * patch_H

        self.seq_lens = [seq_len] * len(target_layers)
        self.ws = [patch_H] * len(target_layers)
        d_models = [768] * len(target_layers)

        if self.args.feature_levels != len(target_layers):
            print(f"Warning: Adjusting feature_levels from {self.args.feature_levels} to {len(target_layers)}")
            self.args.feature_levels = len(target_layers)

        for idx, (seq_len, in_channels, d_model) in enumerate(zip(self.seq_lens, feat_dims, d_models)):
            cur_layer_idx = target_layers[idx]
            cur_layer_name = f'layer{cur_layer_idx}'

            model = MVPI(seq_len=seq_len,
                        in_channels=in_channels,
                        out_channels=in_channels,
                        d_model=d_model,
                        n_heads=self.args.n_heads,
                        n_layers=self.args.n_layers,
                        d_feed_foward_scale=4,
                        dropout=0.0,
                        args=self.args,
                        layer_name=[cur_layer_name],
                        jitter_scale=self.args.jitter_scale)

            print(f'One Model (Layer {cur_layer_idx})...Done')
            models.append(model.to(self.args.device))
            model.layer_key = [cur_layer_name]

        self.models = models
        print('Creating Models...Done')

        params = list(models[0].parameters())
        for l in range(1, self.args.feature_levels):
            params += list(models[l].parameters())

        self.optimizer = torch.optim.Adam(params, lr=self.args.lr)

        # Use local average pooling to reduce spatial misalignment.
        self.avg_pool = torch.nn.AvgPool2d(3, 1, 1)
        # ====== Compute parameter count and FLOPs ======
        # NOTE: keep this enabled when checkpoints are restored with strict=True:
        # thop's profile() registers `total_ops` / `total_params` buffers, which
        # end up inside the saved state_dict.
        try:
            dummy_input = torch.randn(1, 3, self.args.inp_size, self.args.inp_size).to(self.args.device)
            macs_enc, params_enc = profile(self.encoder, inputs=(dummy_input, ), verbose=False)

            features = self.encoder(dummy_input)
            macs_models, params_models = 0, 0
            for fl in range(self.args.feature_levels):
                inp = self.avg_pool(features[fl])
                inp = F.normalize(inp, p=2, dim=1)
                N, D, _, _ = inp.shape
                inp = inp.permute(0, 2, 3, 1).reshape(N, -1, D)
                mac, prm = profile(self.models[fl], inputs=(inp, ), verbose=False)
                macs_models += mac
                params_models += prm

            print(f"[*] Backbone FLOPs (FLOPs): {macs_enc * 2 / 1e9:.2f} G | Parameters: {params_enc / 1e6:.2f} M (Frozen)")
            print(f"[*] Reconstruction FLOPs (FLOPs): {macs_models * 2 / 1e9:.2f} G | Parameters: {params_models / 1e6:.2f} M (Trainable)")

            total_macs = macs_enc + macs_models
            total_params = params_enc + params_models
            print(f"[*] Total model parameters (Params): {total_params / 1e6:.2f} M")
            print(f"[*] Total FLOPs (FLOPs): {total_macs * 2 / 1e9:.2f} GFLOPs")
            self.timing.update({
                'macs_backbone_G': round(macs_enc / 1e9, 3),
                'macs_mrft_G': round(macs_models / 1e9, 3),
                'params_backbone_M': round(params_enc / 1e6, 2),
                'params_mrft_M': round(params_models / 1e6, 2),
            })
        except NameError:
            print("Failed to calculate FLOPs. Install thop with: pip install thop")

    def resolve_view_layout(self):
        """Return ``self.view_layout`` as a plain dict.

        ``DinoV3Wrapper.view_layout`` is a *method* (it needs h/w to derive the
        tiling geometry), but callers such as ``scripts/eval_mvtec.py`` assign
        either a dict or a callable depending on the config. Normalise here so
        ``test()`` never has to care which one it got.
        """
        vl = self.view_layout
        if callable(vl):
            vl = vl(self.args.inp_size, self.args.inp_size)
        if vl is None:
            vl = {'n_views': 1}
        self.view_layout = vl
        return vl

    @torch.no_grad()
    def _build_feature_cache(self):
        """Extract the frozen backbone's train-set features exactly once.

        The encoder is frozen and the train images never change, so re-running it
        every epoch is ``num_epochs``x redundant work. Memory cost on the host is
        N x C x H x W x feature_levels x 4 bytes (e.g. 400 images @448px with 3
        levels -> ~2.9 GB of RAM). Enable with ``--cache_train_features``.
        """
        t0 = time.time()
        caches = [[] for _ in range(self.args.feature_levels)]
        for (images, _, _, _, _) in self.train_loader:
            images = images.float().to(self.args.device)
            features = self.encoder(images)
            for fl in range(self.args.feature_levels):
                caches[fl].append(features[fl].to('cpu'))
            # host-side copy of these tensors happens above; make sure it is done
            # before we reuse the GPU buffers in the next iteration
            torch.cuda.synchronize()
        caches = [torch.cat(c, dim=0) for c in caches]
        mem_mb = sum(c.numel() * c.element_size() for c in caches) / 1e6
        print('[*] Cached training features: {} images x {} layers | {:.0f} MB (CPU RAM) | extraction time {:.1f}s'.format(
            caches[0].shape[0], len(caches), mem_mb, time.time() - t0))
        return caches

    def _cached_batches(self, feat_cache):
        """Yield shuffled mini-batches out of the pre-extracted feature cache."""
        n = feat_cache[0].shape[0]
        order = torch.randperm(n)
        bs = self.args.batch_size
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            yield [c[idx].to(self.args.device) for c in feat_cache], None

    def train(self):
        path = os.path.join(self.args.save_path, self.args.save_prefix)
        if not os.path.exists(path):
            os.makedirs(path)

        start_time = time.time()
        train_steps = len(self.train_loader)
        best_img_auc, best_pix_auc, best_pro_auc = 0.0, 0.0, 0.0
        last_metrics = (0.0, 0.0, 0.0)
        select = getattr(self.args, 'select', 'best_pro')

        # The backbone is frozen, so its train-set features never change. With
        # --cache_train_features they are extracted once instead of once per epoch.
        feat_cache = self._build_feature_cache() if getattr(self.args, 'cache_train_features', 0) else None

        for epoch in range(self.args.num_epochs):
            print("======================TRAIN MODE======================")
            iter_count = 0
            loss_rec_list = []
            epoch_time = time.time()

            for model in self.models:
                model.train()

            if feat_cache is None:
                batch_iter = ((None, imgs) for (imgs, _, _, _, _) in self.train_loader)
            else:
                batch_iter = self._cached_batches(feat_cache)

            for i, (cached_feats, images) in enumerate(batch_iter):
                iter_count += 1
                if cached_feats is None:
                    images = images.float().to(self.args.device)
                    with torch.no_grad():
                        features = self.encoder(images)
                else:
                    features = cached_feats

                for fl in range(self.args.feature_levels):
                    input = self.avg_pool(features[fl])
                    input = F.normalize(input, p=2, dim=1)
                    N, D, _, _ = input.shape
                    input = input.permute(0, 2, 3, 1).reshape(N, -1, D)

                    model = self.models[fl]

                    # Get the model output.
                    output = model(input)

                    # Normalize the output and input before computing their distance.
                    output = F.normalize(output, p=2, dim=-1)

                    # Use cosine reconstruction loss.
                    loss = torch.mean(1 - self.cos_criterion(output, input))

                    loss_rec_list.append(loss.item())

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

            avg_loss = np.mean(loss_rec_list)
            speed = (time.time() - start_time) / iter_count
            left_time = speed * ((self.args.num_epochs - epoch) * train_steps - i)
            print("Epoch: {} cost time: {:.2f}s | avg_loss: {:.6f} | speed: {:.4f}s/iter | left time: {:.2f}s".format(
                epoch + 1, time.time() - epoch_time, avg_loss, speed, left_time))

            iter_count = 0
            start_time = time.time()

            # Evaluate the current epoch.
            img_auc, pix_auc, pro_auc = self.test(vis=False)
            last_metrics = (img_auc, pix_auc, pro_auc)

            print("Epoch: {0}, Class Name: {1}, Image AUC: {2:.7f} | Pixel AUC: {3:.7f} | PRO AUC: {4:.7f}".format(
                epoch + 1, self.args.class_name, img_auc, pix_auc, pro_auc))

            # Save the best metrics.
            if img_auc > best_img_auc:
                best_img_auc = img_auc
            if pix_auc > best_pix_auc:
                best_pix_auc = pix_auc
            if pro_auc > best_pro_auc:
                best_pro_auc = pro_auc
                if select == 'best_pro':
                    if not os.path.exists(path):
                        os.makedirs(path)
                    state = {'state_dict': [model.state_dict() for model in self.models]}
                    torch.save(state, os.path.join(path, self.args.class_name + '-pro.pth'))
            if select == 'last':
                state = {'state_dict': [model.state_dict() for model in self.models]}
                torch.save(state, os.path.join(path, self.args.class_name + '-last.pth'))

        if select == 'last':
            return last_metrics
        return best_img_auc, best_pix_auc, best_pro_auc

    def test(self, vis=False, checkpoint_path=None):
        if checkpoint_path is not None:
            ckpt_name = '-last.pth' if getattr(self.args, 'select', 'best_pro') == 'last' else '-pro.pth'
            ckpt_file = os.path.join(checkpoint_path, self.args.class_name + ckpt_name)
            print(f"[*] Loading checkpoint: {ckpt_file}")
            checkpoint = torch.load(ckpt_file)
            state_dict = checkpoint['state_dict']
            for i, model in enumerate(self.models):
                model.load_state_dict(state_dict[i])

        for model in self.models:
            model.eval()

        print("======================TEST MODE======================")

        scores_list = [list() for _ in range(self.args.feature_levels)]
        test_imgs, gt_label_list, gt_mask_list, file_names, img_types = [], [], [], [], []
        infer_times = []
        total_images = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        for i, (image, label, mask, file_name, img_type) in enumerate(self.test_loader):
            batch_size = image.shape[0]
            # only needed for the visualisation -> saves a blocking D2H copy per image
            if vis:
                test_imgs.append(image.cpu().numpy())
            gt_label_list.extend(label)
            gt_mask_list.extend(mask.numpy())
            file_names.extend(file_name)
            img_types.extend(img_type)

            image = image.float().to(self.args.device)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            with torch.no_grad():
                features = self.encoder(image)

                for fl in range(self.args.feature_levels):
                    input = self.avg_pool(features[fl])
                    input = F.normalize(input, p=2, dim=1)
                    N, D, _, _ = input.shape
                    input = input.permute(0, 2, 3, 1).reshape(N, -1, D)

                    model = self.models[fl]
                    output = model(input, train=False)

                    output = F.normalize(output, p=2, dim=-1)

                    score = 1 - self.cos_criterion(input, output)

                    score = score.reshape(score.shape[0], self.ws[fl], self.ws[fl])
                    score = F.interpolate(score.unsqueeze(1),
                        size=self.args.inp_size, mode='bilinear', align_corners=True).squeeze(1)
                    # keep score maps on CPU: they accumulate over the test set and
                    # would otherwise be counted by the peak-GPU-memory measurement
                    scores_list[fl].append(score.cpu())
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            batch_time = time.time() - start_time
            infer_times.append(batch_time / batch_size)
            total_images += batch_size
        if len(infer_times) > 1:
            avg_infer_time = np.mean(infer_times[1:])
        else:
            avg_infer_time = infer_times[0]
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        print(f"[*] Average inference time per image: {avg_infer_time * 1000:.2f} ms ({1/avg_infer_time:.2f} FPS) | Peak GPU memory: {peak_mem_mb:.1f} MB")
        view_layout = self.resolve_view_layout()
        self.timing.update({
            'infer_ms_per_img': round(avg_infer_time * 1000, 2),
            'fps': round(1 / avg_infer_time, 2),
            'peak_gpu_mem_MB': round(peak_mem_mb, 1),
            'n_views': view_layout.get('n_views', 1),
            'view_layout': view_layout,
        })
        ######################
        # All score operations below run on the GPU.
        ######################
        with torch.no_grad():
            lvl_scores = []
            for l in range(self.args.feature_levels):
                lvl_score = torch.cat(scores_list[l], dim=0)
                lvl_scores.append(lvl_score)

            scores = torch.zeros_like(lvl_scores[0])
            weights = self.args.fusion_weights
            for l in range(self.args.feature_levels):
                scores += weights[l] * lvl_scores[l]

            scores = tv_gaussian_blur(scores.unsqueeze(1), kernel_size=[7, 7], sigma=[1.5, 1.5]).squeeze(1)
            # scores = scores**2

            B = scores.shape[0]
            scores_flat = scores.reshape(B, -1)
            # image-level pooling from the (unchanged) pixel anomaly map.
            # NOTE: pooling only changes the image-level score; P-AUC / PRO are
            # computed from the same pixel anomaly map and must be identical
            # across pooling variants (see corrected Table 7).
            img_pool = getattr(self.args, 'img_pool', 'topk_mean')
            if img_pool == 'topk_mean':
                K = min(getattr(self.args, 'topk', 100), scores_flat.shape[1])
                topk_scores, _ = torch.topk(scores_flat, k=K, dim=1)
                img_scores = topk_scores.mean(dim=1)
            elif img_pool == 'max':
                img_scores = scores_flat.max(dim=1).values
            elif img_pool == 'mean':
                img_scores = scores_flat.mean(dim=1)
            else:
                raise ValueError('unknown img_pool: {}'.format(img_pool))
            img_scores = img_scores.cpu().numpy()

        metric_t0 = time.time()
        scores_np = scores.cpu().numpy()
        gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)
        pix_auc = roc_auc_score(gt_mask.flatten(), scores_np.flatten())
        pro_auc = compute_pro_retrieval_metrics(scores_np, gt_mask)

        gt_label = np.asarray(gt_label_list, dtype=bool)
        img_auc = roc_auc_score(gt_label, img_scores)
        print(f"[*] Metric calculation time (CPU): {time.time() - metric_t0:.2f}s")
        if vis:
            precision, recall, thresholds = precision_recall_curve(gt_label, img_scores)
            a = 2 * precision * recall
            b = precision + recall
            f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            img_threshold = thresholds[np.argmax(f1)]

            visulizer = Visualizer(f'vis_results/{self.args.save_prefix}/{self.args.class_name}')
            max_score = np.percentile(scores_np, 99)
            min_score = np.min(scores_np)
            scores_norm = np.clip((scores_np - min_score) / (max_score - min_score), 0, 1)
            test_imgs = np.concatenate(test_imgs, axis=0)
            visulizer.plot(test_imgs, scores_norm, img_scores, gt_mask, file_names, img_types, img_threshold)

        return img_auc, pix_auc, pro_auc
