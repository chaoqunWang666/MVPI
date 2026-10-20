import argparse
import json
import os
import numpy as np
import torch
from torch.backends import cudnn
from utils.utils import *

from trainer import Trainer
from datasets.mvtec import MVTEC_CLASS_NAMES
from datasets.CAD import CAD_CLASS_NAMES
from datasets.btad import BTAD_CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser()
    # basic config
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--seed', type=int, default=3407)
    # dataset config
    parser.add_argument('--dataset', default='CAD', type=str, metavar='D',
                        help='dataset name: mvtec/cad/btad (default: btad)')
    parser.add_argument('--data_path', default='/path/to/dataset', type=str)
    parser.add_argument('--class_name', default='none', type=str, metavar='C',
                        help='class name (default: none)')
    parser.add_argument('--classes', default='all', type=str,
                        help='comma-separated class list, or "all" (default: all)')
    parser.add_argument('--inp_size', default=448, type=int,
                        help='image resize dimensions (default: 448)')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='train batch size (default: 1)')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--test_batch_size', default=4, type=int,
                        help='inference batch size. The network only uses LayerNorm, so '
                             'this does not change any metric (default: 4)')
    parser.add_argument('--prefetch_factor', default=4, type=int,
                        help='batches prefetched per dataloader worker (default: 4)')
    parser.add_argument('--cache_train_features', type=int, default=0, choices=[0, 1],
                        help='1 = run the frozen backbone on the train set ONCE and reuse '
                             'the features for every epoch (much faster; consumes host RAM '
                             'and changes the RNG stream, so results differ slightly). '
                             '0 = original behaviour (default: 0)')
    # backbone config
    parser.add_argument('--backbone_arch', default='dinov3_vitb16', type=str,
                        help='feature extractor (default: dinov3_vitb16)')
    parser.add_argument('--feature_levels', default=3, type=int)
    # layer selection (Reviewer 3-1: layer-choice ablation; Reviewer 1-3: consistency)
    parser.add_argument('--target_layers', default='2,5,8', type=str,
                        help='comma-separated 0-indexed ViT block indices (default: 2,5,8)')
    # MRFT switch: 1 = multi-scale 3-head reconstruction + weighted fusion (full MRFT),
    #              0 = single-scale reconstruction head on the deepest target layer
    parser.add_argument('--use_mrft', type=int, default=1, choices=[0, 1])
    parser.add_argument('--fusion_weights', default='0.4,0.3,0.3', type=str,
                        help='fixed per-scale fusion coefficients alpha_i (default: 0.4,0.3,0.3)')
    parser.add_argument('--jitter_scale', type=float, default=0.05,
                        help='feature jitter noise magnitude sigma (default: 0.05)')
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=3)
    # MVPI config (Reviewer 1-1/1-7: concrete MVPI settings and tiling baseline)
    parser.add_argument('--use_mvpi', type=int, default=1, choices=[0, 1],
                        help='1 = multi-view overlapping inference, 0 = single full-image view')
    parser.add_argument('--overlap_patch_size', default=224, type=int,
                        help='MVPI crop size in pixels (default: 224)')
    parser.add_argument('--overlap_size', default=64, type=int,
                        help='MVPI overlap size in pixels used to derive the number of views (default: 64)')
    parser.add_argument('--mvpi_agg', default='disjoint', type=str, choices=['disjoint', 'avg'],
                        help="disjoint = MVPI central-region assignment; avg = plain overlapping-tiling "
                             "baseline that averages all overlapping tokens (same crops / same #forward passes)")
    # image-level pooling (corrected Table 7: pooling must not change pixel-level metrics)
    parser.add_argument('--img_pool', default='topk_mean', type=str,
                        choices=['topk_mean', 'max', 'mean'],
                        help='image-level score pooling from the pixel anomaly map (default: topk_mean)')
    parser.add_argument('--topk', type=int, default=100)
    # checkpoint selection protocol (Reviewer 1-2: train/validation procedure)
    parser.add_argument('--select', default='best_pro', type=str, choices=['best_pro', 'last'],
                        help="model selection: best PRO on test set per epoch (original protocol) or last epoch")
    # output
    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--save_prefix', default='btad', type=str)
    parser.add_argument('--checkpoint', default='checkpoints/btad', type=str)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--out_json', default='', type=str,
                        help='write per-run result JSON (config + metrics + timing) to this path')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.img_size = (args.inp_size, args.inp_size)
    args.crop_size = (args.inp_size, args.inp_size)
    args.norm_mean, args.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # parse layer / weight lists
    args.target_layers = [int(x) for x in args.target_layers.split(',') if x.strip() != '']
    args.fusion_weights = [float(x) for x in args.fusion_weights.split(',') if x.strip() != '']
    # single-scale ablation: MRFT off -> keep only the deepest layer, weight 1.0
    if not args.use_mrft:
        args.target_layers = [args.target_layers[-1]]
        args.fusion_weights = [1.0]
    assert len(args.fusion_weights) == len(args.target_layers), \
        'fusion_weights must match target_layers'
    return args


def run_one_class(args, class_name):
    args.class_name = class_name
    cudnn.benchmark = True
    init_seeds(args.seed)

    trainer = Trainer(args)

    if args.mode == 'train':
        img_auc, pix_auc, pro_auc = trainer.train()
    else:
        img_auc, pix_auc, pro_auc = trainer.test(vis=args.vis, checkpoint_path=args.checkpoint)
    print("Class Name: {}".format(args.class_name))
    print('Image AUC: {}'.format(img_auc))
    print('Pixel AUC: {}'.format(pix_auc))
    print('PRO AUC: {}'.format(pro_auc))

    record = {
        'dataset': args.dataset,
        'class_name': args.class_name,
        'mode': args.mode,
        'seed': args.seed,
        'img_auc': float(img_auc),
        'pix_auc': float(pix_auc),
        'pro_auc': float(pro_auc),
        # full config snapshot for reproducibility (Reviewer 1-1 / 1-2)
        'config': {
            'backbone': args.backbone_arch,
            'inp_size': args.inp_size,
            'target_layers': args.target_layers,
            'use_mrft': args.use_mrft,
            'fusion_weights': args.fusion_weights,
            'jitter_scale': args.jitter_scale,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'use_mvpi': args.use_mvpi,
            'overlap_patch_size': args.overlap_patch_size,
            'overlap_size': args.overlap_size,
            'mvpi_agg': args.mvpi_agg,
            'img_pool': args.img_pool,
            'topk': args.topk,
            'select': args.select,
            'lr': args.lr,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'optimizer': 'Adam',
        },
        'timing': getattr(trainer, 'timing', {}),
    }
    if args.out_json:
        out_path = args.out_json
        if '{class}' in out_path:
            out_path = out_path.replace('{class}', class_name)
        else:
            root, ext = os.path.splitext(out_path)
            out_path = '{}__{}{}'.format(root, class_name, ext or '.json')
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(record, f, indent=2)
        print('[*] Result JSON saved to {}'.format(out_path))
    return record


def main():
    args = parse_args()

    args_dict = vars(args).copy()
    args_dict['device'] = str(args.device)
    print('------------ Options -------------')
    for k, v in sorted(args_dict.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    if args.dataset.lower() == 'mvtec':
        CLASS_NAMES = MVTEC_CLASS_NAMES
    elif args.dataset.lower() == 'cad':
        CLASS_NAMES = CAD_CLASS_NAMES
    elif args.dataset.lower() == 'btad':
        CLASS_NAMES = BTAD_CLASS_NAMES
    else:
        raise ValueError("Not recognized dataset: {}!".format(args.dataset))

    if args.classes.strip().lower() != 'all':
        keep = [c.strip() for c in args.classes.split(',') if c.strip()]
        CLASS_NAMES = [c for c in CLASS_NAMES if c in keep]
        missing = [c for c in keep if c not in CLASS_NAMES]
        if missing:
            raise ValueError('Unknown classes for {}: {}'.format(args.dataset, missing))

    records = []
    for class_name in CLASS_NAMES:
        record = run_one_class(args, class_name)
        records.append(record)

    img_aucs = [r['img_auc'] for r in records]
    pix_aucs = [r['pix_auc'] for r in records]
    pro_aucs = [r['pro_auc'] for r in records]
    for r in records:
        print('{}: Image-AUC: {:.6f}, Pixel-AUC: {:.6f}, PRO-AUC: {:.6f}'.format(
            r['class_name'], r['img_auc'], r['pix_auc'], r['pro_auc']))
    print('Mean Image-AUC: {}'.format(np.mean(img_aucs)))
    print('Mean Pixel-AUC: {}'.format(np.mean(pix_aucs)))
    print('Mean PRO-AUC: {}'.format(np.mean(pro_aucs)))


if __name__ == '__main__':
    main()
