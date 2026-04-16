"""Extract backbone+neck state_dict from a full Cascade RCNN checkpoint.

After training Cascade RCNN on COCO-UWNR, the bbox_head has 80 classes.
We cannot directly load this checkpoint into a 10-class RUOD model.
Instead, extract backbone/neck/rpn weights and save as a small checkpoint
suitable for `init_cfg.Pretrained`.

Usage (from my_diff/):
    python tools/extract_backbone.py \
        --checkpoint work_dirs/cascade_r50_coco_uwnr/epoch_24.pth \
        --output work_dirs/cascade_r50_coco_uwnr/backbone_only.pth
"""
import argparse
import torch


# Keys that are safe to transfer (class-count invariant)
SAFE_PREFIXES = [
    'backbone.',
    'neck.',
    'rpn_head.',
]

# Keys that depend on num_classes (must NOT be transferred)
HEAD_PREFIXES = [
    'roi_head.bbox_head.',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Full model checkpoint')
    parser.add_argument('--output', required=True, help='Output path for extracted weights')
    args = parser.parse_args()

    print(f'Loading {args.checkpoint} ...')
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    state_dict = ckpt.get('state_dict', ckpt)
    print(f'  Total keys: {len(state_dict)}')

    extracted = {}
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in SAFE_PREFIXES):
            extracted[k] = v

    print(f'  Extracted keys: {len(extracted)}')

    out_ckpt = {
        'state_dict': extracted,
        'meta': ckpt.get('meta', {}),
    }

    torch.save(out_ckpt, args.output)
    print(f'Saved to {args.output}')
    print('Use with model.init_cfg = dict(type="Pretrained", checkpoint="{args.output}")')


if __name__ == '__main__':
    main()
