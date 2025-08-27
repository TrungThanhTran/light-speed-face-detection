import torch
from mmcv import Config
from mmdet.models import build_detector

def reparam_module(m, scopes=('backbone',)):
    for name, child in m.named_children():
        # recurse only into allowed scopes
        if any(name.startswith(s) or name == s for s in scopes):
            reparam_module(child, scopes)
            if hasattr(child, 'reparam') and callable(child.reparam):
                child.reparam()

def main(cfg_path, ckpt_path, out_path):
    cfg = Config.fromfile(cfg_path)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # fuse all rep-enabled modules
    reparam_module(model)

    ckpt = {
        'state_dict': model.state_dict(),
        'meta': {'CLASSES': ('face',), 'deploy': True}
    }
    torch.save(ckpt, out_path)
    print("Saved fused checkpoint to", out_path)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    main(args.cfg, args.ckpt, args.out)
