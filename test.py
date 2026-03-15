import argparse
import better_exceptions
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import pretrainedmodels
import pretrainedmodels.utils
from model import get_model
from dataset import FaceDataset
from defaults import _C as cfg
from train import validate


def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
    parser.add_argument("--resume", type=str, required=True, help="Model weight to be tested")
    parser.add_argument("--tta", action='store_true', help="Enable standard TTA (30 versions: 2 flips × 5 crops × 3 scales)")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


def generate_tta_versions(img, target_size=224, scales=[1.0, 1.04, 1.10]):
    versions = []
    h, w = img.shape[:2]
    
    for scale in scales:
        scaled_h, scaled_w = int(h * scale), int(w * scale)
        scaled_img = cv2.resize(img, (scaled_w, scaled_h))
        
        crop_positions = []
        top = (scaled_h - target_size) // 2
        left = (scaled_w - target_size) // 2
        crop_positions.append((top, left))
        
        crop_positions.append((0, 0))
        crop_positions.append((0, max(0, scaled_w - target_size)))
        crop_positions.append((max(0, scaled_h - target_size), 0))
        crop_positions.append((max(0, scaled_h - target_size), max(0, scaled_w - target_size)))
        
        for top, left in crop_positions:
            top = max(0, min(top, scaled_h - target_size))
            left = max(0, min(left, scaled_w - target_size))
            crop = scaled_img[top:top+target_size, left:left+target_size]
            
            if crop.shape[0] < target_size or crop.shape[1] < target_size:
                crop = cv2.resize(crop, (target_size, target_size))
            
            versions.append(crop.copy())
            flipped = cv2.flip(crop, 1)
            versions.append(flipped)
    
    return versions


def validate_with_tta(validate_loader, model, device):
    model.eval()
    preds = []
    gt = []
    
    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for x, y in _tqdm:
                batch_size = x.size(0)
                
                for img_idx in range(batch_size):
                    # Get single image: x is (batch, C, H, W), convert to (H, W, C) BGR
                    single_img_tensor = x[img_idx].cpu().numpy()  # (C, H, W)
                    single_img = np.transpose(single_img_tensor, (1, 2, 0)).astype(np.uint8)  # (H, W, C)
                    single_y = y[img_idx].item()
                    
                    # Generate 30 TTA versions from this image
                    aug_imgs = generate_tta_versions(single_img, target_size=cfg.MODEL.IMG_SIZE)
                    
                    # Run inference on all versions and collect probabilities
                    tta_probs = []
                    for aug_img in aug_imgs:
                        aug_img_float = aug_img.astype(np.float32)
                        tensor = torch.from_numpy(np.transpose(aug_img_float, (2, 0, 1))).unsqueeze(0).to(device)
                        outputs = model(tensor)
                        probs = torch.softmax(outputs, dim=-1).cpu().numpy()[0]
                        tta_probs.append(probs)
                    
                    # Average probabilities
                    avg_probs = np.mean(tta_probs, axis=0)
                    preds.append(avg_probs)
                    gt.append(single_y)
                
                _tqdm.update(1)
    
    preds = np.array(preds)
    gt = np.array(gt)
    ages = np.arange(0, 101)
    ave_preds = (preds * ages).sum(axis=-1)
    diff = ave_preds - gt
    mae = np.abs(diff).mean()
    
    return mae


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # load checkpoint
    resume_path = args.resume

    if Path(resume_path).is_file():
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(resume_path))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

    if device == "cuda":
        cudnn.benchmark = True

    test_dataset = FaceDataset(args.data_dir, "test", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    print("=> start testing")
    if args.tta:
        print("=> using test-time augmentation (30 versions per image)")
        test_mae = validate_with_tta(test_loader, model, device)
    else:
        _, _, test_mae = validate(test_loader, model, None, 0, device)
    print(f"test mae: {test_mae:.3f}")


if __name__ == '__main__':
    main()
