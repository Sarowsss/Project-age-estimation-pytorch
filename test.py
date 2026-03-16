import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pretrainedmodels
from model import get_model
from dataset import FaceDataset
from defaults import _C as cfg

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
    parser.add_argument("--weights", type=str, default=None, help="Path to learned TTA weights (.pth)")
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


def validate(validate_loader, model, device, dataset):
    model.eval()
    preds = []
    gt = []
    paths = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for x, y in _tqdm:
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)
                probs = torch.softmax(outputs, dim=-1).cpu().numpy()
                preds.append(probs)
                gt.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    ages = np.arange(0, 101)
    pred_ages = (preds * ages).sum(axis=-1)
    
    # Return image paths and predictions
    paths = dataset.x
    return pred_ages, paths


def apply_weighted_logits(logits_stack, weights=None):
    """
    logits_stack: [num_augs, num_classes]
    weights: None (simple average) or tensor [num_augs] (AugTTA)
    """
    if weights is None:
        weighted = logits_stack.mean(dim=0)
    elif weights.ndim == 1:
        # AugTTA: weights shape [num_augs] -> broadcast to [num_augs, 1]
        weighted = (logits_stack * weights.view(-1, 1)).sum(dim=0)
    else:
        raise ValueError(f"Unexpected weights shape: {weights.shape}")
    return weighted


def validate_with_tta(validate_loader, model, device, dataset, weights=None):
    """
    If weights is None, uses simple average. Otherwise uses learned weights.
    Returns predicted ages and image paths.
    """
    model.eval()
    all_pred_ages = []
    paths = []
    
    ages = np.arange(0, 101)

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for x, y in _tqdm:
                batch_size = x.size(0)
                for img_idx in range(batch_size):
                    single_img_tensor = x[img_idx].cpu().numpy()
                    single_img = np.transpose(single_img_tensor, (1, 2, 0)).astype(np.uint8)

                    aug_imgs = generate_tta_versions(single_img, target_size=cfg.MODEL.IMG_SIZE)

                    # Collect logits for all augmentations
                    logits_list = []
                    for aug_img in aug_imgs:
                        aug_img_float = aug_img.astype(np.float32)
                        tensor = torch.from_numpy(np.transpose(aug_img_float, (2, 0, 1))).unsqueeze(0).to(device)
                        logits = model(tensor)  # [1, num_classes]
                        logits_list.append(logits.squeeze(0))
                    logits_stack = torch.stack(logits_list, dim=0).to(device)  # [num_augs, num_classes]

                    # Apply weighting
                    weighted_logits = apply_weighted_logits(logits_stack, weights)
                    probs = torch.softmax(weighted_logits, dim=-1).cpu().numpy()
                    pred_age = (probs * ages).sum()
                    all_pred_ages.append(pred_age)

    all_pred_ages = np.array(all_pred_ages)
    paths = dataset.x
    
    return all_pred_ages, paths


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

    # load test dataset
    test_dataset = FaceDataset(args.data_dir, "test", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    # load weights if provided
    weights = None
    if args.weights:
        w = torch.load(args.weights, map_location='cpu')
        weights = w['weights'].float().to(device)
        method = w.get('method', 'aug')
        print(f"Loaded learned TTA weights from {args.weights} (method: {method})")
        print(f"Weights shape: {weights.shape}")

    print("=> start testing")
    if args.tta:
        pred_ages, paths = validate_with_tta(test_loader, model, device, test_dataset, weights=weights)
        file_name = "image_mean_tta.txt" if not args.weights else "image_mean_tta_weighted.txt"
    else:
        pred_ages, paths = validate(test_loader, model, device, test_dataset)
        file_name = "image_mean_non_tta.txt"

    # Save predictions to file
    out_dir = Path("./test_results")
    out_dir.mkdir(exist_ok=True)
    
    with open(out_dir / file_name, "w") as f:
        for path, pred in zip(paths, pred_ages):
            f.write(f"{path} {pred:.4f}\n")


if __name__ == '__main__':
    main()
