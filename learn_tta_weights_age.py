import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_model
from dataset import FaceDataset
from test import generate_tta_versions 


class AugTTA(nn.Module):
    def __init__(self, n_augs, n_classes):
        super().__init__()
        self.n_augs = n_augs
        self.n_classes = n_classes
        self.weights = nn.Parameter(torch.ones(n_augs, dtype=torch.float))

    def forward(self, logits_stack):
        weighted_logits = (logits_stack * self.weights.view(1, -1, 1)).sum(dim=1)
        return weighted_logits

    def get_weights(self):
        return self.weights.detach().clone()

    def project_weights(self):
        with torch.no_grad():
            self.weights.clamp_(min=0.0)


def train_tta_weights_age(model, val_dataset, device,
                          n_augs=30, epochs=30, lr=0.01, output_dir='./tta_weights_learned'):

    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    n_classes = 101 

    tta_model = AugTTA(n_augs, n_classes)
    print(f"Using AugTTA: {n_augs} weights (one per augmentation)")
    tta_model.to(device)

    optimizer = optim.SGD(tta_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_maes = []

    print("\nPre-computing TTA logits for validation set...")
    val_tta_logits = []
    val_ages_gt = []

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    with torch.no_grad():
        for imgs, ages in tqdm(val_loader, desc="Computing TTA logits"):
            batch_size = imgs.size(0)
            tta_logits = torch.zeros(batch_size, n_augs, n_classes, device=device)

            for i in range(batch_size):
                img = imgs[i].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                tta_versions = generate_tta_versions(img)

                for aug_idx, aug_img in enumerate(tta_versions):
                    aug_tensor = torch.from_numpy(
                        np.transpose(aug_img.astype(np.float32), (2, 0, 1))
                    ).unsqueeze(0).to(device)

                    logits = model(aug_tensor)
                    tta_logits[i, aug_idx] = logits.squeeze(0)

            val_tta_logits.append(tta_logits.cpu())
            val_ages_gt.append(ages)

    val_tta_logits = torch.cat(val_tta_logits, dim=0)
    val_ages_gt = torch.cat([t.long() for t in val_ages_gt], dim=0)

    print("\nTraining TTA weights...")
    ages_tensor = torch.arange(0, 101, dtype=torch.float32)

    for epoch in range(epochs):
        tta_model.train()
        epoch_loss = 0.0
        epoch_mae = 0.0
        n_samples = 0

        batch_size = 32
        n_batches = (len(val_tta_logits) + batch_size - 1) // batch_size

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(val_tta_logits))

            batch_logits = val_tta_logits[start_idx:end_idx].to(device)
            batch_ages = val_ages_gt[start_idx:end_idx].to(device) 

            weighted_logits = tta_model(batch_logits)

            # Cross-entropy loss
            loss = criterion(weighted_logits, batch_ages)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tta_model.project_weights()

            with torch.no_grad():
                probs = torch.softmax(weighted_logits, dim=-1)
                pred_ages = (probs * ages_tensor.to(device)).sum(dim=-1)
                mae = torch.abs(pred_ages - batch_ages.float()).mean().item()
                epoch_mae += mae * (end_idx - start_idx)

            epoch_loss += loss.item() * (end_idx - start_idx)
            n_samples += (end_idx - start_idx)

            pbar.set_postfix({
                'ce_loss': f"{loss.item():.4f}",
                'mae': f"{mae:.4f}"
            })

        epoch_loss /= n_samples
        epoch_mae /= n_samples
        train_losses.append(epoch_loss)
        train_maes.append(epoch_mae)

        print(f"Epoch [{epoch+1}/{epochs}] - CE Loss: {epoch_loss:.4f}, MAE: {epoch_mae:.4f}")
        weights = tta_model.get_weights().cpu().numpy()
        n_zero = (weights == 0.0).sum()
        print(f"  Learned weights: {weights}  ({n_zero}/{n_augs} zeroed out)")

    weights_path = os.path.join(output_dir, 'tta_weights_aug.pth')
    torch.save({
        'weights': tta_model.get_weights().cpu(),
        'method': 'aug',
        'n_augs': n_augs,
        'n_classes': n_classes
    }, weights_path)
    print(f"\nTTA weights saved to {weights_path}")

    print(f"\nFinal TTA weights (AugTTA):")
    print(tta_model.get_weights().cpu().numpy())

    return tta_model


def get_args():
    parser = argparse.ArgumentParser(description="Learn TTA weights for age estimation")
    parser.add_argument("--data_dir", type=str, default="./appa-real-release",
                        help="Path to APPA-REAL dataset directory")
    parser.add_argument("--model_name", type=str, default="se_resnext50_32x4d",
                        help="Model architecture name")
    parser.add_argument("--resume", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--n_augs", type=int, default=30,
                        help="Number of TTA augmentations (must match generate_tta_versions output)")
    parser.add_argument("--output_dir", type=str, default="./tta_weights_learned",
                        help="Directory to save learned TTA weights")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"\nLoading model: {args.model_name}")
    model = get_model(args.model_name, num_classes=101, pretrained=None)

    print(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"\nLoading validation dataset from: {args.data_dir}")
    val_dataset = FaceDataset(args.data_dir, "valid", img_size=224, augment=False)
    print(f"Validation dataset size: {len(val_dataset)}")

    print(f"\n{'='*60}")
    print("Training TTA weights")
    print(f"{'='*60}\n")

    _ = train_tta_weights_age(
        model=model,
        val_dataset=val_dataset,
        device=device,
        n_augs=args.n_augs,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir
    )

    print(f"\n{'='*60}")
    print("TTA weight learning completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()