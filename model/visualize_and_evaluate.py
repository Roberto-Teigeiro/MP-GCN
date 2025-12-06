#!/usr/bin/env python3
"""
Generate training plots and evaluate a trained model on the validation set.

Saves:
 - history plots: `metrics/training_history.png`
 - evaluation JSON: `metrics/evaluation.json`
 - confusion matrix image: `metrics/confusion_matrix.png`

Usage:
  python model/visualize_and_evaluate.py --checkpoint ../checkpoints/best_model.pth
"""

import os
import json
import argparse
from pathlib import Path

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

from model.config import ModelConfig
from model.dataset import create_data_loaders
from model.trainer import evaluate_model
from model.networks import PlaygroundGCN


def load_checkpoint(checkpoint_path, device):
    # Use weights_only=False to allow loading full checkpoint dict (trusted local file)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return ckpt


def build_model_from_ckpt(ckpt, device):
    cfg = ckpt.get('config', {})
    state = ckpt['model_state_dict']

    # Determine num_nodes from saved adjacency buffer if present
    if 'A' in state:
        A_tensor = state['A']
        if isinstance(A_tensor, torch.Tensor):
            num_nodes = A_tensor.shape[-1]
        else:
            num_nodes = np.array(A_tensor).shape[-1]
    else:
        # Fallback
        num_nodes = cfg.get('num_keypoints', 17)

    model = PlaygroundGCN(
        num_classes=cfg.get('num_classes', 7),
        num_persons=cfg.get('max_persons', 6),
        num_joints=num_nodes,
        num_objects=0,
        in_channels=cfg.get('input_channels', 2),
        base_channels=cfg.get('base_channels', 64),
        num_streams=cfg.get('num_streams', 2),
        dropout=cfg.get('dropout', 0.3),
        use_attention=cfg.get('use_attention', True)
    )

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg


def plot_history(history_path: Path, out_dir: Path):
    if not history_path.exists():
        print(f"History file not found: {history_path}")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    axes[0, 0].plot(history.get('train_loss', []), label='train_loss')
    axes[0, 0].plot(history.get('val_loss', []), label='val_loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()

    # Accuracy
    axes[0, 1].plot(history.get('train_acc', []), label='train_acc')
    axes[0, 1].plot(history.get('val_acc', []), label='val_acc')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()

    # F1
    axes[1, 0].plot(history.get('val_f1', []), label='val_f1')
    axes[1, 0].set_title('Val F1')
    axes[1, 0].legend()

    # LR
    axes[1, 1].plot(history.get('lr', []), label='lr')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].legend()

    plt.tight_layout()
    out_file = out_dir / 'training_history.png'
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved training history plot to {out_file}")


def eval_and_save(model, cfg, device, out_dir: Path):
    # Build ModelConfig object for data loaders
    mc = ModelConfig()
    # Update mc from cfg dict
    for k, v in cfg.items():
        if hasattr(mc, k):
            try:
                setattr(mc, k, v)
            except Exception:
                pass

    # Create data loaders (use small batch for evaluation)
    val_loader = create_data_loaders(
        poses_root='runs/pose_batch',
        annotations_path='annotationsv2.json',
        objects_yaml='objects.yaml',
        config=mc,
        batch_size=mc.batch_size,
    )[1]

    # Custom evaluation loop to avoid classification_report class mismatch
    all_preds = []
    all_labels = []
    all_names = []

    for data, labels, names in val_loader:
        # Handle multi-stream vs single-stream
        if data.dim() == 6 and not getattr(model, 'num_streams', None):
            data = data[:, 0]
        data = data.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits, _ = model(data)
            preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_names.extend(names)

    num_classes = int(cfg.get('num_classes', max(all_labels) + 1))
    class_names = [f'class_{i}' for i in range(num_classes)]
    # If cfg contained mapping, include it
    try:
        from model.config import LABEL_TO_ACTIVITY
        class_names = [LABEL_TO_ACTIVITY.get(i, f'class_{i}') for i in range(num_classes)]
    except Exception:
        pass

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))

    from sklearn.metrics import f1_score
    f1_weighted = float(f1_score(all_labels, all_preds, average='weighted'))
    f1_macro = float(f1_score(all_labels, all_preds, average='macro'))

    report = classification_report(all_labels, all_preds, labels=list(range(num_classes)), target_names=class_names, output_dict=True)

    results = {
        'accuracy': acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': list(zip(all_names, all_preds, all_labels))
    }

    # Save results JSON
    out_json = out_dir / 'evaluation.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation JSON to {out_json}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    out_cm = out_dir / 'confusion_matrix.png'
    fig.savefig(out_cm)
    plt.close(fig)
    print(f"Saved confusion matrix to {out_cm}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/best_model.pth')
    parser.add_argument('--history', type=str, default='../checkpoints/history.json')
    parser.add_argument('--out_dir', type=str, default='../deliverables/metrics')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    ckpt = load_checkpoint(str(ckpt_path), device)
    model, cfg = build_model_from_ckpt(ckpt, device)

    # Plot history
    plot_history(Path(args.history), out_dir)

    # Evaluate and save
    eval_and_save(model, cfg, device, out_dir)


if __name__ == '__main__':
    main()
