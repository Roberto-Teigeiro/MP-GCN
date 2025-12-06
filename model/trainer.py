"""
Training Pipeline for Playground Activity Recognition Model
Includes:
- Training loop with validation
- Learning rate scheduling with warmup
- Label smoothing and mixup regularization
- Checkpoint saving and early stopping
- Metrics logging
"""
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from .config import ACTIVITY_LABELS, LABEL_TO_ACTIVITY, ModelConfig, DEFAULT_CONFIG
from .networks import PlaygroundGCN, PlaygroundGCNLite, create_model
from .panoramic_graph import create_panoramic_graph
from .dataset import create_data_loaders


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        
        # Create smooth labels
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        
        log_probs = torch.log_softmax(pred, dim=-1)
        loss = (-smooth_target * log_probs).sum(dim=-1).mean()
        
        return loss


def mixup_data(x: torch.Tensor, y: torch.Tensor, 
               alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""
    
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
    
    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class Trainer:
    """Training manager for playground activity recognition."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: ModelConfig = None,
                 device: str = 'cuda',
                 save_dir: str = './checkpoints',
                 class_weights: Optional[torch.Tensor] = None,
                 multi_stream: bool = False):
        
        self.config = config or DEFAULT_CONFIG
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Whether to use multi-stream input (for full model) or single stream (lite)
        self.multi_stream = multi_stream
        
        # Model
        self.model = model.to(self.device)
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function
        if self.config.label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(self.config.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device) if class_weights is not None else None
            )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=self.config.warmup_epochs,
            total_epochs=self.config.epochs,
            base_lr=self.config.learning_rate
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        self.use_amp = torch.cuda.is_available()
        
        # Tracking
        self.best_acc = 0.0
        self.best_f1 = 0.0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'lr': []
        }
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, labels, names) in enumerate(self.train_loader):
            # Handle multi-stream input: (N, I, C, T, V, M)
            # For lite model, use only the first stream (joints)
            # For full model, keep all streams
            if data.dim() == 6 and not self.multi_stream:
                data = data[:, 0]  # (N, C, T, V, M)
            
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            # Apply mixup
            if self.config.mixup_alpha > 0:
                data, labels_a, labels_b, lam = mixup_data(
                    data, labels, self.config.mixup_alpha
                )
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                logits, _ = self.model(data)
                
                if self.config.mixup_alpha > 0:
                    loss = mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(logits, labels)
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            if self.config.mixup_alpha > 0:
                correct += (lam * (preds == labels_a).float() + 
                           (1 - lam) * (preds == labels_b).float()).sum().item()
            else:
                correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float, np.ndarray]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for data, labels, names in self.val_loader:
            # Handle multi-stream input
            # For lite model, use only first stream; for full model, keep all
            if data.dim() == 6 and not self.multi_stream:
                data = data[:, 0]  # (N, C, T, V, M)
            
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            with autocast(enabled=self.use_amp):
                logits, _ = self.model(data)
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        return avg_loss, accuracy, f1, conf_matrix
    
    def train(self, epochs: Optional[int] = None, early_stop_patience: int = 10):
        """Full training loop."""
        epochs = epochs or self.config.epochs
        no_improve = 0
        
        print(f"Starting training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 60)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Update learning rate
            lr = self.scheduler.step(epoch)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_f1, conf_matrix = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(lr)
            
            # Check for improvement
            improved = False
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                improved = True
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                improved = True
                self.save_checkpoint('best_model.pth', epoch, val_acc, val_f1)
            
            if improved:
                no_improve = 0
            else:
                no_improve += 1
            
            # Logging
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) | "
                  f"LR: {lr:.2e} | "
                  f"Train: {train_loss:.4f} / {train_acc:.4f} | "
                  f"Val: {val_loss:.4f} / {val_acc:.4f} / F1: {val_f1:.4f} "
                  f"{'*' if improved else ''}")
            
            # Early stopping
            if no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch+1}.pth', epoch, val_acc, val_f1)
        
        # Final save
        self.save_checkpoint('final_model.pth', epochs - 1, val_acc, val_f1)
        self.save_history()
        
        print("-" * 60)
        print(f"Training complete. Best accuracy: {self.best_acc:.4f}, Best F1: {self.best_f1:.4f}")
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, accuracy: float, f1: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'f1': f1,
            'config': vars(self.config),
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def save_history(self):
        """Save training history."""
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


def train_model(
    poses_root: str,
    annotations_path: str,
    objects_yaml: str,
    save_dir: str = './checkpoints',
    config: Optional[ModelConfig] = None,
    num_objects: int = 0,
    lite: bool = False
) -> Trainer:
    """
    Main training function.
    
    Args:
        poses_root: Path to poses directory (runs/pose_batch)
        annotations_path: Path to annotations JSON
        objects_yaml: Path to objects.yaml
        save_dir: Directory to save checkpoints
        config: Model configuration
        num_objects: Number of object nodes to include
        lite: Use lightweight model (faster training) or full model (more capacity)
    """
    config = config or DEFAULT_CONFIG
    
    # Create data loaders (use streams configured in ModelConfig)
    streams = getattr(config, 'streams', 'JBVM')
    train_loader, val_loader = create_data_loaders(
        poses_root=poses_root,
        annotations_path=annotations_path,
        objects_yaml=objects_yaml,
        config=config,
        batch_size=config.batch_size,
        streams=streams
    )
    
    # Get actual number of nodes from first batch
    for batch in train_loader:
        data, _, _ = batch
        # data shape: (N, I, C, T, V, M) for multi-stream
        if data.dim() == 6:
            num_nodes = data.shape[4]
        else:
            num_nodes = data.shape[3]
        break
    
    print(f"Detected {num_nodes} nodes (17 keypoints + {num_nodes - 17} objects)")
    
    # Get class weights for imbalanced data
    class_weights = train_loader.dataset.get_class_weights()
    
    # Create model
    if lite:
        print("Creating PlaygroundGCNLite model...")
        model = PlaygroundGCNLite(
            num_classes=config.num_classes,
            num_persons=config.max_persons,
            num_joints=num_nodes,  # Use actual num_nodes from data
            num_objects=0,  # Already included in num_joints
            in_channels=config.input_channels,
            base_channels=config.base_channels,
            dropout=config.dropout
        )
    else:
        print("Creating full PlaygroundGCN model...")
        # Full model uses per-person processing with multi-stream fusion
        model = PlaygroundGCN(
            num_classes=config.num_classes,
            num_persons=config.max_persons,
            num_joints=num_nodes,  # Use actual num_nodes (joints + objects)
            num_objects=0,  # Already included in num_joints
            in_channels=config.input_channels,
            base_channels=config.base_channels,
            num_streams=getattr(config, 'num_streams', 2),
            dropout=config.dropout,
            use_attention=config.use_attention
        )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_dir=save_dir,
        class_weights=class_weights,
        multi_stream=not lite  # Full model uses multi-stream, lite uses single stream
    )
    
    # Train
    trainer.train()
    
    return trainer


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda'
) -> Dict:
    """Evaluate model and return detailed metrics."""
    model.eval()
    device = device if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    all_names = []
    
    for data, labels, names in data_loader:
        data = data.to(device)
        logits, _ = model(data)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_names.extend(names)
    
    # Compute metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Per-class report
    class_names = [LABEL_TO_ACTIVITY.get(i, f'class_{i}') for i in range(len(ACTIVITY_LABELS))]
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': report,
        'predictions': list(zip(all_names, all_preds, all_labels))
    }
