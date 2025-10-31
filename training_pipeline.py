"""
Complete Training Pipeline for CNN Models
Supports all implemented architectures with logging and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import os
import json
import argparse
from tqdm import tqdm
import time
from typing import Dict, Optional, Tuple
import numpy as np

# Import all model factories
from lenet import create_lenet
from alexnet import create_alexnet
from vggnet import create_vgg16
from googlenet import create_googlenet
from resnet import create_resnet50, create_resnet18
from mobilenet import create_mobilenet
from densenet import create_densenet121
from efficientnet import create_efficientnet_b0
from inceptionv3 import create_inceptionv3
from vit import create_vit_base, create_vit_small


class CNNTrainer:
    """Universal trainer for all CNN models"""
    
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs'
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # TensorBoard writer
        self.writer = SummaryWriter(os.path.join(log_dir, model_name))
        
        # Training stats
        self.best_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def _create_model(self) -> nn.Module:
        """Create model based on name"""
        models = {
            'lenet5': lambda: create_lenet(self.num_classes, input_channels=3),
            'alexnet': lambda: create_alexnet(self.num_classes),
            'vgg16': lambda: create_vgg16(self.num_classes),
            'googlenet': lambda: create_googlenet(self.num_classes),
            'resnet50': lambda: create_resnet50(self.num_classes),
            'resnet18': lambda: create_resnet18(self.num_classes),
            'mobilenet': lambda: create_mobilenet(self.num_classes),
            'densenet121': lambda: create_densenet121(self.num_classes),
            'efficientnet_b0': lambda: create_efficientnet_b0(self.num_classes),
            'inceptionv3': lambda: create_inceptionv3(self.num_classes),
            'vit_base': lambda: create_vit_base(num_classes=self.num_classes),
            'vit_small': lambda: create_vit_small(num_classes=self.num_classes),
        }
        
        if self.model_name not in models:
            raise ValueError(f"Model {self.model_name} not supported. Choose from {list(models.keys())}")
        
        return models[self.model_name]()
    
    def get_data_loaders(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 224
    ) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation data loaders"""
        
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transform (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            transform=train_transform
        )
        
        val_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'val'),
            transform=val_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        epoch: int
    ) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(
        self,
        epoch: int,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        is_best: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': self.best_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'{self.model_name}_latest.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.checkpoint_dir,
                f'{self.model_name}_best.pth'
            )
            torch.save(checkpoint, best_path)
            print(f'💾 Best model saved with accuracy: {self.best_acc:.2f}%')
    
    def load_checkpoint(self, checkpoint_path: str, optimizer: optim.Optimizer = None):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.best_acc = checkpoint.get('best_acc', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_accs = checkpoint.get('val_accs', [])
        
        print(f'✅ Checkpoint loaded from {checkpoint_path}')
        return checkpoint.get('epoch', 0)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        lr: float = 0.001,
        weight_decay: float = 5e-4,
        scheduler_type: str = 'cosine',
        early_stopping_patience: int = 10,
        resume: bool = False
    ):
        """Complete training pipeline"""
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == 'reduce':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
        else:
            scheduler = None
        
        # Resume from checkpoint if requested
        start_epoch = 0
        if resume:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_latest.pth')
            if os.path.exists(checkpoint_path):
                start_epoch = self.load_checkpoint(checkpoint_path, optimizer)
        
        # Early stopping
        patience_counter = 0
        
        print(f"\n{'='*80}")
        print(f"Training {self.model_name}")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning Rate: {lr}")
        print(f"Scheduler: {scheduler_type}")
        print(f"{'='*80}\n")
        
        # Training loop
        for epoch in range(start_epoch + 1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion, epoch)
            
            # Update learning rate
            if scheduler is not None:
                if scheduler_type == 'reduce':
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
            
            # Store statistics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            # Check if best model
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, optimizer, scheduler, is_best)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f'\nEpoch {epoch}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            print(f'  Best Val Acc: {self.best_acc:.2f}%')
            print(f'  Time: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f'\n⚠️  Early stopping triggered after {epoch} epochs')
                break
        
        print(f'\n{"="*80}')
        print(f'Training completed!')
        print(f'Best validation accuracy: {self.best_acc:.2f}%')
        print(f'{"="*80}\n')
        
        self.writer.close()
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_acc': self.best_acc
        }
        
        history_path = os.path.join(self.log_dir, f'{self.model_name}_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)


def main():
    """Main training function with CLI"""
    parser = argparse.ArgumentParser(description='Train CNN Models')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                       choices=['lenet5', 'alexnet', 'vgg16', 'googlenet', 
                               'resnet18', 'resnet50', 'mobilenet', 'densenet121',
                               'efficientnet_b0', 'inceptionv3', 'vit_base', 'vit_small'],
                       help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'reduce', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CNNTrainer(
        model_name=args.model,
        num_classes=args.num_classes,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Get data loaders
    train_loader, val_loader = trainer.get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        early_stopping_patience=args.patience,
        resume=args.resume
    )


if __name__ == '__main__':
    main()