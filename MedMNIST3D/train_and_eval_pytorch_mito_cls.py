#train_and_eval_pytorch_mito_cls.py
import argparse
import os
import time
from collections import OrderedDict, Counter
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from acsconv.converters import ACSConverter, Conv2_5dConverter, Conv3dConverter
from models import ResNet18, ResNet50
from tensorboardX import SummaryWriter
from tqdm import trange
from utils import Transform3D, model_to_syncbn
from mito_dataset import MitoPatchDataset
from sklearn.metrics import roc_auc_score, accuracy_score


def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced dataset"""
    labels = []
    for _, label in dataset:
        labels.append(label.item())
    
    counter = Counter(labels)
    total = len(labels)
    num_classes = len(counter)
    
    # Inverse frequency weighting
    weights = {}
    for class_id in counter:
        weights[class_id] = total / (num_classes * counter[class_id])
    
    return torch.FloatTensor([weights[i] for i in sorted(weights.keys())])


def train(model, train_loader, criterion, optimizer, device, writer):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        # targets = torch.squeeze(targets, 1).long().to(device)
        targets = targets.long().to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / num_batches


def test(model, data_loader, criterion, device, run, save_folder=None):
    """Evaluate model on given dataset"""
    model.eval()
    total_loss = 0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            # targets = torch.squeeze(targets, 1).long().to(device)
            targets = targets.long().to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Get probabilities for positive class
            probs = torch.softmax(outputs, dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(probs[:, 1].detach().cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    
    # Handle case where only one class is present
    unique_targets = np.unique(all_targets)
    if len(unique_targets) < 2:
        print(f"Warning: Only one class present in evaluation: {unique_targets}")
        auc = 0.5
    else:
        auc = roc_auc_score(all_targets, all_predictions)
    
    # Calculate accuracy using 0.5 threshold
    pred_labels = (np.array(all_predictions) > 0.5).astype(int)
    accuracy = accuracy_score(all_targets, pred_labels)
    
    # Debug information for small datasets
    if len(all_targets) <= 50:
        print(f"Debug info - Targets: {np.bincount(all_targets)}")
        print(f"Prediction range: [{min(all_predictions):.4f}, {max(all_predictions):.4f}]")
    
    return [avg_loss, auc, accuracy]


def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, conv, pretrained_3d, download, model_flag, as_rgb, shape_transform, model_path, run):
    """Main training function with original signature"""
    
    # Training hyperparameters
    lr = 0.001  # Increased from 0.00001
    weight_decay = 1e-4  # Reduced from 1e-3
    patience = 15  # Early stopping patience
    
    # Model configuration
    n_channels = 1
    n_classes = 2
    
    # Setup device
    str_ids = gpu_ids.split(',')
    gpu_ids_list = [int(str_id) for str_id in str_ids if int(str_id) >= 0]
    
    if len(gpu_ids_list) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids_list[0])
        device = torch.device(f'cuda:{gpu_ids_list[0]}')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Setup output directory
    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    os.makedirs(output_root, exist_ok=True)
    
    print('==> Preparing data...')
    
    # Data transforms
    train_transform = Transform3D(mul='random') if shape_transform else Transform3D()
    eval_transform = Transform3D(mul='0.5') if shape_transform else Transform3D()
    
    # Create datasets
    train_dataset = MitoPatchDataset(
        csv_path='mito_train.csv',
        data_dir='mito_downloaded_patch/',
        transform=train_transform
    )


    # Insert this debug check
    from collections import Counter
    raw_labels = [label for _, label in train_dataset.samples]
    print("Raw label distribution from CSV (before mapping):", Counter(raw_labels))

    mapped_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    print("Mapped label distribution (after __getitem__):", Counter(mapped_labels))


    val_dataset = MitoPatchDataset(
        csv_path='mito_val.csv',
        data_dir='mito_downloaded_patch/',
        transform=eval_transform
    )
    test_dataset = MitoPatchDataset(
        csv_path='mito_test.csv',
        data_dir='mito_downloaded_patch/',
        transform=eval_transform
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # For evaluation without data augmentation
    train_loader_at_eval = data.DataLoader(
        dataset=MitoPatchDataset(
            csv_path='mito_train.csv',
            data_dir='mito_downloaded_patch/',
            transform=eval_transform
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print('==> Building model...')
    
    # Create model
    if model_flag == 'resnet18':
        model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        model = ResNet50(in_channels=n_channels, num_classes=n_classes)
    else:
        raise ValueError(f"Unsupported model: {model_flag}")
    
    # Apply convolution converters
    if conv == 'ACSConv':
        model = model_to_syncbn(ACSConverter(model))
    elif conv == 'Conv2_5d':
        model = model_to_syncbn(Conv2_5dConverter(model))
    elif conv == 'Conv3d':
        if pretrained_3d == 'i3d':
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))
        else:
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=None))
    
    model = model.to(device)
    print(f"Model created with {conv} conversion")
    
    # Calculate class weights for imbalanced dataset
    try:
        class_weights = calculate_class_weights(train_dataset)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
    except Exception as e:
        print(f"Failed to calculate class weights, using standard CrossEntropyLoss: {e}")
        criterion = nn.CrossEntropyLoss()
    
    # Load pretrained model if specified
    if model_path is not None:
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if 'net' in checkpoint:
            model.load_state_dict(checkpoint['net'], strict=True)
        else:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
        # Evaluate loaded model
        print("Evaluating loaded model...")
        train_metrics = test(model, train_loader_at_eval, criterion, device, run, output_root)
        val_metrics = test(model, val_loader, criterion, device, run, output_root)
        test_metrics = test(model, test_loader, criterion, device, run, output_root)
        
        print(f'Loaded model performance:')
        print(f'Train - Loss: {train_metrics[0]:.5f}, AUC: {train_metrics[1]:.5f}, Acc: {train_metrics[2]:.5f}')
        print(f'Val   - Loss: {val_metrics[0]:.5f}, AUC: {val_metrics[1]:.5f}, Acc: {val_metrics[2]:.5f}')
        print(f'Test  - Loss: {test_metrics[0]:.5f}, AUC: {test_metrics[1]:.5f}, Acc: {test_metrics[2]:.5f}')
    
    # If only testing, return early
    if num_epochs == 0:
        return
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-8
    )
    

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Setup logging
    writer = SummaryWriter(log_dir=os.path.join(output_root, 'tensorboard'))
    
    # Training loop
    print('==> Starting training...')
    
    best_val_auc = 0
    patience_counter = 0
    best_model = deepcopy(model)
    
    for epoch in trange(num_epochs, desc="Training"):
        # Train for one epoch
        train_loss = train(model, train_loader, criterion, optimizer, device, writer)
        
        # Evaluate on all splits
        train_metrics = test(model, train_loader_at_eval, criterion, device, run, None)
        val_metrics = test(model, val_loader, criterion, device, run, None)
        test_metrics = test(model, test_loader, criterion, device, run, None)
        
        train_eval_loss, train_auc, train_acc = train_metrics
        val_loss, val_auc, val_acc = val_metrics
        test_loss, test_auc, test_acc = test_metrics
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging to tensorboard
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('AUC/Train', train_auc, epoch)
            writer.add_scalar('AUC/Val', val_auc, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Val', val_acc, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Print progress
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train - Loss: {train_loss:.5f}, AUC: {train_auc:.5f}, Acc: {train_acc:.5f}')
        print(f'Val   - Loss: {val_loss:.5f}, AUC: {val_auc:.5f}, Acc: {val_acc:.5f}')
        print(f'Test  - Loss: {test_loss:.5f}, AUC: {test_auc:.5f}, Acc: {test_acc:.5f}')
        print(f'LR: {current_lr:.6f}')
        
        # Save best model based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model = deepcopy(model)
            
            # Save best model
            best_model_path = os.path.join(output_root, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_auc': best_val_auc,
            }, best_model_path)
            print(f'New best model saved! Val AUC: {val_auc:.5f}')
            
        else:
            patience_counter += 1
            print(f'Patience: {patience_counter}/{patience}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            print(f'Best validation AUC: {best_val_auc:.5f}')
            break
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(output_root, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_auc': best_val_auc,
            }, checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch+1}')
    
    if writer:
        writer.close()
    
    # Final evaluation with best model
    print('\n==> Final evaluation with best model...')
    model = best_model
    
    final_train_metrics = test(model, train_loader_at_eval, criterion, device, run, None)
    final_val_metrics = test(model, val_loader, criterion, device, run, None)
    final_test_metrics = test(model, test_loader, criterion, device, run, None)
    
    print(f'\nFinal Results (Best Model):')
    print(f'Train - Loss: {final_train_metrics[0]:.5f}, AUC: {final_train_metrics[1]:.5f}, Acc: {final_train_metrics[2]:.5f}')
    print(f'Val   - Loss: {final_val_metrics[0]:.5f}, AUC: {final_val_metrics[1]:.5f}, Acc: {final_val_metrics[2]:.5f}')
    print(f'Test  - Loss: {final_test_metrics[0]:.5f}, AUC: {final_test_metrics[1]:.5f}, Acc: {final_test_metrics[2]:.5f}')
    
    # Save final results
    results = {
        'train': {'loss': final_train_metrics[0], 'auc': final_train_metrics[1], 'acc': final_train_metrics[2]},
        'val': {'loss': final_val_metrics[0], 'auc': final_val_metrics[1], 'acc': final_val_metrics[2]},
        'test': {'loss': final_test_metrics[0], 'auc': final_test_metrics[1], 'acc': final_test_metrics[2]}
    }
    
    import json
    with open(os.path.join(output_root, 'final_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mitochondria Classification Training Script')
    
    parser.add_argument('--data_flag', default='mitochondria', type=str,
                        help='Dataset flag for naming')
    parser.add_argument('--output_root', default='./output', type=str,
                        help='Output root directory for saving models and logs')
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='Number of training epochs (set to 0 for testing only)')
    parser.add_argument('--size', default=28, type=int,
                        help='Image size (28 or 64)')
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='GPU IDs to use (comma separated)')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--conv', default='ACSConv', type=str,
                        choices=['ACSConv', 'Conv2_5d', 'Conv3d'],
                        help='Convolution type converter')
    parser.add_argument('--pretrained_3d', default='i3d', type=str,
                        help='3D pretraining type')
    parser.add_argument('--download', action='store_true',
                        help='Download dataset if needed')
    parser.add_argument('--as_rgb', action='store_true',
                        help='Convert grayscale to RGB by copying channels')
    parser.add_argument('--shape_transform', action='store_true',
                        help='Apply shape transformation (multiply by 0.5 at eval)')
    parser.add_argument('--model_path', default=None, type=str,
                        help='Path to pretrained model for testing or resume training')
    parser.add_argument('--model_flag', default='resnet18', type=str,
                        choices=['resnet18', 'resnet50'],
                        help='Model architecture to use')
    parser.add_argument('--run', default='experiment1', type=str,
                        help='Experiment name for identification')
    
    args = parser.parse_args()
    
    print("Starting mitochondria classification training...")
    print(f"Configuration: {vars(args)}")
    
    # Call main function with individual arguments (original signature)
    main(
        args.data_flag, args.output_root, args.num_epochs, args.gpu_ids, 
        args.batch_size, args.size, args.conv, args.pretrained_3d, 
        args.download, args.model_flag, args.as_rgb, args.shape_transform, 
        args.model_path, args.run
    )