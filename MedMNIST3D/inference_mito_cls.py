#inference_mito_cls.py

#!/usr/bin/env python3
"""
Mitochondria Classification Inference Script
Loads a trained model and performs inference on new data
"""

import argparse
import os
import time
import json
import csv
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from acsconv.converters import ACSConverter, Conv2_5dConverter, Conv3dConverter
from models import ResNet18, ResNet50
from utils import Transform3D, model_to_syncbn
from mito_dataset import MitoPatchDataset
from sklearn.metrics import roc_auc_score, accuracy_score


def load_model(model_path, model_flag, conv, pretrained_3d, n_channels, n_classes, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {model_path}")
    
    # Create model architecture
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
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'], strict=True)
        print(f"Model loaded successfully. Best validation AUC: {checkpoint.get('best_auc', 'N/A')}")
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("Model loaded successfully from state dict")
    
    model.eval()
    return model


def inference(model, data_loader, device, return_predictions=True):
    """Perform inference on dataset"""
    model.eval()
    
    all_predictions = []predictions = torch.argmax(outputs, dim=1)
    all_probabilities = []
    all_targets = []
    
    print("Running inference...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.long().to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions (class with highest probability)
            predictions = torch.argmax(outputs, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")
    
    print(f"Inference completed on {len(all_predictions)} samples")
    
    # Calculate metrics if ground truth is available (not all zeros)
    unique_targets = np.unique(all_targets)
    if len(unique_targets) > 1:
        probs_positive = [prob[1] for prob in all_probabilities]
        auc = roc_auc_score(all_targets, probs_positive)
        accuracy = accuracy_score(all_targets, all_predictions)
        print(f"Performance metrics:")
        print(f"  AUC: {auc:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
    else:
        print("Ground truth labels are all the same (placeholder values), skipping metric calculation")
    
    return all_predictions, all_probabilities, all_targets


def save_predictions(predictions, probabilities, targets, csv_path, output_path):
    """Save predictions to CSV file with original coordinates"""
    print(f"Saving predictions to {output_path}")
    
    # Read original CSV to get coordinates
    original_data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header if present
        for row in reader:
            original_data.append(row)
    
    # Create output CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['z', 'y', 'x', 'original_label', 'predicted_label', 'prob_class_0', 'prob_class_1', 'confidence'])
        
        # Write data
        for i, row in enumerate(original_data):
            if i < len(predictions):
                z, y, x, original_label = row
                pred_label = predictions[i]
                prob_0, prob_1 = probabilities[i]
                confidence = max(prob_0, prob_1)  # Confidence is the maximum probability
                
                writer.writerow([z, y, x, original_label, pred_label, f"{prob_0:.4f}", f"{prob_1:.4f}", f"{confidence:.4f}"])
    
    print(f"Predictions saved successfully!")


def analyze_predictions(predictions, probabilities):
    """Analyze and summarize predictions"""
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    print("\n=== Prediction Analysis ===")
    
    # Class distribution
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"Predicted class distribution:")
    for class_id, count in zip(unique, counts):
        percentage = count / len(predictions) * 100
        print(f"  Class {class_id}: {count} samples ({percentage:.1f}%)")
    
    # Confidence statistics
    confidences = np.max(probabilities, axis=1)
    print(f"\nConfidence statistics:")
    print(f"  Mean confidence: {np.mean(confidences):.4f}")
    print(f"  Median confidence: {np.median(confidences):.4f}")
    print(f"  Min confidence: {np.min(confidences):.4f}")
    print(f"  Max confidence: {np.max(confidences):.4f}")
    
    # High/low confidence samples
    high_conf_threshold = 0.9
    low_conf_threshold = 0.6
    
    high_conf_count = np.sum(confidences >= high_conf_threshold)
    low_conf_count = np.sum(confidences < low_conf_threshold)
    
    print(f"\nConfidence distribution:")
    print(f"  High confidence (≥{high_conf_threshold}): {high_conf_count} samples ({high_conf_count/len(predictions)*100:.1f}%)")
    print(f"  Low confidence (<{low_conf_threshold}): {low_conf_count} samples ({low_conf_count/len(predictions)*100:.1f}%)")
    
    return {
        'class_distribution': dict(zip(unique.tolist(), counts.tolist())),
        'confidence_stats': {
            'mean': float(np.mean(confidences)),
            'median': float(np.median(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        },
        'high_confidence_count': int(high_conf_count),
        'low_confidence_count': int(low_conf_count)
    }


def main():
    parser = argparse.ArgumentParser(description='Mitochondria Classification Inference Script')
    
    # Model and data paths
    parser.add_argument('--model_path', 
                        default='/projects/weilab/qiongwang/MedMNIST_experiments/MedMNIST_experiments/MedMNIST3D/output/mitochondria/250627_172909/best_model.pth',
                        type=str, help='Path to trained model checkpoint')
    parser.add_argument('--csv_path', 
                        default='/projects/weilab/qiongwang/MedMNIST_experiments/MedMNIST_experiments/MedMNIST3D/mito_infer_centers.csv',
                        type=str, help='Path to inference CSV file')
    parser.add_argument('--data_dir', 
                        default='/projects/weilab/qiongwang/MedMNIST_experiments/MedMNIST_experiments/MedMNIST3D/mito_infer_downloaded_patch',
                        type=str, help='Directory containing inference patches')
    parser.add_argument('--output_dir', default='./inference_results', type=str,
                        help='Output directory for saving results')
    
    # Model configuration (should match training configuration)
    parser.add_argument('--model_flag', default='resnet18', type=str,
                        choices=['resnet18', 'resnet50'], help='Model architecture')
    parser.add_argument('--conv', default='ACSConv', type=str,
                        choices=['ACSConv', 'Conv2_5d', 'Conv3d'], help='Convolution type')
    parser.add_argument('--pretrained_3d', default='i3d', type=str,
                        help='3D pretraining type')
    parser.add_argument('--shape_transform', action='store_true',
                        help='Apply shape transformation (multiply by 0.5)')
    
    # Inference configuration
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for inference')
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='GPU IDs to use')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Setup device
    str_ids = args.gpu_ids.split(',')
    gpu_ids_list = [int(str_id) for str_id in str_ids if int(str_id) >= 0]
    
    if len(gpu_ids_list) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids_list[0])
        device = torch.device(f'cuda:{gpu_ids_list[0]}')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = time.strftime("%y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"inference_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Model parameters
    n_channels = 1
    n_classes = 2
    
    # Load model
    model = load_model(
        args.model_path, args.model_flag, args.conv, args.pretrained_3d,
        n_channels, n_classes, device
    )
    
    # Prepare data
    print("==> Preparing inference data...")
    
    # Data transform (no augmentation for inference)
    transform = Transform3D(mul='0.5') if args.shape_transform else Transform3D()
    
    # Create dataset
    inference_dataset = MitoPatchDataset(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        transform=transform
    )
    
    print(f"Inference dataset size: {len(inference_dataset)}")
    
    # Create data loader
    inference_loader = data.DataLoader(
        dataset=inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Important: don't shuffle for inference
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Run inference
    print("==> Running inference...")
    predictions, probabilities, targets = inference(model, inference_loader, device)
    
    # Save predictions
    output_csv = os.path.join(output_dir, 'infer_predictions.csv')
    save_predictions(predictions, probabilities, targets, args.csv_path, output_csv)
    
    # Analyze predictions
    analysis = analyze_predictions(predictions, probabilities)
    
    # Save analysis results
    analysis_file = os.path.join(output_dir, 'analysis.json')
    with open(analysis_file, 'w') as f:
        json.dump({
            'inference_config': vars(args),
            'dataset_size': len(inference_dataset),
            'analysis': analysis
        }, f, indent=2)
    
    print(f"\nInference completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"  - Predictions: {output_csv}")
    print(f"  - Analysis: {analysis_file}")


if __name__ == '__main__':
    main()






# import argparse
# import os
# import json
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.utils.data as data
# from acsconv.converters import ACSConverter, Conv2_5dConverter, Conv3dConverter
# from models import ResNet18, ResNet50
# from utils import Transform3D, model_to_syncbn
# from mito_dataset import MitoPatchDataset
# from sklearn.metrics import roc_auc_score, accuracy_score
# from tqdm import tqdm


# def inference(model, data_loader, device):
#     """Perform inference on given dataset"""
#     model.eval()
#     all_predictions = []
#     all_probabilities = []
#     all_filenames = []
    
#     with torch.no_grad():
#         for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="Inference")):
#             if len(batch_data) == 3:  # (inputs, targets, filenames)
#                 inputs, targets, filenames = batch_data
#                 all_filenames.extend(filenames)
#             else:  # (inputs, targets)
#                 inputs, targets = batch_data
#                 # Generate dummy filenames if not provided
#                 batch_size = inputs.size(0)
#                 batch_filenames = [f"sample_{batch_idx}_{i}" for i in range(batch_size)]
#                 all_filenames.extend(batch_filenames)
            
#             inputs = inputs.to(device)
            
#             outputs = model(inputs)
            
#             # Get probabilities and predictions
#             probs = torch.softmax(outputs, dim=1)
#             predictions = torch.argmax(outputs, dim=1)
            
#             # Store results
#             all_predictions.extend(predictions.cpu().numpy())
#             all_probabilities.extend(probs.cpu().numpy())
    
#     return all_predictions, all_probabilities, all_filenames


# def evaluate_with_labels(model, data_loader, device):
#     """Evaluate model on dataset with ground truth labels"""
#     model.eval()
#     all_targets = []
#     all_predictions = []
#     all_probabilities = []
    
#     with torch.no_grad():
#         for batch_data in tqdm(data_loader, desc="Evaluation"):
#             if len(batch_data) >= 2:
#                 inputs, targets = batch_data[0], batch_data[1]
#             else:
#                 continue
                
#             inputs = inputs.to(device)
#             targets = targets.long().to(device)
            
#             outputs = model(inputs)
            
#             # Get probabilities for positive class
#             probs = torch.softmax(outputs, dim=1)
#             predictions = torch.argmax(outputs, dim=1)
            
#             all_targets.extend(targets.cpu().numpy())
#             all_predictions.extend(predictions.cpu().numpy())
#             all_probabilities.extend(probs[:, 1].cpu().numpy())
    
#     # Calculate metrics
#     accuracy = accuracy_score(all_targets, all_predictions)
    
#     # Calculate AUC if both classes are present
#     unique_targets = np.unique(all_targets)
#     if len(unique_targets) < 2:
#         print(f"Warning: Only one class present in evaluation: {unique_targets}")
#         auc = 0.5
#     else:
#         auc = roc_auc_score(all_targets, all_probabilities)
    
#     return accuracy, auc, all_targets, all_predictions, all_probabilities


# def save_results(predictions, probabilities, filenames, output_path):
#     """Save inference results to CSV file"""
#     results_df = pd.DataFrame({
#         'filename': filenames,
#         'predicted_class': predictions,
#         'prob_class_0': [prob[0] for prob in probabilities],
#         'prob_class_1': [prob[1] for prob in probabilities],
#         'confidence': [max(prob) for prob in probabilities]
#     })
    
#     results_df.to_csv(output_path, index=False)
#     print(f"Results saved to: {output_path}")
    
#     # Print summary statistics
#     print(f"\nInference Summary:")
#     print(f"Total samples: {len(predictions)}")
#     print(f"Class distribution:")
#     print(f"  Class 0: {sum(p == 0 for p in predictions)} ({sum(p == 0 for p in predictions)/len(predictions)*100:.1f}%)")
#     print(f"  Class 1: {sum(p == 1 for p in predictions)} ({sum(p == 1 for p in predictions)/len(predictions)*100:.1f}%)")
#     print(f"Average confidence: {np.mean([max(prob) for prob in probabilities]):.4f}")


# def main():
#     parser = argparse.ArgumentParser(description='Mitochondria Classification Inference Script')
    
#     # Required arguments
#     parser.add_argument('--model_path', 
#                         default='/projects/weilab/qiongwang/MedMNIST_experiments/MedMNIST_experiments/MedMNIST3D/output/mitochondria/250627_172909/best_model.pth',
#                         type=str, help='Path to trained model')
#     parser.add_argument('--data_dir', 
#                         default='/projects/weilab/qiongwang/MedMNIST_experiments/MedMNIST_experiments/MedMNIST3D/mito_infer_patch',
#                         type=str, help='Directory containing inference data')
#     parser.add_argument('--csv_path', 
#                         default='/projects/weilab/qiongwang/MedMNIST_experiments/MedMNIST_experiments/MedMNIST3D/mito_infer_centers.csv',
#                         type=str, help='CSV file with inference data info')
#     parser.add_argument('--output_dir', default='./inference_output', type=str,
#                         help='Output directory for saving results')
    
#     # Model configuration (should match training configuration)
#     parser.add_argument('--model_flag', default='resnet18', type=str,
#                         choices=['resnet18', 'resnet50'],
#                         help='Model architecture (must match training)')
#     parser.add_argument('--conv', default='ACSConv', type=str,
#                         choices=['ACSConv', 'Conv2_5d', 'Conv3d'],
#                         help='Convolution type (must match training)')
#     parser.add_argument('--pretrained_3d', default='i3d', type=str,
#                         help='3D pretraining type (must match training)')
    
#     # Inference configuration
#     parser.add_argument('--batch_size', default=32, type=int,
#                         help='Batch size for inference')
#     parser.add_argument('--gpu_ids', default='0', type=str,
#                         help='GPU IDs to use')
#     parser.add_argument('--shape_transform', action='store_true',
#                         help='Apply shape transformation (multiply by 0.5)')
#     parser.add_argument('--has_labels', action='store_true',
#                         help='Whether the dataset has ground truth labels for evaluation')
    
#     args = parser.parse_args()
    
#     print("Starting mitochondria classification inference...")
#     print(f"Configuration: {vars(args)}")
    
#     # Setup device
#     str_ids = args.gpu_ids.split(',')
#     gpu_ids_list = [int(str_id) for str_id in str_ids if int(str_id) >= 0]
    
#     if len(gpu_ids_list) > 0:
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids_list[0])
#         device = torch.device(f'cuda:{gpu_ids_list[0]}')
#     else:
#         device = torch.device('cpu')
    
#     print(f"Using device: {device}")
    
#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     # Setup data transform
#     transform = Transform3D(mul='0.5') if args.shape_transform else Transform3D()
    
#     # Create inference dataset
#     print('==> Loading inference data...')
#     try:
#         inference_dataset = MitoPatchDataset(
#             csv_path=args.csv_path,
#             data_dir=args.data_dir,
#             transform=transform
#         )
#         print(f"Inference dataset size: {len(inference_dataset)}")
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         return
    
#     # Create data loader
#     inference_loader = data.DataLoader(
#         dataset=inference_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True if device.type == 'cuda' else False
#     )
    
#     # Build model
#     print('==> Building model...')
#     n_channels = 1
#     n_classes = 2
    
#     if args.model_flag == 'resnet18':
#         model = ResNet18(in_channels=n_channels, num_classes=n_classes)
#     elif args.model_flag == 'resnet50':
#         model = ResNet50(in_channels=n_channels, num_classes=n_classes)
#     else:
#         raise ValueError(f"Unsupported model: {args.model_flag}")
    
#     # Apply convolution converters (must match training configuration)
#     if args.conv == 'ACSConv':
#         model = model_to_syncbn(ACSConverter(model))
#     elif args.conv == 'Conv2_5d':
#         model = model_to_syncbn(Conv2_5dConverter(model))
#     elif args.conv == 'Conv3d':
#         if args.pretrained_3d == 'i3d':
#             model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))
#         else:
#             model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=None))
    
#     model = model.to(device)
    
#     # Load trained model
#     print(f'==> Loading model from {args.model_path}...')
#     try:
#         checkpoint = torch.load(args.model_path, map_location=device)
#         if 'net' in checkpoint:
#             model.load_state_dict(checkpoint['net'], strict=True)
#             print(f"Model loaded successfully. Best AUC from training: {checkpoint.get('best_auc', 'N/A')}")
#         else:
#             model.load_state_dict(checkpoint['model_state_dict'], strict=True)
#             print("Model loaded successfully.")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return
    
#     # Perform inference
#     print('==> Starting inference...')
    
#     if args.has_labels:
#         # Evaluate with ground truth labels
#         accuracy, auc, targets, predictions, probabilities = evaluate_with_labels(model, inference_loader, device)
#         print(f'\nEvaluation Results:')
#         print(f'Accuracy: {accuracy:.4f}')
#         print(f'AUC: {auc:.4f}')
        
#         # Save evaluation results
#         eval_results = {
#             'accuracy': float(accuracy),
#             'auc': float(auc),
#             'total_samples': len(targets),
#             'class_distribution': {
#                 'class_0': int(sum(t == 0 for t in targets)),
#                 'class_1': int(sum(t == 1 for t in targets))
#             }
#         }
        
#         with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
#             json.dump(eval_results, f, indent=2)
        
#         # Create detailed results DataFrame
#         detailed_results = pd.DataFrame({
#             'ground_truth': targets,
#             'predicted_class': predictions,
#             'probability_class_1': probabilities,
#             'correct': [t == p for t, p in zip(targets, predictions)]
#         })
#         detailed_results.to_csv(os.path.join(args.output_dir, 'detailed_evaluation.csv'), index=False)
        
#     else:
#         # Pure inference without labels
#         predictions, probabilities, filenames = inference(model, inference_loader, device)
        
#         # Save inference results
#         results_path = os.path.join(args.output_dir, 'inference_results.csv')
#         save_results(predictions, probabilities, filenames, results_path)
    
#     print(f'\nInference completed. Results saved to: {args.output_dir}')


# if __name__ == '__main__':
#     main()