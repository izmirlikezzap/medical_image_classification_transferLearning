#!/usr/bin/env python3

import os
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import cv2
import re

from models_n import get_all_model_configs
from calculate_metrics import calculate_metrics as calc_metrics

# Görüntü işleme versiyonları
IMAGE_VERSIONS = ["original", "clahe", "negative", "clahe_gamma"]

# Veri kümesi konumu - augmentasyonlu klasör
DATASET_ROOT = "/media/white-shark/backup_8tb_m2/Padchest/disease_classification/seperate/diseased_regions_aug"

# Veri kümesi kombinasyonları
DATASET_COMBINATIONS = [
    {'name': f"images_{img_version}", 'image_version': img_version, 'images_root': DATASET_ROOT}
    for img_version in IMAGE_VERSIONS
]

# Sınıflandırma sınıfları
TARGET_CLASSES_3 = ['cardiomegaly', 'normal', 'mediastinal_widening']
CLASSIFICATION_TASKS = {
    '3class': TARGET_CLASSES_3,
    'cardio_vs_normal': ['cardiomegaly', 'normal'],
    'mediastinal_vs_normal': ['mediastinal_widening', 'normal'],
}


class Config:
    def __init__(self, dataset_combination, device, classification_task='3class'):
        self.dataset_combination = dataset_combination
        self.images_root = dataset_combination['images_root']
        self.image_version = dataset_combination['image_version']
        self.dataset_name = dataset_combination['name']
        self.classification_task = classification_task
        self.results_dir = f"aug_results_{classification_task}_{self.dataset_name}"
        self.class_names = CLASSIFICATION_TASKS[classification_task]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.batch_size = 128
        self.num_epochs = 25
        self.learning_rate = 0.0001
        self.weight_decay = 1e-4
        self.patience = 15
        self.device = device
        os.makedirs(self.results_dir, exist_ok=True)


class XrayDataset(Dataset):
    def __init__(self, images_root, image_version='original', transform=None, class_names=None):
        self.images_root = Path(images_root)
        self.image_version = image_version
        self.transform = transform
        self.class_names = class_names or TARGET_CLASSES_3
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.samples = []
        self._load_samples()
        class_dist = self._get_class_distribution()
        total_samples = sum(class_dist.values())
        print(f"Dataset loaded: {total_samples} samples | {dict(class_dist)}")

    def _get_class_distribution(self):
        class_counts = {class_name: 0 for class_name in self.class_names}
        for _, class_idx in self.samples:
            class_name = self.class_names[class_idx]
            class_counts[class_name] += 1
        return class_counts

    def _load_samples(self):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        for class_name in self.class_names:
            class_dir = self.images_root / class_name
            if not class_dir.exists():
                print(f"WARNING: Directory not found: {class_dir}")
                continue
            class_idx = self.class_to_idx[class_name]
            for ext in image_extensions:
                for image_file in class_dir.glob(f'*{ext}'):
                    if self.image_version == 'original':
                        # original versiyonda _clahe, _negative, _clahe_gamma içermeyen dosyalar
                        if ('_clahe' not in image_file.stem and
                            '_negative' not in image_file.stem and
                            '_clahe_gamma' not in image_file.stem):
                            self.samples.append((str(image_file), class_idx))
                    else:
                        # Diğer versiyonlar için dosya isminde image_version içermeli
                        if f"_{self.image_version}" in image_file.stem:
                            self.samples.append((str(image_file), class_idx))
                # Aynı şeyi büyük harfli uzantı için
                for image_file in class_dir.glob(f'*{ext.upper()}'):
                    if self.image_version == 'original':
                        if ('_clahe' not in image_file.stem and
                            '_negative' not in image_file.stem and
                            '_clahe_gamma' not in image_file.stem):
                            self.samples.append((str(image_file), class_idx))
                    else:
                        if f"_{self.image_version}" in image_file.stem:
                            self.samples.append((str(image_file), class_idx))
            print(f"Found {len([s for s in self.samples if s[1] == class_idx])} images for class {class_name} [{self.image_version}]")

    def __len__(self):
        return len(self.samples)

    def _process_image(self, image):
        image_np = np.array(image)
        if self.image_version == 'original':
            return image
        elif self.image_version == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(image_np)
            return Image.fromarray(enhanced).convert('L')
        elif self.image_version == 'negative':
            negative = 255 - image_np
            return Image.fromarray(negative).convert('L')
        elif self.image_version == 'clahe_gamma':
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(image_np)
            gamma_corrected = np.power(enhanced / 255.0, 1.2) * 255.0
            gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)
            return Image.fromarray(gamma_corrected).convert('L')
        return image

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        try:
            image = Image.open(image_path).convert('L')  # Siyah-beyaz
            processed_img = self._process_image(image)
            if self.transform:
                processed_img = self.transform(processed_img)
            return processed_img, label
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return torch.zeros(1, 224, 224), label


def get_xray_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Siyah-beyaz için tek kanal
    ])


def extract_base_id_from_image_path(image_path):
    image_name = Path(image_path).stem
    numeric_matches = re.findall(r'\d+', image_name)
    return max(numeric_matches, key=len) if numeric_matches else image_name


def get_id_based_split(dataset, train_ratio=0.8, random_seed=42):
    sample_ids = {}
    for idx, (image_path, label) in enumerate(dataset.samples):
        base_id = extract_base_id_from_image_path(image_path)
        if base_id not in sample_ids:
            sample_ids[base_id] = {'indices': [], 'label': label}
        sample_ids[base_id]['indices'].append(idx)
    print(f"Found {len(sample_ids)} unique base IDs")
    class_ids = {}
    for base_id, info in sample_ids.items():
        label = info['label']
        if label not in class_ids:
            class_ids[label] = []
        class_ids[label].append(base_id)
    np.random.seed(random_seed)
    train_indices, val_indices = [], []
    for label, ids in class_ids.items():
        np.random.shuffle(ids)
        split_point = int(len(ids) * train_ratio)
        train_ids = ids[:split_point]
        val_ids = ids[split_point:]
        print(f"Class {label}: {len(train_ids)} train IDs, {len(val_ids)} val IDs")
        for base_id in train_ids:
            train_indices.extend(sample_ids[base_id]['indices'])
        for base_id in val_ids:
            val_indices.extend(sample_ids[base_id]['indices'])
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    return train_indices, val_indices


def verify_id_based_split(dataset, train_indices, val_indices):
    print("Verifying ID-based split...")
    train_base_ids = set(extract_base_id_from_image_path(dataset.samples[idx][0]) for idx in train_indices)
    val_base_ids = set(extract_base_id_from_image_path(dataset.samples[idx][0]) for idx in val_indices)
    overlap = train_base_ids.intersection(val_base_ids)
    if overlap:
        print(f"DATA LEAKAGE DETECTED! {len(overlap)} IDs overlap")
        return False
    print(f"No data leakage detected! Train IDs: {len(train_base_ids)}, Val IDs: {len(val_base_ids)}")
    return True


def get_data_loaders(config):
    xray_transform = get_xray_transforms()
    full_dataset = XrayDataset(
        images_root=config.images_root,
        image_version=config.image_version,
        transform=xray_transform,
        class_names=config.class_names
    )
    if len(full_dataset) == 0:
        raise ValueError(f"No samples found in dataset: {config.images_root}")
    train_indices, val_indices = get_id_based_split(full_dataset)
    print(f"ID-based split: Train={len(train_indices)}, Val={len(val_indices)}")
    verify_id_based_split(full_dataset, train_indices, val_indices)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    return DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True), \
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)


def train_epoch(model, loader, criterion, optimizer, device, class_names=None):
    model.train()
    total_loss, preds, targets = 0.0, [], []
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds += torch.argmax(outputs, 1).cpu().tolist()
        targets += labels.cpu().tolist()
    return total_loss / len(loader), calc_metrics(targets, preds, class_names)


def validate_epoch(model, loader, criterion, device, class_names=None):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds += torch.argmax(outputs, 1).cpu().tolist()
            targets += labels.cpu().tolist()
    return total_loss / len(loader), calc_metrics(targets, preds, class_names)


def train_single_model(model_name, model_factory, dataset_combination, device_ids, classification_task='3class'):
    config = Config(dataset_combination, device_ids[0], classification_task)
    result_file = os.path.join(config.results_dir, f"aug_{model_name}_{classification_task}_results.json")
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    try:
        model = model_factory()
        # Modeli tek kanallı giriş için uyarla
        if hasattr(model, 'conv1'):
            model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                                    stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
        # DataParallel ile birden fazla GPU kullan
        model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device_ids[0])
        train_loader, val_loader = get_data_loaders(config)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        best_acc, patience_counter = 0.0, 0
        print(f"Training {model_name} | {config.dataset_name} | {classification_task} on GPUs {device_ids}")
        for epoch in range(config.num_epochs):
            train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device_ids[0],
                                                    config.class_names)
            val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device_ids[0], config.class_names)
            scheduler.step(val_loss)
            current_acc = val_metrics.get('accuracy', 0)
            if current_acc > best_acc:
                best_acc = current_acc
                patience_counter = 0
                torch.save(model.module.state_dict(),
                           os.path.join(config.results_dir, f"{model_name}_{classification_task}_best_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            print(
                f"Epoch {epoch + 1}/{config.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {current_acc:.4f} | Best: {best_acc:.4f}")

        # Sınıf bazında metrikleri al
        per_class_metrics = val_metrics.get('per_class_metrics', {})
        class_metrics = {}
        for cls in config.class_names:
            class_metrics.update({
                f'Sensitivity {cls}': per_class_metrics.get(cls, {}).get('recall', 0),
                f'Specificity {cls}': val_metrics.get('specificity_per_class', [0] * config.num_classes)[
                    config.class_to_idx[cls]],
                f'Support {cls}': per_class_metrics.get(cls, {}).get('support', 0)
            })

        # Final Score: accuracy, f1_score, sensitivity, specificity ortalaması
        final_score = np.mean([
            val_metrics.get('accuracy', 0),
            val_metrics.get('f1_weighted', 0),
            val_metrics.get('sensitivity', 0),
            val_metrics.get('specificity', 0)
        ])

        result = {
            'model_name': model_name,
            'dataset_combination': config.dataset_name,
            'image_version': config.image_version,
            'classification_task': classification_task,
            'num_classes': config.num_classes,
            'class_names': config.class_names,
            'best_val_accuracy': best_acc,
            'accuracy': val_metrics.get('accuracy', 0),
            'precision': val_metrics.get('precision_weighted', 0),
            'f1_score': val_metrics.get('f1_weighted', 0),
            'TP': val_metrics.get('TP', 0),
            'FP': val_metrics.get('FP', 0),
            'FN': val_metrics.get('FN', 0),
            'TN': val_metrics.get('TN', 0),
            'sensitivity': val_metrics.get('sensitivity', 0),
            'specificity': val_metrics.get('specificity', 0),
            'macro_iou': val_metrics.get('macro_iou', 0),
            'final_score': final_score,
            **class_metrics,
            'timestamp': datetime.now().isoformat()
        }
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Completed {model_name} ({classification_task}) - Best Acc: {best_acc:.4f}")
        return result
    except Exception as e:
        print(f"Error training {model_name} ({classification_task}): {e}")
        return None


def run_training():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    device_ids = [0, 1]  # cuda:0 ve cuda:1
    all_results = []
    start_time = time.time()
    print(f"Starting X-ray Training:")
    print(f"   Classification Tasks: {len(CLASSIFICATION_TASKS)}")
    print(f"   Dataset Combinations: {len(DATASET_COMBINATIONS)}")
    print(f"   Devices: {device_ids}")
    for task_idx, (task_name, task_classes) in enumerate(CLASSIFICATION_TASKS.items()):
        print(f"\n{'=' * 50}")
        print(f"TASK: {task_name.upper()} - Classes: {task_classes}")
        print(f"{'=' * 50}")
        model_configs = get_all_model_configs(num_classes=len(task_classes))
        for combo_idx, dataset_combination in enumerate(DATASET_COMBINATIONS):
            images_path = dataset_combination['images_root']
            if not os.path.exists(images_path):
                print(f"WARNING: Images directory not found: {images_path}")
                continue
            print(f"\nDataset Combination {combo_idx + 1}/{len(DATASET_COMBINATIONS)}: {dataset_combination['name']}")
            for model_idx, (model_name, model_factory) in enumerate(model_configs.items()):
                print(f"\nModel {model_idx + 1}/{len(model_configs)}: {model_name}")
                result = train_single_model(model_name, model_factory, dataset_combination, device_ids, task_name)
                if result:
                    all_results.append(result)
    total_time = time.time() - start_time
    save_final_results(all_results, total_time)
    print(f"\nX-ray Training Completed!")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Total experiments: {len(all_results)}")


def save_final_results(results, total_time):
    results_dir = Path("final_results_aug")
    results_dir.mkdir(exist_ok=True)
    summary = {
        'experiment_info': {
            'purpose': 'X-ray classification with image processing combinations',
            'classification_tasks': CLASSIFICATION_TASKS,
            'image_versions': IMAGE_VERSIONS,
            'dataset_combinations': DATASET_COMBINATIONS,
        },
        'timestamp': datetime.now().isoformat(),
        'total_training_time_hours': total_time / 3600,
        'total_experiments': len(results),
        'device': 'cuda:0, cuda:1',
        'results': results
    }
    with open(results_dir / "xray_training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    if results:
        df_data = []
        for r in results:
            row = {
                'Model Name': r['model_name'],
                'Best Val Accuracy': r['best_val_accuracy'],
                'TP': r.get('TP', 0),
                'FP': r.get('FP', 0),
                'FN': r.get('FN', 0),
                'TN': r.get('TN', 0),
                'Sensitivity': r.get('sensitivity', 0),
                'Specificity': r.get('specificity', 0),
                'Precision': r.get('precision', 0),
                'F1-Score': r.get('f1_score', 0),
                'Accuracy': r.get('accuracy', 0),
                'Final Score': r.get('final_score', 0),
                'Classification Task': r['classification_task'],
                'Image Version': r['image_version']
            }
            # Sınıf bazında metrikler
            for cls in r['class_names']:
                row[f'Sensitivity {cls}'] = r.get(f'Sensitivity {cls}', 0)
                row[f'Specificity {cls}'] = r.get(f'Specificity {cls}', 0)
                row[f'Support {cls}'] = r.get(f'Support {cls}', 0)
            df_data.append(row)
        df = pd.DataFrame(df_data)
        df = df.sort_values(['Classification Task', 'Model Name', 'Image Version'])
        df.to_csv(results_dir / "xray_detailed_results_aug.csv", index=False)
        print(f"\nTOP PERFORMING MODELS:")
        top_models = df.nlargest(5, 'Best Val Accuracy')
        for idx, row in top_models.iterrows():
            print(f"  {idx + 1:2}. {row['Model Name'][:15]:<15} ({row['Classification Task']}) "
                  f"{row['Image Version']:<12} - Acc: {row['Best Val Accuracy']:.4f}")


if __name__ == "__main__":
    run_training()
