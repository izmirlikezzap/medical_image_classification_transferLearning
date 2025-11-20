#!/usr/bin/env python3

import os
import time
import json
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import cv2a

from models_n import get_all_model_configs
from calculate_metrics import calculate_metrics as calc_metrics

# Veri kümesi konumu
DATASET_ROOT = "/media/white-shark/backup_8tb_m2/Padchest/disease_classification/seperate/seperate_originals"

# >>> EKLENDİ: Maskelerin kök klasörü
MASKS_ROOT = "/media/white-shark/backup_8tb_m2/Padchest/disease_classification/seperate/diseased_regions"

# Sınıflandırma sınıfları
# Artık 3 sınıflı değil, iki ayrı ikili sınıflandırma görevi tanımlıyoruz
CLASSIFICATION_TASKS = {
    'binary_cardiomegaly': ['normal', 'cardiomegaly'],
    'binary_mediastinal_widening': ['normal', 'mediastinal_widening'],
}

IMAGE_VERSIONS = ["original", "clahe", "clahe_gamma", "negative"]


class Config:
    def __init__(self, dataset_combination, device, classification_task):
        self.dataset_combination = dataset_combination
        self.images_root = dataset_combination['images_root']
        self.image_version = dataset_combination['image_version']
        self.dataset_name = dataset_combination['name']
        self.classification_task = classification_task
        self.results_dir = f"aug_results_{classification_task}_{self.dataset_name}"
        self.class_names = CLASSIFICATION_TASKS[classification_task]
        self.num_classes = len(self.class_names)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.batch_size = 32
        self.num_epochs = 25
        self.learning_rate = 0.0001
        self.weight_decay = 1e-4
        self.patience = 15
        self.device = device
        os.makedirs(self.results_dir, exist_ok=True)



class XrayDataset(Dataset):
    # >>> GÜNCELLENDİ: masks_root parametresi eklendi
    def __init__(self, images_root, image_version='original', transform=None, class_names=None, masks_root=None):
        self.images_root = Path(images_root)
        self.masks_root = Path(masks_root) if masks_root is not None else None
        self.image_version = image_version
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.samples = []
        self._load_samples()

        print(f"Dataset loaded for {self.image_version}: {len(self.samples)} samples from {self.images_root.name}")

    def _load_samples(self):
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.images_root / class_name
            if not class_dir.exists():
                print(f"WARNING: Directory not found: {class_dir}")
                continue

            class_idx = self.class_to_idx[class_name]
            png_files = list(class_dir.glob('*.png'))

            for image_file in png_files:
                # 2-kanallı kullanım için maskeyi bul
                mask_path = self._infer_mask_path(image_file, class_name)
                self.samples.append((str(image_file), class_idx, mask_path))

    # >>> GÜNCELLENDİ: Maskeyi diseased_regions/<class>/ss içinden bul
    def _infer_mask_path(self, image_file: Path, class_name: str):
        """
        Maskeler şu yapıdadır: MASKS_ROOT/<class>/ss/<mask_dosyaları>
        Eşleme sezgileri:
          - Aynı dosya adı (örn. img123.png -> img123.png)
          - *_mask.* veya *-mask.* eşleşmeleri
          - Aynı "stem" ile başlayan ilk eşleşme
        'normal' sınıfında genelde maske yok -> None döner.
        """
        if self.masks_root is None:
            return None

        # normal için maske zorunlu değil
        if class_name.lower() == "normal":
            return None

        class_ss_dir = self.masks_root / class_name / "ss"
        if not class_ss_dir.exists():
            # güvenli hata toleransı
            return None

        p = Path(image_file)
        stem = p.stem

        # 1) Aynı isim
        cand_same = class_ss_dir / p.name
        if cand_same.exists():
            return str(cand_same)

        # 2) _mask ve -mask varyantları (aynı uzantı)
        cand_mask1 = class_ss_dir / f"{stem}_mask{p.suffix}"
        if cand_mask1.exists():
            return str(cand_mask1)
        cand_mask2 = class_ss_dir / f"{stem}-mask{p.suffix}"
        if cand_mask2.exists():
            return str(cand_mask2)

        # 3) Stem ile başlayan ilk png/jpg dosyası
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            matches = list(class_ss_dir.glob(f"{stem}*{ext[1:]}"))
            if matches:
                return str(matches[0])

        # 4) bulunamazsa None
        return None

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
        elif self.image_version == 'clahe_gamma':
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(image_np)
            gamma_corrected = np.power(enhanced / 255.0, 1.2) * 255.0
            gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)
            return Image.fromarray(gamma_corrected).convert('L')
        elif self.image_version == 'negative':
            negative = 255 - image_np
            return Image.fromarray(negative).convert('L')
        return image

    def __getitem__(self, idx):
        # samples elemanları: (image_path, label, mask_path)
        image_path, label, mask_path = self.samples[idx]
        try:
            # 1. kanal: işlenmiş X-ray
            image = Image.open(image_path).convert('L')
            processed_img = self._process_image(image)

            # 2. kanal: maske (yoksa sıfır maske)
            if mask_path is not None and os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert('L')
            else:
                mask_img = Image.new('L', processed_img.size, color=0)

            # Tutarlı dönüşümler
            if self.transform:
                img_t = self.transform['img'](processed_img)
                mask_t = self.transform['mask'](mask_img)
            else:
                default_tf = get_xray_transforms()
                img_t = default_tf['img'](processed_img)
                mask_t = default_tf['mask'](mask_img)

            # (2,H,W)
            two_ch = torch.cat([img_t, mask_t], dim=0)

            return two_ch, label

        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return torch.zeros(2, 224, 224), label


def get_xray_transforms():
    img_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    mask_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),   # 0..1
    ])
    return {'img': img_tf, 'mask': mask_tf}


def get_data_loaders(config):
    xray_transform = get_xray_transforms()
    full_dataset = XrayDataset(
        images_root=config.images_root,
        image_version=config.image_version,
        transform=xray_transform,
        class_names=config.class_names,
        masks_root=MASKS_ROOT,              # >>> EKLENDİ
    )
    if len(full_dataset) == 0:
        raise ValueError(f"No samples found in dataset for classes: {config.class_names}")

    samples = full_dataset.samples

    paths, labels, _ = zip(*samples)
    train_indices, test_indices = train_test_split(
        range(len(samples)),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    train_counts = {cls: 0 for cls in config.class_names}
    test_counts = {cls: 0 for cls in config.class_names}
    for _, label, __ in [samples[i] for i in train_indices]:
        class_name = config.class_names[label]
        train_counts[class_name] = train_counts.get(class_name, 0) + 1
    for _, label, __ in [samples[i] for i in test_indices]:
        class_name = config.class_names[label]
        test_counts[class_name] = test_counts.get(class_name, 0) + 1

    print(
        f"Train dataset for {config.image_version} ({config.classification_task}): {len(train_dataset)} samples | {train_counts}")
    print(
        f"Test dataset for {config.image_version} ({config.classification_task}): {len(test_dataset)} samples | {test_counts}")

    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device, class_names=None):
    model.train()
    total_loss, preds, targets = 0.0, [], []
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
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
    return total_loss / len(loader), calc_metrics(targets, preds, class_names), preds, targets


def save_confusion_matrix(preds, targets, class_names, output_path):
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(output_path)
    print(f"Confusion matrix saved to {output_path}")


def train_single_model(model_name, model_factory, dataset_combination, device_ids, classification_task, max_retries=3):
    config = Config(dataset_combination, device_ids[0], classification_task)
    result_file = os.path.join(config.results_dir, f"aug_{model_name}_{classification_task}_results.json")
    partial_file = os.path.join(config.results_dir, f"aug_{model_name}_{classification_task}_partial_results.json")

    # Eğer eğitim tamamlandıysa atla
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            print(f"Skipping {model_name} ({classification_task}) - results already exist.")
            return json.load(f)

    attempt = 0
    all_metrics = []

    while attempt < max_retries:
        try:
            print(f"Starting training {model_name} ({classification_task}) - Attempt {attempt + 1}")
            model = model_factory()  # 2-kanal için ilk katman in_channels=2 olmalı
            model = nn.DataParallel(model, device_ids=device_ids)
            model = model.to(device_ids[0])
            train_loader, test_loader = get_data_loaders(config)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            best_acc, patience_counter = 0.0, 0
            epoch_losses = []

            for epoch in range(config.num_epochs):
                train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device_ids[0], config.class_names)
                val_loss, val_metrics, val_preds, val_targets = validate_epoch(model, test_loader, criterion, device_ids[0], config.class_names)
                scheduler.step(val_loss)
                current_acc = val_metrics.get('accuracy', 0)

                epoch_losses.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': current_acc
                })

                partial_result = {
                    'model_name': model_name,
                    'epoch': epoch + 1,
                    'best_val_accuracy_so_far': max(best_acc, current_acc),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'accuracy': current_acc,
                    'timestamp': datetime.now().isoformat()
                }
                all_metrics.append(partial_result)
                with open(partial_file, 'w') as pf:
                    json.dump(all_metrics, pf, indent=2)

                if current_acc > best_acc:
                    best_acc = current_acc
                    patience_counter = 0
                    torch.save(model.module.state_dict(),
                               os.path.join(config.results_dir, f"aug_{model_name}_{classification_task}_best_model.pth"))
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

                print(f"Epoch {epoch + 1}/{config.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {current_acc:.4f} | Best: {best_acc:.4f}")

            # Eğitim bitince sonuçları kaydet
            losses_df = pd.DataFrame(epoch_losses)
            losses_path = os.path.join(config.results_dir, f"{model_name}_{classification_task}_epoch_losses.csv")
            losses_df.to_csv(losses_path, index=False)

            cm_path = os.path.join(config.results_dir, f"{model_name}_{classification_task}_confusion_matrix.csv")
            save_confusion_matrix(val_preds, val_targets, config.class_names, cm_path)

            per_class_metrics = val_metrics.get('per_class_metrics', {})
            class_metrics = {}
            for cls in config.class_names:
                class_metrics.update({
                    f'Sensitivity {cls}': per_class_metrics.get(cls, {}).get('recall', 0),
                    f'Specificity {cls}': val_metrics.get('specificity_per_class', [0] * config.num_classes)[
                        config.class_to_idx[cls]],
                    f'Support {cls}': per_class_metrics.get(cls, {}).get('support', 0)
                })

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
            return result

        except RuntimeError as e:
            print(f"RuntimeError on attempt {attempt + 1} for {model_name} ({classification_task}): {e}")
            attempt += 1
            torch.cuda.empty_cache()
            del model
            del optimizer
            del criterion
            del train_loader
            del test_loader
            time.sleep(10)
            if all_metrics:
                with open(partial_file, 'w') as pf:
                    json.dump(all_metrics, pf, indent=2)

        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1} for {model_name} ({classification_task}): {e}")
            if all_metrics:
                with open(partial_file, 'w') as pf:
                    json.dump(all_metrics, pf, indent=2)
            return None

    print(f"Max retries reached for {model_name} ({classification_task}), skipping training.")
    return None




def run_training():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    device_ids = [2]
    for id in device_ids:
        print(f"Using device {id}: {torch.cuda.get_device_name(id)}")
    all_results = []
    start_time = time.time()
    print(f"Starting X-ray Training:")
    print(f"   Classification Tasks: {len(CLASSIFICATION_TASKS)}")
    print(f"   Image Versions: {IMAGE_VERSIONS}")
    print(f"   Devices: {device_ids}")

    for image_version in IMAGE_VERSIONS:
        print(f"\n{'=' * 80}")
        print(f"STARTING IMAGE VERSION: {image_version}")
        print(f"{'=' * 80}")
        dataset_combination = {
            'name': f"images_{image_version}",
            'image_version': image_version,
            'images_root': DATASET_ROOT
            # (MASKS_ROOT sabitinden okunuyor; dataset_combination'a eklemedim)
        }

        images_path = dataset_combination['images_root']
        if not os.path.exists(images_path):
            print(f"WARNING: Images directory not found: {images_path}")
            continue

        print(f"\nUsing dataset combination: {dataset_combination['name']}")

        for task_name, task_classes in CLASSIFICATION_TASKS.items():
            print(f"\n{'=' * 50}")
            print(f"TASK: {task_name.upper()} - Classes: {task_classes}")
            print(f"{'=' * 50}")
            model_configs = get_all_model_configs(num_classes=len(task_classes))

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
    results_dir = Path("final_results_xray_analysis_aug")
    results_dir.mkdir(exist_ok=True)
    summary = {
        'experiment_info': {
            'purpose': 'X-ray classification with image processing combinations',
            'classification_tasks': list(CLASSIFICATION_TASKS.keys()),
        },
        'timestamp': datetime.now().isoformat(),
        'total_training_time_hours': total_time / 3600,
        'total_experiments': len(results),
        'device': 'cuda:0, cuda:1',
        'results': results
    }
    with open(results_dir / "aug_xray_training_summary.json", 'w') as f:
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
            for cls in r['class_names']:
                row[f'Sensitivity {cls}'] = r.get(f'Sensitivity {cls}', 0)
                row[f'Specificity {cls}'] = r.get(f'Specificity {cls}', 0)
                row[f'Support {cls}'] = r.get(f'Support {cls}', 0)
            df_data.append(row)
        df = pd.DataFrame(df_data)
        df = df.sort_values(['Classification Task', 'Model Name', 'Image Version'])
        df.to_csv(results_dir / "aug_xray_detailed_results.csv", index=False)
        print(f"\nTOP PERFORMING MODELS:")
        top_models = df.nlargest(5, 'Best Val Accuracy')
        for idx, row in top_models.iterrows():
            print(f"  {idx + 1:2}. {row['Model Name'][:15]:<15} ({row['Classification Task']}) "
                  f"{row['Image Version']:<12} - Acc: {row['Best Val Accuracy']:.4f}")


if __name__ == "__main__":
    run_training()
