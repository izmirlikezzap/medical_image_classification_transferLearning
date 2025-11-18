import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm
import random
import shutil

DATASET_ROOT = Path("/media/white-shark/backup_8tb_m2/Padchest/disease_classification/seperate/diseased_regions")
OUTPUT_ROOT = Path("/media/white-shark/backup_8tb_m2/Padchest/disease_classification/seperate/diseased_regions_aug")

CLASSES = ['cardiomegaly', 'normal', 'mediastinal_widening']

# Train-test split oranı
TRAIN_RATIO = 0.8

class MedicalImageAugmentation:
    # (Senin verdiğin sınıf aynen buraya gelecek)
    def __init__(self, severity='minimal'):
        self.severity = severity
        self.severity_params = {
            'minimal': {
                'rotation_limit': 2,
                'brightness_limit': 0.05,
                'contrast_limit': 0.05,
                'noise_var': 0.005
            },
            'mild': {
                'rotation_limit': 3,
                'brightness_limit': 0.08,
                'contrast_limit': 0.08,
                'noise_var': 0.01
            },
            'moderate': {
                'rotation_limit': 5,
                'brightness_limit': 0.1,
                'contrast_limit': 0.1,
                'noise_var': 0.015
            }
        }
        self.params = self.severity_params[severity]

def augment_and_save(image_path, output_folder, aug: MedicalImageAugmentation):
    image = cv2.imread(str(image_path))
    if image is None:
        return False

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    base_name = image_path.stem

    # 1. CLAHE enhanced
    clahe_transform = A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8))
    augmented_1 = clahe_transform(image=image)['image']
    save_path_1 = output_folder / f"{base_name}_clahe.png"
    cv2.imwrite(str(save_path_1), cv2.cvtColor(augmented_1, cv2.COLOR_RGB2BGR))

    # 2. Brightness adjusted
    brightness_transform = A.RandomBrightnessContrast(
        brightness_limit=aug.params['brightness_limit'],
        contrast_limit=aug.params['contrast_limit']
    )
    augmented_2 = brightness_transform(image=image)['image']
    save_path_2 = output_folder / f"{base_name}_bright.png"
    cv2.imwrite(str(save_path_2), cv2.cvtColor(augmented_2, cv2.COLOR_RGB2BGR))

    # 3. Minimal rotation
    rotation_transform = A.Rotate(
        limit=aug.params['rotation_limit'],
        border_mode=cv2.BORDER_CONSTANT,
        value=0
    )
    augmented_3 = rotation_transform(image=image)['image']
    save_path_3 = output_folder / f"{base_name}_rotate.png"
    cv2.imwrite(str(save_path_3), cv2.cvtColor(augmented_3, cv2.COLOR_RGB2BGR))

    # 4. Equipment noise
    noise_transform = A.GaussNoise(var_limit=(0, aug.params['noise_var']))
    augmented_4 = noise_transform(image=image)['image']
    save_path_4 = output_folder / f"{base_name}_noise.png"
    cv2.imwrite(str(save_path_4), cv2.cvtColor(augmented_4, cv2.COLOR_RGB2BGR))

    # 5. Gamma correction
    gamma_transform = A.RandomGamma(gamma_limit=(95, 105))
    augmented_5 = gamma_transform(image=image)['image']
    save_path_5 = output_folder / f"{base_name}_gamma.png"
    cv2.imwrite(str(save_path_5), cv2.cvtColor(augmented_5, cv2.COLOR_RGB2BGR))

    return True

def create_train_test_split_and_augment(dataset_root, output_root, train_ratio=0.8, severity='minimal'):
    random.seed(42)  # Tekrar edilebilirlik için

    aug = MedicalImageAugmentation(severity=severity)

    train_root = output_root / "train"
    test_root = output_root / "test"
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        class_folder = dataset_root / cls
        train_class_folder = train_root / cls
        test_class_folder = test_root / cls
        train_class_folder.mkdir(parents=True, exist_ok=True)
        test_class_folder.mkdir(parents=True, exist_ok=True)

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        all_images = [f for f in class_folder.iterdir() if f.suffix.lower() in image_extensions]

        # Karıştır ve böl
        random.shuffle(all_images)
        split_idx = int(len(all_images) * train_ratio)
        train_images = all_images[:split_idx]
        test_images = all_images[split_idx:]

        print(f"{cls}: {len(all_images)} images → {len(train_images)} train, {len(test_images)} test")

        # Test görüntülerini orijinal haliyle kopyala
        for img_path in tqdm(test_images, desc=f"Copying test images {cls}"):
            shutil.copy2(img_path, test_class_folder / img_path.name)

        # Train görüntülerini augment et ve kaydet
        for img_path in tqdm(train_images, desc=f"Augmenting train images {cls}"):
            augment_and_save(img_path, train_class_folder, aug)

if __name__ == "__main__":
    print("Starting train-test split + augmentation...")

    severity_choice = input("Choose augmentation severity: 1) minimal, 2) mild, 3) moderate [default 1]: ").strip() or "1"
    severity_map = {'1': 'minimal', '2': 'mild', '3': 'moderate'}
    severity = severity_map.get(severity_choice, 'minimal')

    create_train_test_split_and_augment(DATASET_ROOT, OUTPUT_ROOT, TRAIN_RATIO, severity)

    print("Done.")
