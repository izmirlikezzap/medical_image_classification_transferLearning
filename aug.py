
#!/usr/bin/env python3
import os
import cv2
import albumentations as A
from pathlib import Path
from tqdm import tqdm


class MedicalImageAugmentation:
    """
    Conservative medical image augmentation for chest X-rays.
    """

    def __init__(self, severity='minimal'):
        self.severity = severity

        # Conservative parameters for medical imaging
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


def create_augmented_dataset(input_path, output_path, severity='minimal'):
    """
    Create augmented dataset with exactly 5 variations per image.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    aug = MedicalImageAugmentation(severity=severity)

    total_processed = 0
    total_generated = 0

    print(f"Creating {severity} severity augmentations...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Process each class folder
    for class_folder in input_path.iterdir():
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name
        print(f"\nProcessing: {class_name}")

        output_class_folder = output_path / class_name
        output_class_folder.mkdir(exist_ok=True)

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in class_folder.iterdir()
                       if f.suffix.lower() in image_extensions]

        print(f"Found {len(image_files)} images → {len(image_files) * 5} augmented")

        for img_file in tqdm(image_files, desc=f"Augmenting {class_name}"):
            try:
                # Read and resize image
                image = cv2.imread(str(img_file))
                if image is None:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                base_name = img_file.stem

                # 1. CLAHE enhanced
                clahe_transform = A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8))
                augmented_1 = clahe_transform(image=image)['image']
                save_path_1 = output_class_folder / f"{base_name}_clahe.png"
                cv2.imwrite(str(save_path_1), cv2.cvtColor(augmented_1, cv2.COLOR_RGB2BGR))

                # 2. Brightness adjusted
                brightness_transform = A.RandomBrightnessContrast(
                    brightness_limit=aug.params['brightness_limit'],
                    contrast_limit=aug.params['contrast_limit']
                )
                augmented_2 = brightness_transform(image=image)['image']
                save_path_2 = output_class_folder / f"{base_name}_bright.png"
                cv2.imwrite(str(save_path_2), cv2.cvtColor(augmented_2, cv2.COLOR_RGB2BGR))

                # 3. Minimal rotation
                rotation_transform = A.Rotate(
                    limit=aug.params['rotation_limit'],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                )
                augmented_3 = rotation_transform(image=image)['image']
                save_path_3 = output_class_folder / f"{base_name}_rotate.png"
                cv2.imwrite(str(save_path_3), cv2.cvtColor(augmented_3, cv2.COLOR_RGB2BGR))

                # 4. Equipment noise
                noise_transform = A.GaussNoise(var_limit=(0, aug.params['noise_var']))
                augmented_4 = noise_transform(image=image)['image']
                save_path_4 = output_class_folder / f"{base_name}_noise.png"
                cv2.imwrite(str(save_path_4), cv2.cvtColor(augmented_4, cv2.COLOR_RGB2BGR))

                # 5. Gamma correction
                gamma_transform = A.RandomGamma(gamma_limit=(95, 105))
                augmented_5 = gamma_transform(image=image)['image']
                save_path_5 = output_class_folder / f"{base_name}_gamma.png"
                cv2.imwrite(str(save_path_5), cv2.cvtColor(augmented_5, cv2.COLOR_RGB2BGR))

                total_processed += 1
                total_generated += 5

            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue

    print(f"\n=== Complete ===")
    print(f"Processed: {total_processed} images")
    print(f"Generated: {total_generated} augmented images")
    print(f"Factor: 5x")


if __name__ == "__main__":
    INPUT_DATASET = "/media/white-shark/backup_8tb_m2/Padchest/disease_classification/seperate/seperate_originals"
    OUTPUT_DATASET = "/media/white-shark/backup_8tb_m2/Padchest/disease_classification/seperate/seperate_augmented"

    severity = 'minimal'  # 'minimal', 'mild', 'moderate'
    create_augmented_dataset(INPUT_DATASET, OUTPUT_DATASET, severity)
    print("Augmentation completed!")
