import os
import pandas as pd

# İşlenecek klasör yolu
folder_path = "/Users/ayano/Desktop/2class_mediastinal_results"

# Silinecek sütunlar
drop_columns = [
    "Sensitivity Normal",
    "Specificity Normal",
    "Sensitivity mediastinal_widening",
    "Specificity mediastinal_widening",
    "Support mediastinal_widening",
    "Macro IoU",

    "Support Normal",
    "Sensitivity Cardiomegaly",
    "Specificity Cardiomegaly",
    "Support Cardiomegaly",

]

# Sütun adlarını yeniden adlandırma eşlemesi
rename_map = {
    "model_name": "Model Name",
    "best_val_accuracy": "Best Accuracy",
    "precision": "Precision",
    "f1_score": "F1 Score",
    "TP": "TP",
    "FP": "FP",
    "FN": "FN",
    "TN": "TN",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "macro_iou": "Macro IoU",
    "final_score": "Final Score",
    "Sensitivity normal": "Sensitivity Normal",
    "Specificity normal": "Specificity Normal",
    "Support normal": "Support Normal",
    "Sensitivity cardiomegaly": "Sensitivity Cardiomegaly",
    "Specificity cardiomegaly": "Specificity Cardiomegaly",
    "Support cardiomegaly": "Support Cardiomegaly"
}

# Klasördeki tüm CSV dosyalarını bul
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)

        # CSV dosyasını oku
        df = pd.read_csv(file_path)

        # Belirtilen sütunları kaldır
        df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore")

        # Sütun adlarını değiştir
        df = df.rename(columns=rename_map)

        # Sayısal değerleri 2 ondalık basamağa yuvarla
        df = df.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

        # Düzenlenmiş CSV dosyasını kaydet (üzerine yazıyor)
        df.to_csv(file_path, index=False)

        print(f"✔ Düzenlendi: {file_name}")
