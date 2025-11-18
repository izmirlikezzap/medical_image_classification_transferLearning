import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set folder path
folder_path = "file:/Users/ayano/PycharmProjects/cardiomegaly_mediastinal_normal_classificatioın/loss_graph_and_cm/"
save_path = os.path.join(folder_path, "saved_plots")
os.makedirs(save_path, exist_ok=True)

# ==========================
# 1. Confusion Matrices
# ==========================
def plot_confusion_matrix(csv_path, title):
    df = pd.read_csv(csv_path, index_col=0)

    # Rename labels: capitalize and replace underscores with spaces
    df.index = df.index.str.replace("_", " ").str.title()
    df.columns = df.columns.str.replace("_", " ").str.title()

    plt.figure(figsize=(6, 5))
    sns.heatmap(df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.tight_layout()

    # Save plot
    file_name = f"{title}_confusion_matrix.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=300)
    plt.close()

# AlexNet confusion matrix
plot_confusion_matrix(
    os.path.join(folder_path, "AlexNet_binary_mediastinal_widening_confusion_matrix.csv"),
    "AlexNet"
)

# InceptionResNet confusion matrix
plot_confusion_matrix(
    os.path.join(folder_path, "InceptionResNet_v2_binary_cardiomegaly_confusion_matrix.csv"),
    "InceptionResNetV2"
)

# ==========================
# 2. Training & Validation Loss Graphs
# ==========================
def plot_losses(csv_path, title):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='s')  # square marker
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='^')  # triangle marker
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save plot
    file_name = f"{title}_loss_graph.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=300)
    plt.close()

# AlexNet loss graph
plot_losses(
    os.path.join(folder_path, "AlexNet_binary_mediastinal_widening_epoch_losses.csv"),
    "AlexNet"
)

# InceptionResNet loss graph
plot_losses(
    os.path.join(folder_path, "InceptionResNet_v2_binary_cardiomegaly_epoch_losses.csv"),
    "InceptionResNetV2"
)
