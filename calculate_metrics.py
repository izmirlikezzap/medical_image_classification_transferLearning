from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
import numpy as np


def calculate_metrics(y_true, y_pred, class_names=None):
    """Calculate detailed classification metrics including per-class metrics and binary confusion components (TP, FP, TN, FN)."""

    # Tüm benzersiz sınıfları al
    unique_labels_true = np.unique(y_true)
    unique_labels_pred = np.unique(y_pred)
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    n_classes_detected = len(unique_labels)

    # Eğer class_names verilmişse, tüm sınıfları dahil et
    if class_names:
        n_classes_expected = len(class_names)
        # Tüm olası sınıf labelları (0, 1, 2, ...)
        all_labels = list(range(n_classes_expected))
    else:
        n_classes_expected = n_classes_detected
        all_labels = list(unique_labels)

    # Classification report oluştur - labels parametresi ile tüm sınıfları belirt
    try:
        report = classification_report(
            y_true, y_pred,
            labels=all_labels,  # Tüm sınıfları belirt
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
    except ValueError as e:
        # Eğer hala hata varsa, target_names olmadan dene
        print(f"Warning: Classification report error: {e}")
        try:
            report = classification_report(
                y_true, y_pred,
                labels=all_labels,
                output_dict=True,
                zero_division=0
            )
        except:
            # Son çare: sadece mevcut labellarla
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Weighted metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Confusion Matrix - tüm sınıfları dahil et
    if class_names:
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    else:
        cm = confusion_matrix(y_true, y_pred)

    # Debug için confusion matrix bilgilerini yazdır (sadece gerektiğinde)
    if len(unique_labels_true) < n_classes_expected:
        missing_classes = set(all_labels) - set(unique_labels_true)
        # Removed debug prints to reduce output noise

    # Multi-class classification için metrikler
    total_samples = np.sum(cm)
    n_classes = cm.shape[0]

    # Micro-averaged metrics
    TP = np.trace(cm)  # Correctly classified samples (diagonal sum)
    FP = total_samples - TP  # Incorrectly classified samples
    FN = FP  # For micro-averaging, FP = FN
    TN = total_samples * (n_classes - 1) - FP if n_classes > 1 else 0  # Approximation for multi-class

    # Macro-averaged sensitivity (same as macro recall)
    sensitivity_per_class = []
    specificity_per_class = []

    for i in range(n_classes):
        tp_i = cm[i, i]
        fp_i = cm[:, i].sum() - tp_i  # Column sum minus diagonal
        fn_i = cm[i, :].sum() - tp_i  # Row sum minus diagonal
        tn_i = total_samples - (tp_i + fp_i + fn_i)

        # Sensitivity (Recall) for class i
        sens_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0
        sensitivity_per_class.append(sens_i)

        # Specificity for class i
        spec_i = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0
        specificity_per_class.append(spec_i)

    sensitivity = np.mean(sensitivity_per_class)  # Macro-averaged sensitivity
    specificity = np.mean(specificity_per_class)  # Macro-averaged specificity

    # Binary classification için özel durumlar
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        # Removed debug print to reduce output noise

        # Binary classification için ek metrikler
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

        # Binary için TP, FP, FN, TN'yi güncelle
        TP, FP, FN, TN = tp, fp, fn, tn
    else:
        ppv = precision  # For multi-class, use weighted precision
        npv = 0  # NPV is not well-defined for multi-class

    # Per-class IoU (Intersection over Union)
    iou_per_class = {}
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom != 0 else 0.0
        class_label = class_names[i] if class_names and i < len(class_names) else str(i)
        iou_per_class[class_label] = iou

    # Macro IoU
    macro_iou = np.mean(list(iou_per_class.values()))

    # Per-class metrics
    per_class_metrics = {}
    if class_names:
        for i, class_name in enumerate(class_names):
            if class_name in report:
                per_class_metrics[class_name] = {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1-score': report[class_name]['f1-score'],
                    'support': report[class_name]['support'],
                    'iou': iou_per_class.get(class_name, 0.0)
                }

    result = {
        'accuracy': float(accuracy),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'TN': int(TN),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'macro_iou': float(macro_iou),
        'classification_report': report,
        'iou_per_class': iou_per_class,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist()
    }

    # Binary classification için ek metrikler
    if n_classes == 2:
        result.update({
            'ppv': float(ppv),  # Positive Predictive Value (Precision)
            'npv': float(npv),  # Negative Predictive Value
        })

    return result