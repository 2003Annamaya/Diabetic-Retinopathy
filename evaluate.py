"""
Diabetic Retinopathy – Model Evaluation (PyTorch)
===================================================
Generate confusion matrix, classification report, and ROC curves.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

import config as cfg
from models.retinopathy_model import build_model
from utils.preprocessing import get_test_loader


def evaluate(model_path: str = cfg.BEST_MODEL_PATH):
    print("=" * 60)
    print("  Diabetic Retinopathy – Evaluation (PyTorch)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load model
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Load test data
    test_loader = get_test_loader(batch_size=1)
    print(f"[INFO] Test samples: {len(test_loader.dataset)}")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_pred_probs = np.array(all_probs)

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=cfg.CLASS_NAMES, digits=4)
    print("\n── Classification Report ──")
    print(report)

    report_path = os.path.join(cfg.BASE_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[INFO] Report saved → {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cfg.CLASS_NAMES, yticklabels=cfg.CLASS_NAMES)
    plt.title("Confusion Matrix – Diabetic Retinopathy")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(cfg.BASE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved → {cm_path}")

    # ROC Curves
    plt.figure(figsize=(10, 7))
    for i, class_name in enumerate(cfg.CLASS_NAMES):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.title("ROC Curves – One-vs-Rest")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(cfg.BASE_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"[INFO] ROC curves saved → {roc_path}")

    print("\n[INFO] Evaluation complete ✓")


if __name__ == "__main__":
    evaluate()
