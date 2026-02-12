"""
Diabetic Retinopathy – Training Pipeline (PyTorch + GPU)
=========================================================
Two-phase training with RTX 3050 GPU acceleration.
  Phase 1: Frozen backbone → train classification head (10 epochs)
  Phase 2: Unfreeze deeper layers → fine-tune (10 epochs)
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from models.retinopathy_model import build_model
from utils.preprocessing import get_train_loader, get_val_loader


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    return device


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print(f"    Step {batch_idx+1}/{len(loader)} - "
                  f"Loss: {loss.item():.4f} - Acc: {100.*correct/total:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_acc"], label="Train Accuracy")
    ax1.plot(history["val_acc"], label="Val Accuracy")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["train_loss"], label="Train Loss")
    ax2.plot(history["val_loss"], label="Val Loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved → {save_path}")


def train_phase(model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, num_epochs, phase_name, best_acc):
    """Train for a phase and return updated best_acc + history."""
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    patience_counter = 0

    for epoch in range(num_epochs):
        start = time.time()
        print(f"\n  Epoch {epoch+1}/{num_epochs}")
        print("  " + "-" * 40)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]['lr']

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"  LR: {lr:.2e} | Time: {elapsed:.1f}s")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.EARLY_STOP_PATIENCE:
                print(f"  ⚠ Early stopping triggered after {cfg.EARLY_STOP_PATIENCE} epochs without improvement")
                break

    return best_acc, history


def train():
    print("=" * 60)
    print("  👁️  Diabetic Retinopathy Detection – Training (PyTorch)")
    print("=" * 60)

    device = get_device()

    # ── Data ─────────────────────────────────────────────────
    print("\n[INFO] Loading data...")
    train_loader = get_train_loader()
    val_loader = get_val_loader()
    print(f"[INFO] Training batches : {len(train_loader)} ({len(train_loader.dataset)} images)")
    print(f"[INFO] Validation batches: {len(val_loader)} ({len(val_loader.dataset)} images)")
    print(f"[INFO] Classes: {cfg.CLASS_NAMES}")

    # ── Model ────────────────────────────────────────────────
    model = build_model(pretrained=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    num_phase_epochs = cfg.EPOCHS // 2  # 10 each

    # ── Phase 1: Frozen backbone ─────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 1: Training classification head ({num_phase_epochs} epochs)")
    print(f"  Backbone: FROZEN")
    print(f"{'='*60}")

    model.freeze_backbone()
    optimizer1 = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=cfg.LEARNING_RATE)
    scheduler1 = ReduceLROnPlateau(optimizer1, mode='min',
                                   factor=cfg.REDUCE_LR_FACTOR,
                                   patience=cfg.REDUCE_LR_PATIENCE)

    best_acc, history1 = train_phase(
        model, train_loader, val_loader, criterion, optimizer1,
        scheduler1, device, num_phase_epochs, "Phase 1", best_acc
    )

    plot_history(history1, os.path.join(cfg.BASE_DIR, "training_phase1.png"))

    # ── Phase 2: Fine-tune backbone ──────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 2: Fine-tuning backbone ({num_phase_epochs} epochs)")
    print(f"  Backbone: UNFROZEN (layer3 + layer4)")
    print(f"{'='*60}")

    model.unfreeze_backbone(from_layer=6)
    optimizer2 = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=cfg.LEARNING_RATE / 10)
    scheduler2 = ReduceLROnPlateau(optimizer2, mode='min',
                                   factor=cfg.REDUCE_LR_FACTOR,
                                   patience=cfg.REDUCE_LR_PATIENCE)

    best_acc, history2 = train_phase(
        model, train_loader, val_loader, criterion, optimizer2,
        scheduler2, device, num_phase_epochs, "Phase 2", best_acc
    )

    plot_history(history2, os.path.join(cfg.BASE_DIR, "training_phase2.png"))

    # ── Done ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✅ Training complete!")
    print(f"  Best validation accuracy: {best_acc:.2f}%")
    print(f"  Model saved: {cfg.BEST_MODEL_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
