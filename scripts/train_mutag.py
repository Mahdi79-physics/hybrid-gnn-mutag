# ===============================
# scripts/train_mutag.py
# ===============================

import os
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Import our modular code
from src.models import HybridGIN_GAT_Model
from src.early_stopping import EarlyStopping
from src.utils import train_epoch, eval_epoch

# ===============================
# Hyperparameters
# ===============================
input_dim = None  # Will be set after loading dataset
hidden_dim = 128
output_dim = 128
num_layers = 5
heads = 2
dropout = 0.2
batch_size = 32
learning_rate = 0.001
num_epochs = 100
patience = 20  # Early stopping patience
n_splits = 10  # K-fold CV

# ===============================
# Load dataset
# ===============================
dataset = TUDataset(root='data/TUDataset', name='MUTAG')
input_dim = dataset.num_features

# ===============================
# Prepare K-Fold cross-validation
# ===============================
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store results
fold_results = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# Ensure figures folder exists
os.makedirs("figures", exist_ok=True)

# ===============================
# Cross-validation loop
# ===============================
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n=== Fold {fold+1}/{n_splits} ===")

    # Split dataset
    train_dataset = dataset[train_idx.tolist()]
    val_dataset = dataset[val_idx.tolist()]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, criterion, scheduler
    model = HybridGIN_GAT_Model(input_dim, hidden_dim, output_dim, num_layers, heads, dropout)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # ===============================
    # Training loop for this fold
    # ===============================
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)

        # Scheduler step
        scheduler.step(val_loss)

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Store results for this fold
    fold_results['train_loss'].append(train_loss)
    fold_results['train_acc'].append(train_acc)
    fold_results['val_loss'].append(val_loss)
    fold_results['val_acc'].append(val_acc)

# ===============================
# Compute average metrics
# ===============================
avg_train_loss = np.mean(fold_results['train_loss'])
avg_train_acc = np.mean(fold_results['train_acc'])
avg_val_loss = np.mean(fold_results['val_loss'])
avg_val_acc = np.mean(fold_results['val_acc'])

print("\n=== 10-Fold Cross-Validation Results ===")
print(f"Average Train Loss: {avg_train_loss:.4f}")
print(f"Average Train Accuracy: {avg_train_acc:.4f}")
print(f"Average Validation Loss: {avg_val_loss:.4f}")
print(f"Average Validation Accuracy: {avg_val_acc:.4f}")

# ===============================
# Plot results
# ===============================

# Train & Val Loss
plt.figure()
plt.plot(range(1, n_splits+1), fold_results['train_loss'], marker='o', label='Train Loss')
plt.plot(range(1, n_splits+1), fold_results['val_loss'], marker='o', label='Val Loss')
plt.xlabel('Fold')
plt.ylabel('Loss')
plt.title('Train & Validation Loss per Fold')
plt.legend()
plt.grid(True)
plt.savefig('figures/loss.png', dpi=300)
plt.close()

# Train & Val Accuracy
plt.figure()
plt.plot(range(1, n_splits+1), fold_results['train_acc'], marker='o', label='Train Accuracy')
plt.plot(range(1, n_splits+1), fold_results['val_acc'], marker='o', label='Val Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Train & Validation Accuracy per Fold')
plt.legend()
plt.grid(True)
plt.savefig('figures/accuracy.png', dpi=300)
plt.close()

print("\nFigures saved in 'figures/' folder.")
