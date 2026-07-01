import numpy as np
import pandas as pd
import sys, os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from models.gat_gcn_gin import GAT_GCN_GIN
from utils import *
import random

# ---------- Training function ----------
def train(model, device, train_loader, optimizer, epoch, lambda_pearson=0.01):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    mse_loss_fn = nn.MSELoss()

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = data.y.view(-1, 1).float().to(device)

        mse_loss = mse_loss_fn(output, target)

        # Differentiable Pearson correlation coefficient
        pred_flat = output.squeeze()
        true_flat = target.squeeze()
        pred_mean = pred_flat.mean()
        true_mean = true_flat.mean()
        pred_centered = pred_flat - pred_mean
        true_centered = true_flat - true_mean
        cov = (pred_centered * true_centered).mean()
        std_pred = pred_centered.std(unbiased=False)
        std_true = true_centered.std(unbiased=False)
        if std_pred == 0 or std_true == 0:
            pearson = torch.tensor(0.0, device=device)
        else:
            pearson = cov / (std_pred * std_true + 1e-8)

        loss = mse_loss + lambda_pearson * (1 - pearson)

        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tMSE: {:.6f}\tPearson: {:.4f}'.format(
                epoch,
                batch_idx * len(data.x),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item(),
                mse_loss.item(),
                pearson.item()
            ))

# ---------- Prediction function ----------
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

# ---------- Main program ----------
# Command line arguments: first argument is model index (0-4), second optional is GPU device id
if len(sys.argv) < 2:
    print("Usage: python script.py <model_index> [cuda_device]")
    print("model_index: 0=GINConvNet, 1=GATNet, 2=GAT_GCN, 3=GCNNet, 4=GAT_GCN_GIN")
    sys.exit(1)

modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet,  GAT_GCN_GIN][int(sys.argv[1])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 2:
    cuda_name = "cuda:" + str(int(sys.argv[2]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 16
LR = 0.001          # Can be adjusted as needed
LOG_INTERVAL = 20
NUM_EPOCHS = 100
LAMBDA_PEARSON = 0.01
SEED = 42           # Fix seed for reproducibility

print('Learning rate:', LR)
print('Epochs:', NUM_EPOCHS)
print('Random seed:', SEED)

# Set random seed
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f'\n===== Running on {model_st} =====')
# Data file paths (fixed names)
processed_train = 'data/processed/train.pt'
processed_val   = 'data/processed/val.pt'
processed_test  = 'data/processed/test.pt'

if not all(os.path.isfile(p) for p in [processed_train, processed_val, processed_test]):
    print('Please run create_data.py first!')
    sys.exit(1)

# Load data
train_data = TestbedDataset(root='data', dataset='train')
val_data   = TestbedDataset(root='data', dataset='val')
test_data  = TestbedDataset(root='data', dataset='test')

train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
model = modeling().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_mse = 1000
best_epoch = -1
model_file = f'model_{model_st}_best.model'        # Save best model
result_file = f'result_{model_st}_best.csv'        # Best validation metrics
test_result_file = f'test_result_{model_st}_best.csv'  # Test set metrics

for epoch in range(NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch + 1, lambda_pearson=LAMBDA_PEARSON)
    G, P = predicting(model, device, val_loader)  # Evaluate on validation set

    # Calculate all metrics
    mse_val = mse(G, P)
    rmse_val = rmse(G, P)
    mae_val = mae(G, P)
    pearson_val = pearson(G, P)
    spearman_val = spearman(G, P)
    ci_val = ci(G, P)
    r2_val = r2_score(G, P)
    ret = [mse_val, rmse_val, mae_val, pearson_val, spearman_val, ci_val, r2_val]

    if mse_val < best_val_mse:
        best_val_mse = mse_val
        best_epoch = epoch + 1
        best_metrics = ret
        # Save model and validation metrics
        torch.save(model.state_dict(), model_file)
        with open(result_file, 'w') as f:
            f.write(','.join(map(str, ret)))
        print(f'>>> Validation MSE improved at epoch {best_epoch}: {mse_val:.4f} (model saved)')

    print(f'Epoch {epoch+1}: MSE={mse_val:.4f}, RMSE={rmse_val:.4f}, MAE={mae_val:.4f}, '
          f'Pearson={pearson_val:.4f}, Spearman={spearman_val:.4f}, CI={ci_val:.4f}, R2={r2_val:.4f}')

# Training finished, load best model and evaluate on test set
model.load_state_dict(torch.load(model_file))
test_G, test_P = predicting(model, device, test_loader)
test_ret = [mse(test_G, test_P), rmse(test_G, test_P), mae(test_G, test_P),
            pearson(test_G, test_P), spearman(test_G, test_P), ci(test_G, test_P), r2_score(test_G, test_P)]
with open(test_result_file, 'w') as f:
    f.write(','.join(map(str, test_ret)))

# Print only validation best results (test results are not printed to terminal)
print(f'\n=== Best model from epoch {best_epoch} ===')
print(f'Validation: MSE={best_metrics[0]:.4f}, RMSE={best_metrics[1]:.4f}, MAE={best_metrics[2]:.4f}, '
      f'Pearson={best_metrics[3]:.4f}, Spearman={best_metrics[4]:.4f}, CI={best_metrics[5]:.4f}, R2={best_metrics[6]:.4f}')