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
from scipy import stats
from copy import deepcopy


# ------------------ Training function (same as original, with Pearson regularization) ------------------
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


# ------------------ Prediction function ------------------
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


# ------------------ Main program (multiple runs + statistics) ------------------
N_RUNS = 10               # Number of repeated runs
BASE_SEED = 1234          # Base random seed

# Command line arguments: model index and optional GPU device id
if len(sys.argv) < 2:
    print("Usage: python training.py <model_index> [cuda_device]")
    print("model_index: 0=GINConvNet, 1=GATNet, 2=GAT_GCN, 3=GCNNet, 4=GAT_GCN_GIN")
    sys.exit(1)

modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet, GAT_GCN_GIN][int(sys.argv[1])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 2:
    cuda_name = "cuda:" + str(int(sys.argv[2]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 16
LR = 0.001
LOG_INTERVAL = 20
NUM_EPOCHS = 100
LAMBDA_PEARSON = 0.01   # Can be adjusted, keep consistent with original

print('Learning rate:', LR)
print('Epochs:', NUM_EPOCHS)
print(f'Number of runs: {N_RUNS}')

# Fixed data files (train + validation, no test set)
processed_train = 'data/processed/train.pt'
processed_val   = 'data/processed/val.pt'

if not all(os.path.isfile(p) for p in [processed_train, processed_val]):
    print('Please run 3Dcreate.py first!')
    sys.exit(1)

# Load data once, same split for all runs
train_data = TestbedDataset(root='data', dataset='train')
val_data   = TestbedDataset(root='data', dataset='val')

# For recording results across runs
all_results = []
global_best_mse = float('inf')
global_best_model_state = None

for run in range(N_RUNS):
    seed = BASE_SEED + run
    print(f'\n--- Run {run + 1}/{N_RUNS} (seed={seed}) ---')

    # Set all random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_mse = float('inf')
    best_epoch = -1
    best_metrics = None
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        # Train using global LAMBDA_PEARSON
        train(model, device, train_loader, optimizer, epoch + 1, lambda_pearson=LAMBDA_PEARSON)
        G, P = predicting(model, device, val_loader)   # Evaluate on validation set

        # Calculate all metrics (ensure these functions are in utils)
        mse_val = mse(G, P)
        rmse_val = rmse(G, P)
        mae_val = mae(G, P)
        pearson_val = pearson(G, P)
        spearman_val = spearman(G, P)
        ci_val = ci(G, P)
        r2_val = r2_score(G, P)
        ret = [mse_val, rmse_val, mae_val, pearson_val, spearman_val, ci_val, r2_val]

        if ret[0] < best_mse:
            best_mse = ret[0]
            best_epoch = epoch + 1
            best_metrics = ret
            # Keep a deep copy of the current best model in this run
            best_model_state = deepcopy(model.state_dict())

    # Record this run's best metrics
    all_results.append([run + 1, model_st] + best_metrics)
    print(
        f"Run {run + 1} best metrics: MSE={best_metrics[0]:.4f}, RMSE={best_metrics[1]:.4f}, MAE={best_metrics[2]:.4f}, "
        f"Pearson={best_metrics[3]:.4f}, Spearman={best_metrics[4]:.4f}, CI={best_metrics[5]:.4f}, R2={best_metrics[6]:.4f}")

    # Update the overall best model (across runs) if this run is better
    if best_mse < global_best_mse:
        global_best_mse = best_mse
        global_best_model_state = best_model_state
        print(f'*** New overall best model (run {run+1}) with validation MSE: {best_mse:.4f}')

# ------------------ Summary statistics ------------------
cols = ['Run', 'Model', 'MSE', 'RMSE', 'MAE', 'Pearson', 'Spearman', 'CI', 'R2']
df_results = pd.DataFrame(all_results, columns=cols)

summary = df_results[['MSE', 'RMSE', 'MAE', 'Pearson', 'Spearman', 'CI', 'R2']].agg(['mean', 'std'])
print("\n===== Summary over {} runs =====".format(N_RUNS))
print(summary)

print("\n----- 95% Confidence Intervals (t-distribution) -----")

def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - h, mean + h

for metric in ['MSE', 'RMSE', 'MAE', 'Pearson', 'Spearman', 'CI', 'R2']:
    mean, lower, upper = mean_confidence_interval(df_results[metric].values)
    print(f"{metric}: {mean:.4f}  (95% CI: [{lower:.4f}, {upper:.4f}])")

# Save results to CSV
df_results.to_csv(f'result_{model_st}_10runs.csv', index=False)
summary.to_csv(f'summary_{model_st}_10runs.csv')
print("\nResults saved to CSV files.")

# Save the globally best model from all runs
if global_best_model_state is not None:
    model_file = f'model_{model_st}_best.model'
    torch.save(global_best_model_state, model_file)
    print(f'Best model (overall validation MSE: {global_best_mse:.4f}) saved to {model_file}')
else:
    print('No model saved (all runs may have failed).')