#!/usr/bin/env python
"""
One-click clustering split evaluation script (with integrated 3D feature extraction)
1. Read data from the full CSV, perform Butina clustering split
2. Use the same molecular graph generation function as training to save the split subsets as .pt files
3. Load the pre-trained best model, evaluate on the cluster test set, and provide Bootstrap confidence intervals
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.ML.Cluster import Butina
import networkx as nx
import os, random, argparse

# Suppress RDKit warnings
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
from utils import TestbedDataset, mse, rmse, mae, pearson, spearman, ci, r2_score


# ==================== Atom feature functions ====================
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


# ==================== 3D molecular graph generation ====================
def smile_to_graph_with_3d(smile, numConfs=50):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    if mol is None:
        return None

    try:
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=numConfs,
                                          params=AllChem.ETKDGv3())
    except Exception as e:
        return None

    if not cids:
        return None

    energies = []
    for cid in cids:
        try:
            mmff_status = MMFFOptimizeMolecule(mol, confId=cid)
        except Exception:
            mmff_status = -1
        if mmff_status == 0:
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            if mp is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
                if ff is not None:
                    energy = ff.CalcEnergy()
                    energies.append((cid, energy))
                else:
                    energies.append((cid, float('inf')))
            else:
                energies.append((cid, float('inf')))
        else:
            energies.append((cid, float('inf')))

    if not energies:
        return None

    best_cid = min(energies, key=lambda x: x[1])[0]

    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        features.append(atom_features(atom))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    coords = []
    conf = mol.GetConformer(best_cid)
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])

    edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    coords_t = torch.tensor(coords, dtype=torch.float)
    row, col = edge_index_t[0], edge_index_t[1]
    dist = torch.norm(coords_t[row] - coords_t[col], dim=1, keepdim=True)

    if dist.max() > 0:
        edge_s = dist / dist.max()
    else:
        edge_s = dist

    return c_size, features, edge_index, coords, edge_s


# ==================== Butina clustering split ====================
def butina_cluster_split(csv_path, fingerprint_radius=2, fingerprint_bits=1024, cluster_cutoff=0.4, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(csv_path).dropna(subset=['compound_iso_smiles', 'affinity'])
    smiles_list = df['compound_iso_smiles'].tolist()
    affinity_list = df['affinity'].tolist()

    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=fingerprint_radius, nBits=fingerprint_bits)
            fps.append(fp)
            valid_idx.append(i)

    n = len(fps)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            dist = 1.0 - sim
            dist_mat[i, j] = dist_mat[j, i] = dist

    dist_list = [dist_mat[i, j] for i in range(1, n) for j in range(i)]
    clusters = Butina.ClusterData(dist_list, n, cluster_cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)

    cluster_labels = [-1] * n
    for cid, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_labels[idx] = cid

    cluster_ids = list(range(len(clusters)))
    random.shuffle(cluster_ids)
    n_train = int(len(cluster_ids) * 0.8)
    n_val = int(len(cluster_ids) * 0.1)

    train_clusters = set(cluster_ids[:n_train])
    val_clusters = set(cluster_ids[n_train:n_train + n_val])
    test_clusters = set(cluster_ids[n_train + n_val:])

    splits = {'train': [], 'val': [], 'test': []}
    for i, c in enumerate(cluster_labels):
        if c in train_clusters:
            splits['train'].append(valid_idx[i])
        elif c in val_clusters:
            splits['val'].append(valid_idx[i])
        elif c in test_clusters:
            splits['test'].append(valid_idx[i])

    return df, splits


# ==================== Generate PT files (using PyG standard format) ====================
def save_pt_files(df, splits, output_dir='data/processed', prefix='cluster'):
    """
    Save the clustered subsets as PyG standard format (data, slices) tuples,
    exactly matching the format generated by create_data.py so that TestbedDataset can load them correctly.
    """
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        data_list = []
        for idx in splits[split]:
            row = df.iloc[idx]
            result = smile_to_graph_with_3d(row['compound_iso_smiles'], numConfs=1)
            if result is None:
                continue
            c_size, features, edge_index, coords, edge_s = result
            data = Data(x=torch.Tensor(features),
                        edge_index=torch.LongTensor(edge_index).t().contiguous(),
                        y=torch.FloatTensor([row['affinity']]),
                        pos=torch.Tensor(coords),
                        edge_s=edge_s)
            data_list.append(data)
        # Use InMemoryDataset.collate to convert the list to PyG standard format (data, slices)
        data, slices = InMemoryDataset.collate(data_list)
        pt_path = os.path.join(output_dir, f'{prefix}_{split}.pt')
        torch.save((data, slices), pt_path)
        print(f'{split}: {len(data_list)} graphs saved to {pt_path}')


# ==================== Evaluation function ====================
def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap=1000, seed=42):
    n = len(y_true)
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    scores = np.array(scores)
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def predicting(model, device, loader):
    model.eval()
    total_preds, total_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds.append(output.cpu())
            total_labels.append(data.y.view(-1, 1).cpu())
    return torch.cat(total_labels).numpy().flatten(), torch.cat(total_preds).numpy().flatten()


def evaluate_on_cluster(args):
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    from models.gat import GATNet
    from models.gat_gcn import GAT_GCN
    from models.gcn import GCNNet
    from models.ginconv import GINConvNet
    from models.gat_gcn_gin import GAT_GCN_GIN

    model_dict = {
        'GINConvNet': GINConvNet, 'GAT': GATNet, 'GAT_GCN': GAT_GCN,
        'GCN': GCNNet, 'GAT_GCN_GIN': GAT_GCN_GIN
    }
    model_class = model_dict[args.model_class]
    model = model_class().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    test_pt = f'data/processed/cluster_test.pt'
    if not os.path.isfile(test_pt):
        raise FileNotFoundError(f"PT file not found: {test_pt}")

    test_data = TestbedDataset(root='data', dataset='cluster_test')
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    labels, preds = predicting(model, device, test_loader)
    mse_val = mse(labels, preds)
    pearson_val = pearson(labels, preds)
    spearman_val = spearman(labels, preds)
    ci_val = ci(labels, preds)
    rmse_val = rmse(labels, preds)
    mae_val = mae(labels, preds)
    r2_val = r2_score(labels, preds)

    mse_boot_mean, mse_low, mse_up = bootstrap_metric(labels, preds, mse)
    pearson_boot_mean, p_low, p_up = bootstrap_metric(labels, preds, pearson)

    print(f'\nCluster Test Results for {args.model_class}:')
    print(f'MSE: {mse_val:.4f} (95% CI: [{mse_low:.4f}, {mse_up:.4f}])')
    print(f'Pearson: {pearson_val:.4f} (95% CI: [{p_low:.4f}, {p_up:.4f}])')
    print(f'RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, Spearman: {spearman_val:.4f}, CI: {ci_val:.4f}, R2: {r2_val:.4f}')

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    res_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'Pearson', 'Spearman', 'CI', 'R2',
                   'MSE_CI_low', 'MSE_CI_up', 'Pearson_CI_low', 'Pearson_CI_up'],
        'Value': [mse_val, rmse_val, mae_val, pearson_val, spearman_val, ci_val, r2_val,
                  mse_low, mse_up, p_low, p_up]
    })
    res_df.to_csv(os.path.join(args.output_dir, f'cluster_{model_name}.csv'), index=False)


# ==================== Main program ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to full dataset CSV')
    parser.add_argument('--model', type=str, required=True, help='Model weight file')
    parser.add_argument('--model_class', type=str, required=True, help='Model class name')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--cutoff', type=float, default=0.4, help='Butina clustering cutoff (Tanimoto distance)')
    parser.add_argument('--output_dir', type=str, default='cluster_eval_results')
    parser.add_argument('--skip_cluster', action='store_true', help='Skip clustering if PT already exists')
    parser.add_argument('--prefix', type=str, default='cluster', help='Prefix for PT files')
    args = parser.parse_args()

    if not args.skip_cluster:
        print("Performing Butina clustering...")
        df, splits = butina_cluster_split(args.csv, cluster_cutoff=args.cutoff)
        print("Generating PT files...")
        save_pt_files(df, splits, prefix=args.prefix)
    else:
        print("Skipping clustering, assuming PT files already exist.")

    evaluate_on_cluster(args)


if __name__ == '__main__':
    main()