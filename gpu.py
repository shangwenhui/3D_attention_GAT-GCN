#!/usr/bin/env python
# computational_efficiency.py
"""
测量模型的计算效率指标：参数量、推理时间、训练时间、GPU显存占用
"""

import torch
import time
import numpy as np
import argparse
from torch_geometric.loader import DataLoader
from utils import TestbedDataset

# ---------- 导入模型类 ----------
from models.gcn import GCNNet
from models.gat_gcn import GAT_GCN


def measure_efficiency(model, device, data_loader, num_warmup=10, num_repeat=100):
    """
    测量推理时间和显存占用
    """
    model.eval()

    # 获取一个 batch 的数据
    batch = next(iter(data_loader)).to(device)

    # ---------- 1. 参数量 ----------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ---------- 2. GPU 预热 ----------
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(batch)

    # ---------- 3. 推理时间 ----------
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_repeat):
            _ = model(batch)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start_time

    avg_batch_time_ms = (elapsed / num_repeat) * 1000  # 每 batch 毫秒
    batch_size = batch.num_graphs
    avg_molecule_time_ms = avg_batch_time_ms / batch_size  # 每分子毫秒

    # ---------- 4. GPU 显存占用 ----------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(batch)
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory_mb = 0.0

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'batch_time_ms': avg_batch_time_ms,
        'molecule_time_ms': avg_molecule_time_ms,
        'peak_memory_mb': peak_memory_mb,
        'batch_size': batch_size
    }


def main():
    parser = argparse.ArgumentParser(description='Computational efficiency measurement')
    parser.add_argument('--model_class', type=str, required=True,
                        choices=['GCN', 'GAT_GCN', 'GAT', 'GIN', 'GAT_GCN_GIN'],
                        help='Model class to evaluate')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model file')
    parser.add_argument('--cuda', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=16, help='Test batch size')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载测试数据（用原始随机划分的测试集即可）
    test_data = TestbedDataset(root='data', dataset='test')
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # 模型映射
    model_dict = {
        'GCN': GCNNet,
        'GAT_GCN': GAT_GCN,
    }

    if args.model_class not in model_dict:
        from models.gat import GATNet
        from models.ginconv import GINConvNet
        from models.gat_gcn_gin import GAT_GCN_GIN
        model_dict.update({
            'GAT': GATNet,
            'GIN': GINConvNet,
            'GAT_GCN_GIN': GAT_GCN_GIN
        })

    model_class = model_dict[args.model_class]
    model = model_class().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print(f'\n===== Measuring {args.model_class} =====')
    results = measure_efficiency(model, device, test_loader)

    print(f'Total parameters:       {results["total_params"]:,}')
    print(f'Trainable parameters:   {results["trainable_params"]:,}')
    print(f'Batch size:             {results["batch_size"]}')
    print(f'Avg batch time:         {results["batch_time_ms"]:.4f} ms')
    print(f'Avg per molecule time:  {results["molecule_time_ms"]:.6f} ms')
    print(f'Throughput:             {1000 / results["molecule_time_ms"]:.1f} molecules/sec')
    print(f'Peak GPU memory:        {results["peak_memory_mb"]:.2f} MB')

    # 保存结果
    import pandas as pd
    df = pd.DataFrame([results])
    df.to_csv(f'efficiency_{args.model_class}.csv', index=False)
    print(f'\nResults saved to efficiency_{args.model_class}.csv')


if __name__ == '__main__':
    main()