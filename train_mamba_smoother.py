#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mamba Smoother 训练脚本 - 完整CUDA加速版本
使用Mamba SSM进行时序平滑，支持GPU加速
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import f1_score, classification_report

print("[*] 尝试导入Mamba...")
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("  [OK] Mamba-SSM可用")
except ImportError as e:
    MAMBA_AVAILABLE = False
    print(f"  [WARN] Mamba-SSM不可用: {e}")
    print("  [WARN] 将跳过Mamba训练,仅使用ESN")


def get_device():
    """获取最佳可用设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  [OK] 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"       VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  [WARN] CUDA不可用,使用CPU")
    return device


class SequenceDataset(Dataset):
    """按participant分组的序列数据集"""
    
    def __init__(self, inputs, targets, groups):
        self.sequences = []
        self.targets_list = []
        
        unique_groups = np.unique(groups)
        for g in unique_groups:
            mask = groups == g
            self.sequences.append(torch.FloatTensor(inputs[mask]))
            self.targets_list.append(torch.LongTensor(targets[mask]))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets_list[idx]


def collate_sequences(batch):
    """自定义collate函数 - 处理变长序列"""
    sequences, targets = zip(*batch)
    # 直接返回列表，不进行padding（逐序列处理）
    return sequences, targets


if MAMBA_AVAILABLE:
    class MambaSmoother(nn.Module):
        """Mamba Smoother for temporal smoothing - GPU加速版"""
        
        def __init__(self, n_classes=4, d_model=64, n_layers=2, d_state=16, d_conv=4,
                     expand=2, dropout=0.1, aux_dim=0):
            super().__init__()
            self.n_classes = n_classes
            self.d_model = d_model
            
            input_dim = n_classes + aux_dim
            self.input_proj = nn.Linear(input_dim, d_model)
            
            self.mamba_layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(n_layers)
            ])
            
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            self.output_proj = nn.Linear(d_model, n_classes)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            # x: (batch, seq_len, input_dim) 或 (seq_len, input_dim)
            if x.dim() == 2:
                x = x.unsqueeze(0)  # 添加batch维度
            
            x = self.input_proj(x)
            
            for mamba, norm in zip(self.mamba_layers, self.norms):
                x_out = mamba(x)
                x = norm(x + self.dropout(x_out))
            
            logits = self.output_proj(x)
            return logits.squeeze(0) if logits.size(0) == 1 else logits


class MambaSmootherWrapper:
    """Mamba Smoother的sklearn风格包装器"""
    
    def __init__(self, n_classes=4, d_model=64, n_layers=2, epochs=50, lr=1e-3,
                 batch_size=8, aux_dim=0, device=None):
        self.n_classes = n_classes
        self.d_model = d_model
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.aux_dim = aux_dim
        self.device = device or get_device()
        self.model = None
    
    def fit(self, y_pred_proba, y_true, groups, aux_features=None):
        """训练Mamba smoother"""
        print(f"\n[*] 训练Mamba Smoother (CUDA: {self.device.type == 'cuda'})")
        
        # 准备输入
        if aux_features is not None:
            inputs = np.hstack([y_pred_proba, aux_features])
            self.aux_dim = aux_features.shape[1]
        else:
            inputs = y_pred_proba
        
        # 创建数据集
        dataset = SequenceDataset(inputs, y_true, groups)
        
        # 初始化模型
        input_dim = inputs.shape[1]
        self.model = MambaSmoother(
            n_classes=self.n_classes,
            d_model=self.d_model,
            n_layers=self.n_layers,
            aux_dim=0,  # 已经在input_dim中了
        ).to(self.device)
        
        # 修正input_proj的输入维度
        self.model.input_proj = nn.Linear(input_dim, self.d_model).to(self.device)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            n_samples = 0
            
            # 随机打乱序列顺序
            indices = np.random.permutation(len(dataset))
            
            for idx in indices:
                seq, targets = dataset[idx]
                seq = seq.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(seq)
                loss = criterion(logits, targets)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item() * len(targets)
                n_samples += len(targets)
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / n_samples
                print(f"    Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        print("  [OK] 训练完成")
    
    def predict(self, y_pred_proba, groups, aux_features=None):
        """预测"""
        if aux_features is not None:
            inputs = np.hstack([y_pred_proba, aux_features])
        else:
            inputs = y_pred_proba
        
        self.model.eval()
        all_preds = []
        
        unique_groups = np.unique(groups)
        with torch.no_grad():
            for g in unique_groups:
                mask = groups == g
                seq = torch.FloatTensor(inputs[mask]).to(self.device)
                logits = self.model(seq)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.append((mask, preds))
        
        # 重组预测结果
        y_pred = np.zeros(len(groups), dtype=np.int64)
        for mask, preds in all_preds:
            y_pred[mask] = preds
        
        return y_pred


def compute_auxiliary_features(X_raw, mode='enmo'):
    """计算辅助特征"""
    if mode == 'none':
        return None
    elif mode == 'enmo':
        # 简单ENMO统计
        enmo = np.sqrt((X_raw**2).sum(axis=-1)).mean(axis=-1, keepdims=True)
        return enmo
    else:  # full
        enmo = np.sqrt((X_raw**2).sum(axis=-1))
        return np.column_stack([
            enmo.mean(axis=-1),
            enmo.std(axis=-1),
            enmo.max(axis=-1),
        ])


def main():
    if not MAMBA_AVAILABLE:
        print("\n[!] Mamba-SSM不可用,无法训练Mamba模型")
        print("[!] 请安装: pip install mamba-ssm causal-conv1d>=1.1.0")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='训练Mamba Smoother (CUDA加速)')
    parser.add_argument('--exp_id', type=str, required=True, help='实验ID')
    parser.add_argument('--d_model', type=int, default=64, help='隐藏维度')
    parser.add_argument('--n_layers', type=int, default=2, help='层数')
    parser.add_argument('--aux_features', type=str, default='enmo',
                       choices=['none', 'enmo', 'full'], help='辅助特征')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--data_dir', type=str, default='./prepared_data')
    parser.add_argument('--output_dir', type=str, default='./models')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  训练Mamba Smoother: {args.exp_id} (CUDA加速)")
    print(f"{'='*70}\n")
    
    device = get_device()
    
    # 加载数据
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("[1/4] 加载数据...")
    split_info = joblib.load(data_dir / 'train_test_split.pkl')
    train_mask = split_info['train_mask']
    test_mask = split_info['test_mask']
    
    y_train_proba = np.load(data_dir / 'y_train_proba_rf.npy')
    y_test_proba = np.load(data_dir / 'y_test_proba_rf.npy')
    
    Y = np.load(data_dir / 'Y_Walmsley2020.npy')
    P = np.load(data_dir / 'P.npy')
    
    y_train = Y[train_mask]
    y_test = Y[test_mask]
    P_train = P[train_mask]
    P_test = P[test_mask]
    
    print(f"  训练集: {y_train_proba.shape}")
    print(f"  测试集: {y_test_proba.shape}")
    
    # 辅助特征
    aux_train = None
    aux_test = None
    if args.aux_features != 'none':
        print(f"\n[2/4] 计算辅助特征 ({args.aux_features})...")
        try:
            X_raw = np.load(data_dir / 'X.npy', mmap_mode='r')
            aux_train = compute_auxiliary_features(X_raw[train_mask], args.aux_features)
            aux_test = compute_auxiliary_features(X_raw[test_mask], args.aux_features)
            print(f"  辅助特征维度: {aux_train.shape[1]}")
        except FileNotFoundError:
            print("  [WARN] X.npy不存在,跳过辅助特征")
    else:
        print("\n[2/4] 跳过辅助特征...")
    
    # 训练
    print(f"\n[3/4] 训练Mamba Smoother...")
    mamba = MambaSmootherWrapper(
        n_classes=4,
        d_model=args.d_model,
        n_layers=args.n_layers,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    
    mamba.fit(y_train_proba, y_train, P_train, aux_train)
    
    # 评估
    print("\n[4/4] 评估模型...")
    y_pred = mamba.predict(y_test_proba, P_test, aux_test)
    
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    print("\n" + "="*70)
    print(f"  {args.exp_id} 测试集结果")
    print("="*70)
    print(classification_report(y_test, y_pred))
    print(f"\nMacro F1: {f1_macro:.4f}")
    print(f"Per-class F1: {f1_per_class}")
    
    # 保存模型
    model_path = output_dir / f'mamba_{args.exp_id}.pkl'
    joblib.dump(mamba, model_path)
    print(f"\n[OK] 模型已保存: {model_path}")
    
    print(f"\n{'='*70}")
    print(f"  [OK] 训练完成! Macro F1 = {f1_macro:.4f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
