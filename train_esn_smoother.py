#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESN (Echo State Network) Smoother 训练脚本 - CUDA加速版
使用PyTorch实现Reservoir Computing，支持GPU加速
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn as nn


def get_device():
    """获取最佳可用设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  [OK] 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"       VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  [INFO] 使用CPU (ESN在CPU上也很快)")
    return device


class ESNSmootherCUDA:
    """Echo State Network Smoother - PyTorch CUDA加速版"""
    
    def __init__(
        self,
        n_classes=4,
        n_reservoir=800,
        spectral_radius=0.9,
        sparsity=0.9,
        input_scaling=1.0,
        ridge_alpha=1e-6,
        random_state=42,
        device=None,
    ):
        self.n_classes = n_classes
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.ridge_alpha = ridge_alpha
        self.device = device or get_device()
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # 初始化权重 (先在CPU创建，然后移到GPU)
        self.W_in = None
        self.W_res = None
        self.W_out = None
        self.input_dim = None
    
    def _init_weights(self, input_dim):
        """初始化ESN权重矩阵"""
        self.input_dim = input_dim
        
        # 输入权重
        self.W_in = torch.randn(self.n_reservoir, input_dim, device=self.device) * self.input_scaling
        
        # Reservoir权重: 稀疏随机矩阵
        W_res = torch.randn(self.n_reservoir, self.n_reservoir, device=self.device)
        # 稀疏化
        mask = torch.rand(self.n_reservoir, self.n_reservoir, device=self.device) > self.sparsity
        W_res = W_res * mask.float()
        
        # 调整spectral radius
        eigenvalues = torch.linalg.eigvals(W_res)
        max_eigenvalue = torch.max(torch.abs(eigenvalues)).item()
        if max_eigenvalue > 0:
            W_res = W_res * (self.spectral_radius / max_eigenvalue)
        
        self.W_res = W_res
    
    def _run_reservoir(self, inputs):
        """运行reservoir，收集状态 - GPU加速"""
        seq_len = inputs.shape[0]
        states = torch.zeros(seq_len, self.n_reservoir, device=self.device)
        
        state = torch.zeros(self.n_reservoir, device=self.device)
        
        for t in range(seq_len):
            # state = tanh(W_in @ input + W_res @ state)
            state = torch.tanh(
                self.W_in @ inputs[t] + self.W_res @ state
            )
            states[t] = state
        
        return states
    
    def fit(self, y_pred_proba, y_true, groups, aux_features=None):
        """训练ESN smoother - GPU加速"""
        print(f"\n[*] 训练ESN Smoother (CUDA: {self.device.type == 'cuda'})")
        
        # 准备输入
        if aux_features is not None:
            inputs = np.hstack([y_pred_proba, aux_features])
        else:
            inputs = y_pred_proba
        
        # 初始化权重
        self._init_weights(inputs.shape[1])
        
        # 收集所有reservoir状态
        all_states = []
        all_targets = []
        
        unique_groups = np.unique(groups)
        print(f"  Processing {len(unique_groups)} participants on {self.device}...")
        
        for i, g in enumerate(unique_groups):
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i+1}/{len(unique_groups)}")
            
            mask = groups == g
            seq_inputs = torch.FloatTensor(inputs[mask]).to(self.device)
            seq_targets = y_true[mask]
            
            # 运行reservoir (GPU加速)
            states = self._run_reservoir(seq_inputs)
            
            # 移回CPU用于Ridge回归
            all_states.append(states.cpu().numpy())
            all_targets.append(seq_targets)
        
        # 合并所有状态
        X_states = np.vstack(all_states)
        y_targets = np.concatenate(all_targets)
        
        print(f"  Reservoir states: {X_states.shape}")
        
        # 使用Ridge回归训练输出层 (CPU，因为sklearn不支持GPU)
        # 但可以用PyTorch实现GPU版Ridge
        print("  Training output layer (Ridge regression)...")
        
        # PyTorch Ridge回归 (GPU加速)
        X_tensor = torch.FloatTensor(X_states).to(self.device)
        
        # One-hot编码目标
        y_onehot = np.zeros((len(y_targets), self.n_classes))
        y_onehot[np.arange(len(y_targets)), y_targets] = 1
        y_tensor = torch.FloatTensor(y_onehot).to(self.device)
        
        # Ridge回归闭式解: W = (X^T X + αI)^{-1} X^T y
        XtX = X_tensor.T @ X_tensor
        XtX += self.ridge_alpha * torch.eye(self.n_reservoir, device=self.device)
        Xty = X_tensor.T @ y_tensor
        
        self.W_out = torch.linalg.solve(XtX, Xty)
        
        print("  [OK] 训练完成")
    
    def predict(self, y_pred_proba, groups, aux_features=None):
        """预测 - GPU加速"""
        if aux_features is not None:
            inputs = np.hstack([y_pred_proba, aux_features])
        else:
            inputs = y_pred_proba
        
        all_preds = []
        unique_groups = np.unique(groups)
        
        for g in unique_groups:
            mask = groups == g
            seq_inputs = torch.FloatTensor(inputs[mask]).to(self.device)
            
            # 运行reservoir
            states = self._run_reservoir(seq_inputs)
            
            # 计算输出
            logits = states @ self.W_out
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
    parser = argparse.ArgumentParser(description='训练ESN Smoother (CUDA加速)')
    parser.add_argument('--exp_id', type=str, required=True, help='实验ID')
    parser.add_argument('--n_reservoir', type=int, default=800, help='Reservoir大小')
    parser.add_argument('--spectral_radius', type=float, default=0.9, help='谱半径')
    parser.add_argument('--ridge_alpha', type=float, default=1e-6, help='Ridge正则化')
    parser.add_argument('--aux_features', type=str, default='enmo',
                       choices=['none', 'enmo', 'full'], help='辅助特征')
    parser.add_argument('--data_dir', type=str, default='./prepared_data')
    parser.add_argument('--output_dir', type=str, default='./models')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  训练ESN Smoother: {args.exp_id} (CUDA加速)")
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
    print(f"\n[3/4] 训练ESN Smoother...")
    esn = ESNSmootherCUDA(
        n_classes=4,
        n_reservoir=args.n_reservoir,
        spectral_radius=args.spectral_radius,
        ridge_alpha=args.ridge_alpha,
        device=device,
    )
    
    esn.fit(y_train_proba, y_train, P_train, aux_train)
    
    # 评估
    print("\n[4/4] 评估模型...")
    y_pred = esn.predict(y_test_proba, P_test, aux_test)
    
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    print("\n" + "="*70)
    print(f"  {args.exp_id} 测试集结果")
    print("="*70)
    print(classification_report(y_test, y_pred))
    print(f"\nMacro F1: {f1_macro:.4f}")
    print(f"Per-class F1: {f1_per_class}")
    
    # 保存模型
    model_path = output_dir / f'esn_{args.exp_id}.pkl'
    joblib.dump(esn, model_path)
    print(f"\n[OK] 模型已保存: {model_path}")
    
    print(f"\n{'='*70}")
    print(f"  [OK] 训练完成! Macro F1 = {f1_macro:.4f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
