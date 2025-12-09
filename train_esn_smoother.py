#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESN (Echo State Network) Smoother 训练脚本
使用Reservoir Computing进行时序平滑
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, classification_report
from scipy import sparse

class ESNSmoother:
    """Echo State Network Smoother for temporal smoothing"""
    
    def __init__(
        self,
        n_classes=4,
        n_reservoir=800,
        spectral_radius=0.9,
        sparsity=0.9,
        input_scaling=1.0,
        ridge_alpha=1e-6,
        random_state=42,
    ):
        self.n_classes = n_classes
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.ridge_alpha = ridge_alpha
        
        np.random.seed(random_state)
        
        # 初始化固定权重
        self.W_in = np.random.randn(n_reservoir, n_classes) * input_scaling
        
        # Reservoir权重: 稀疏随机矩阵
        self.W_res = sparse.random(
            n_reservoir, n_reservoir, 
            density=1-sparsity,
            random_state=random_state
        ).toarray()
        
        # 调整spectral radius
        eigenvalues = np.linalg.eigvals(self.W_res)
        self.W_res *= spectral_radius / np.max(np.abs(eigenvalues))
        
        self.ridge = None
    
    def fit(self, y_pred_proba, y_true, groups, aux_features=None):
        """训练ESN smoother"""
        
        # 拼接辅助特征
        if aux_features is not None:
            aux_dim = aux_features.shape[1]
            W_in_aux = np.random.randn(self.n_reservoir, aux_dim) * 0.5
            self.W_in = np.hstack([self.W_in, W_in_aux])
            inputs = np.hstack([y_pred_proba, aux_features])
        else:
            inputs = y_pred_proba
        
        # 收集reservoir状态 (按participant分组)
        all_states = []
        all_targets = []
        
        unique_groups = np.unique(groups)
        print(f"  Processing {len(unique_groups)} participants...")
        
        for i, g in enumerate(unique_groups):
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i+1}/{len(unique_groups)}")
            
            mask = groups == g
            seq_inputs = inputs[mask]
            seq_targets = y_true[mask]
            
            # 运行reservoir
            states = self._run_reservoir(seq_inputs)
            all_states.append(states)
            all_targets.append(seq_targets)
        
        # 合并所有状态
        X_train = np.vstack(all_states)
        y_train = np.concatenate(all_targets).astype(np.int32)  # 确保是整数类型
        
        # One-hot编码
        y_train_onehot = np.eye(self.n_classes)[y_train]
        
        # Ridge回归训练输出层
        print(f"  Training Ridge regression on {X_train.shape[0]} samples...")
        self.ridge = Ridge(alpha=self.ridge_alpha)
        self.ridge.fit(X_train, y_train_onehot)
        
        print(f"  [OK] ESN training complete!")
    
    def predict(self, y_pred_proba, groups, aux_features=None):
        """预测"""
        if aux_features is not None:
            inputs = np.hstack([y_pred_proba, aux_features])
        else:
            inputs = y_pred_proba
        
        all_predictions = []
        unique_groups = np.unique(groups)
        
        for g in unique_groups:
            mask = groups == g
            seq_inputs = inputs[mask]
            
            # 运行reservoir
            states = self._run_reservoir(seq_inputs)
            
            # 预测
            proba = self.ridge.predict(states)
            preds = np.argmax(proba, axis=1)
            
            all_predictions.append(preds)
        
        return np.concatenate(all_predictions)
    
    def _run_reservoir(self, inputs):
        """运行reservoir动态"""
        T = len(inputs)
        states = np.zeros((T, self.n_reservoir))
        h = np.zeros(self.n_reservoir)
        
        for t in range(T):
            h = np.tanh(self.W_in @ inputs[t] + self.W_res @ h)
            states[t] = h
        
        return states

def compute_auxiliary_features(X_raw, feature_type='enmo'):
    """从原始信号提取辅助特征"""
    if feature_type == 'none':
        return None
    
    # ENMO统计
    enmo = np.linalg.norm(X_raw, axis=2) - 1.0
    enmo_mean = enmo.mean(axis=1)
    enmo_std = enmo.std(axis=1)
    enmo_max = enmo.max(axis=1)
    
    if feature_type == 'enmo':
        return np.column_stack([enmo_mean, enmo_std, enmo_max])
    
    elif feature_type == 'full':
        # 主频
        from scipy.fft import rfft, rfftfreq
        fft_vals = np.abs(rfft(enmo, axis=1))
        freqs = rfftfreq(1000, 1/100)
        dominant_freq = freqs[fft_vals.argmax(axis=1)]
        
        # 姿态角度
        gravity_vec = X_raw.mean(axis=1)
        postural_angle = np.arctan2(
            np.linalg.norm(gravity_vec[:, :2], axis=1),
            gravity_vec[:, 2]
        )
        
        # Jerk
        jerk = np.linalg.norm(np.diff(X_raw, axis=1), axis=2).mean(axis=1)
        
        return np.column_stack([
            enmo_mean, enmo_std, enmo_max,
            dominant_freq, postural_angle, jerk
        ])

def main():
    parser = argparse.ArgumentParser(description='训练ESN Smoother')
    parser.add_argument('--exp_id', type=str, required=True, help='实验ID')
    parser.add_argument('--n_reservoir', type=int, default=800, help='Reservoir大小')
    parser.add_argument('--spectral_radius', type=float, default=0.9, help='谱半径')
    parser.add_argument('--aux_features', type=str, default='enmo',
                       choices=['none', 'enmo', 'full'], help='辅助特征类型')
    parser.add_argument('--ridge_alpha', type=float, default=1e-6, help='Ridge正则化')
    parser.add_argument('--data_dir', type=str, default='../../prepared_data')
    parser.add_argument('--output_dir', type=str, default='./models')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  训练ESN Smoother: {args.exp_id}")
    print(f"{'='*70}\n")
    print(f"  Reservoir size: {args.n_reservoir}")
    print(f"  Spectral radius: {args.spectral_radius}")
    print(f"  Auxiliary features: {args.aux_features}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    data_dir = Path(args.data_dir)
    
    # 加载划分信息
    split_info = joblib.load(data_dir / 'train_test_split.pkl')
    train_mask = split_info['train_mask']
    test_mask = split_info['test_mask']
    
    # 加载RF概率输出
    y_train_proba = np.load(data_dir / 'y_train_proba_rf.npy')
    y_test_proba = np.load(data_dir / 'y_test_proba_rf.npy')
    
    # 加载真实标签和groups
    Y = np.load(data_dir / 'Y_Walmsley2020.npy')
    P = np.load(data_dir / 'P.npy')
    
    y_train = Y[train_mask]
    y_test = Y[test_mask]
    P_train = P[train_mask]
    P_test = P[test_mask]
    
    print(f"  训练集: {y_train_proba.shape}")
    print(f"  测试集: {y_test_proba.shape}")
    
    # 加载辅助特征
    aux_train = None
    aux_test = None
    if args.aux_features != 'none':
        print(f"\n[2/4] 计算辅助特征 ({args.aux_features})...")
        X_raw = np.load(data_dir / 'X.npy')
        
        X_train_raw = X_raw[train_mask]
        X_test_raw = X_raw[test_mask]
        
        aux_train = compute_auxiliary_features(X_train_raw, args.aux_features)
        aux_test = compute_auxiliary_features(X_test_raw, args.aux_features)
        
        print(f"  辅助特征维度: {aux_train.shape[1]}")
    else:
        print("\n[2/4] 跳过辅助特征...")
    
    # 训练ESN
    print(f"\n[3/4] 训练ESN...")
    esn = ESNSmoother(
        n_reservoir=args.n_reservoir,
        spectral_radius=args.spectral_radius,
        ridge_alpha=args.ridge_alpha,
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
