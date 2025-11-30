"""
Gait Biomarkers Computation & Evaluation
========================================
使用 Benchmark RF+HMM 模型识别步行段，计算并评估 Gait Biomarkers

执行: python experiments/gait_filter/compute_biomarkers.py
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 添加项目根目录到路径
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmark import train_test_split
from classifier import Classifier
from scipy import signal
from scipy.ndimage import median_filter

# ============================================================================
# 配置
# ============================================================================
N_JOBS = 8
WINDOW_SEC = 10  # 与 prepared_data 一致
SAMPLE_RATE = 30  # 假设重采样到 30Hz

# 输出路径
OUTPUT_DIR = ROOT / 'artifacts' / 'gait_biomarkers'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / f'execution_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

def log(msg):
    """打印并写入日志"""
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# ============================================================================
# Biomarker 计算函数
# ============================================================================

def compute_enmo(xyz):
    """计算 ENMO (mg)"""
    if xyz.ndim == 1:
        return max(0, np.linalg.norm(xyz) - 1) * 1000
    norm = np.linalg.norm(xyz, axis=1)
    enmo = np.maximum(norm - 1.0, 0) * 1000
    return enmo

def estimate_cadence_autocorr(v, sample_rate=30):
    """基于自相关估计步频"""
    if len(v) < 2 * sample_rate:
        return 0, 0
    
    v = median_filter(v, size=5)
    v = v - np.mean(v)
    
    max_lag = min(2 * sample_rate, len(v) - 1)
    acf = np.correlate(v, v, mode='full')
    acf = acf[len(acf)//2 : len(acf)//2 + max_lag]
    if acf[0] > 0:
        acf = acf / acf[0]
    
    try:
        peaks, props = signal.find_peaks(acf, prominence=0.1, distance=int(0.4*sample_rate))
        if len(peaks) > 0:
            step_period = peaks[0] / sample_rate
            cadence = 60 / step_period
            confidence = props['prominences'][0]
            return cadence, confidence
    except:
        pass
    return 0, 0

def detect_bouts_from_predictions(y_pred, P, window_sec=10, min_bout_sec=30):
    """从预测序列中提取步行 Bout"""
    bouts = []
    
    # 按参与者分组
    unique_participants = np.unique(P)
    
    for pid in unique_participants:
        mask = P == pid
        pred_p = y_pred[mask]
        
        # 找连续步行段
        is_walking = (pred_p == 'light') | (pred_p == 'moderate-vigorous')
        
        in_bout = False
        bout_start = 0
        
        for i, walking in enumerate(is_walking):
            if walking and not in_bout:
                bout_start = i
                in_bout = True
            elif not walking and in_bout:
                bout_end = i
                duration_sec = (bout_end - bout_start) * window_sec
                
                if duration_sec >= min_bout_sec:
                    bouts.append({
                        'participant': pid,
                        'start_idx': bout_start,
                        'end_idx': bout_end,
                        'duration_sec': duration_sec,
                        'n_windows': bout_end - bout_start
                    })
                in_bout = False
        
        # 处理末尾
        if in_bout:
            bout_end = len(is_walking)
            duration_sec = (bout_end - bout_start) * window_sec
            if duration_sec >= min_bout_sec:
                bouts.append({
                    'participant': pid,
                    'start_idx': bout_start,
                    'end_idx': bout_end,
                    'duration_sec': duration_sec,
                    'n_windows': bout_end - bout_start
                })
    
    return bouts

def compute_fragmentation_index(bouts, total_walking_time_sec):
    """G2.1: 碎片化指数"""
    if total_walking_time_sec <= 0:
        return 0
    
    short_bouts = [b for b in bouts if b['duration_sec'] <= 120]
    short_time = sum(b['duration_sec'] for b in short_bouts)
    short_ratio = short_time / total_walking_time_sec
    
    n_bouts_normalized = min(len(bouts) / 50, 1.0)
    
    fragmentation = 0.7 * short_ratio + 0.3 * n_bouts_normalized
    return fragmentation

def compute_long_bout_deficit(bouts, ref_count=3.0, ref_duration=2000):
    """G2.2: 长 Bout 缺陷分数"""
    long_bouts = [b for b in bouts if b['duration_sec'] >= 600]
    
    count = len(long_bouts)
    duration = sum(b['duration_sec'] for b in long_bouts)
    
    count_deficit = 1 - min(count / ref_count, 1.0)
    duration_deficit = 1 - min(duration / ref_duration, 1.0)
    
    score = (count_deficit + duration_deficit) / 2 * 100
    return score

def compute_participant_biomarkers(pid, bouts, y_pred, P, X_feats):
    """计算单个参与者的所有 biomarkers"""
    p_bouts = [b for b in bouts if b['participant'] == pid]
    p_mask = P == pid
    p_pred = y_pred[p_mask]
    p_feats = X_feats[p_mask]
    
    # 基础统计
    is_walking = (p_pred == 'light') | (p_pred == 'moderate-vigorous')
    total_walking_windows = is_walking.sum()
    total_walking_sec = total_walking_windows * WINDOW_SEC
    
    results = {
        'participant': pid,
        'n_windows': len(p_pred),
        'total_time_sec': len(p_pred) * WINDOW_SEC,
    }
    
    # G8.1: daily_brisk_minutes (使用 moderate-vigorous 作为 brisk)
    brisk_windows = (p_pred == 'moderate-vigorous').sum()
    results['gait.priority.daily_brisk_minutes'] = brisk_windows * WINDOW_SEC / 60
    
    # G8.2: max_continuous_bout
    if p_bouts:
        max_bout = max(b['duration_sec'] for b in p_bouts)
        results['gait.priority.max_continuous_bout_sec'] = max_bout
    else:
        results['gait.priority.max_continuous_bout_sec'] = 0
    
    # G2.1: fragmentation_index
    results['gait.bout.fragmentation_index'] = compute_fragmentation_index(p_bouts, total_walking_sec)
    
    # G2.2: long_bout_deficit
    results['gait.bout.long_deficit'] = compute_long_bout_deficit(p_bouts)
    
    # G8: bout 统计
    results['gait.bout.count'] = len(p_bouts)
    results['gait.bout.total_walking_min'] = total_walking_sec / 60
    
    if p_bouts:
        results['gait.bout.mean_duration_sec'] = np.mean([b['duration_sec'] for b in p_bouts])
        results['gait.bout.median_duration_sec'] = np.median([b['duration_sec'] for b in p_bouts])
        
        # 分类统计
        short = len([b for b in p_bouts if b['duration_sec'] <= 120])
        medium = len([b for b in p_bouts if 120 < b['duration_sec'] <= 600])
        long = len([b for b in p_bouts if b['duration_sec'] > 600])
        results['gait.bout.short_count'] = short
        results['gait.bout.medium_count'] = medium
        results['gait.bout.long_count'] = long
    else:
        results['gait.bout.mean_duration_sec'] = 0
        results['gait.bout.median_duration_sec'] = 0
        results['gait.bout.short_count'] = 0
        results['gait.bout.medium_count'] = 0
        results['gait.bout.long_count'] = 0
    
    # 活动强度特征 (从 X_feats 提取，假设第一列是 mean ENMO)
    # X_feats 的特征顺序需要确认，这里用统计量
    walking_feats = p_feats[is_walking] if is_walking.any() else None
    if walking_feats is not None and len(walking_feats) > 0:
        # 假设特征包含 avg (index 0), std (index 1)
        results['gait.intensity.mean_enmo'] = np.mean(walking_feats[:, 0]) if walking_feats.shape[1] > 0 else 0
        results['gait.intensity.std_enmo'] = np.mean(walking_feats[:, 1]) if walking_feats.shape[1] > 1 else 0
    else:
        results['gait.intensity.mean_enmo'] = 0
        results['gait.intensity.std_enmo'] = 0
    
    return results

# ============================================================================
# 主流程
# ============================================================================

def main():
    log("=" * 80)
    log("Gait Biomarkers Computation & Evaluation")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80)
    
    # 1. 加载数据
    log("\n[1/5] 加载数据...")
    X_feats = pd.read_pickle(ROOT / 'prepared_data/X_feats.pkl').values
    Y_walmsley = np.load(ROOT / 'prepared_data/Y_Walmsley2020.npy', allow_pickle=True)
    Y_anno = np.load(ROOT / 'prepared_data/Y_anno.npy', allow_pickle=True)
    P = np.load(ROOT / 'prepared_data/P.npy', allow_pickle=True)
    
    log(f"  - X_feats: {X_feats.shape}")
    log(f"  - Y_walmsley: {Y_walmsley.shape}")
    log(f"  - 参与者数: {len(np.unique(P))}")
    log(f"  - 标签分布: {pd.Series(Y_walmsley).value_counts().to_dict()}")
    
    # 2. 划分数据
    log("\n[2/5] 划分训练/测试集...")
    train_ids, test_ids = train_test_split(P)
    
    X_train, y_train, P_train = X_feats[train_ids], Y_walmsley[train_ids], P[train_ids]
    X_test, y_test, P_test = X_feats[test_ids], Y_walmsley[test_ids], P[test_ids]
    
    log(f"  - 训练集: {len(X_train)} windows, {len(np.unique(P_train))} participants")
    log(f"  - 测试集: {len(X_test)} windows, {len(np.unique(P_test))} participants")
    
    # 3. 训练 RF+HMM 模型 (Benchmark 最佳)
    log("\n[3/5] 训练 RF+HMM 模型...")
    model = Classifier('rf_hmm', verbose=0)
    model.fit(X_train, y_train, P_train)
    log("  - 模型训练完成")
    
    # 4. 预测
    log("\n[4/5] 预测步行状态...")
    y_pred_train = model.predict(X_train, P_train)
    y_pred_test = model.predict(X_test, P_test)
    
    # 统计预测结果
    train_walking = ((y_pred_train == 'light') | (y_pred_train == 'moderate-vigorous')).sum()
    test_walking = ((y_pred_test == 'light') | (y_pred_test == 'moderate-vigorous')).sum()
    
    log(f"  - 训练集步行窗口: {train_walking} / {len(y_pred_train)} ({train_walking/len(y_pred_train)*100:.2f}%)")
    log(f"  - 测试集步行窗口: {test_walking} / {len(y_pred_test)} ({test_walking/len(y_pred_test)*100:.2f}%)")
    
    # 5. 计算 Biomarkers
    log("\n[5/5] 计算 Gait Biomarkers...")
    
    # 检测 Bouts
    log("  - 检测步行 Bouts...")
    train_bouts = detect_bouts_from_predictions(y_pred_train, P_train)
    test_bouts = detect_bouts_from_predictions(y_pred_test, P_test)
    
    log(f"    训练集: {len(train_bouts)} bouts")
    log(f"    测试集: {len(test_bouts)} bouts")
    
    # 计算每个参与者的 biomarkers
    log("  - 计算参与者级 Biomarkers...")
    
    train_results = []
    for pid in np.unique(P_train):
        res = compute_participant_biomarkers(pid, train_bouts, y_pred_train, P_train, X_train)
        res['split'] = 'train'
        train_results.append(res)
    
    test_results = []
    for pid in np.unique(P_test):
        res = compute_participant_biomarkers(pid, test_bouts, y_pred_test, P_test, X_test)
        res['split'] = 'test'
        test_results.append(res)
    
    # 合并结果
    all_results = pd.DataFrame(train_results + test_results)
    
    # 保存结果
    output_file = OUTPUT_DIR / 'participant_biomarkers.csv'
    all_results.to_csv(output_file, index=False)
    log(f"  - 结果已保存: {output_file}")
    
    # ============================================================================
    # 评估报告
    # ============================================================================
    log("\n" + "=" * 80)
    log("BIOMARKER 统计摘要")
    log("=" * 80)
    
    biomarker_cols = [c for c in all_results.columns if c.startswith('gait.')]
    
    for col in biomarker_cols:
        train_vals = all_results[all_results['split'] == 'train'][col]
        test_vals = all_results[all_results['split'] == 'test'][col]
        
        log(f"\n{col}:")
        log(f"  训练集: mean={train_vals.mean():.3f}, std={train_vals.std():.3f}, "
            f"median={train_vals.median():.3f}, [min={train_vals.min():.3f}, max={train_vals.max():.3f}]")
        log(f"  测试集: mean={test_vals.mean():.3f}, std={test_vals.std():.3f}, "
            f"median={test_vals.median():.3f}, [min={test_vals.min():.3f}, max={test_vals.max():.3f}]")
    
    # Bout 分布
    log("\n" + "-" * 40)
    log("BOUT 分布统计")
    log("-" * 40)
    
    if train_bouts:
        train_durations = [b['duration_sec'] for b in train_bouts]
        log(f"\n训练集 Bout 时长 (秒):")
        log(f"  count={len(train_durations)}, mean={np.mean(train_durations):.1f}, "
            f"std={np.std(train_durations):.1f}, median={np.median(train_durations):.1f}")
        log(f"  分位数: 25%={np.percentile(train_durations, 25):.1f}, "
            f"75%={np.percentile(train_durations, 75):.1f}, "
            f"95%={np.percentile(train_durations, 95):.1f}")
    
    if test_bouts:
        test_durations = [b['duration_sec'] for b in test_bouts]
        log(f"\n测试集 Bout 时长 (秒):")
        log(f"  count={len(test_durations)}, mean={np.mean(test_durations):.1f}, "
            f"std={np.std(test_durations):.1f}, median={np.median(test_durations):.1f}")
        log(f"  分位数: 25%={np.percentile(test_durations, 25):.1f}, "
            f"75%={np.percentile(test_durations, 75):.1f}, "
            f"95%={np.percentile(test_durations, 95):.1f}")
    
    # 完成
    log("\n" + "=" * 80)
    log(f"完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"日志文件: {LOG_FILE}")
    log(f"结果文件: {output_file}")
    log("=" * 80)

if __name__ == '__main__':
    main()
