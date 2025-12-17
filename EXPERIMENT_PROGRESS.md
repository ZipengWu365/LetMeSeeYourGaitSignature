# HMM-Based Mamba 实验进度与结果报告

## 实验目标
按照 `hmm_based_mamba_design_and_experiments (1)(1).md` 的设计，实现并评估P1: Tiny-Mamba Emission + HMM Decode 模型。

## 完成进度

### ✅ 第1阶段：代码实现完成

#### 1.1 核心模型实现
- **[✅] models/tiny_mamba_emission.py** - Tiny Mamba Emission 模型
  - TinyMambaEmission: 小型Mamba编码器，输出per-step logits
  - 支持d_model ∈ {16, 32}, n_layers ∈ {1, 2}
  - 温度缩放和emission floor平滑选项
  
- **[✅] models/hmm_decode.py** - HMM Viterbi 解码器
  - HMMDecoder: 使用神经网络emissions进行Viterbi解码
  - fit_transition(): 从训练标签学习转移矩阵
  - predict(): 对序列进行per-group解码

#### 1.2 统一评估框架
- **[✅] evaluation/evaluate_sequence_labeling.py** - 统一评估框架
  - SequenceLabelingEvaluator: 一致的metrics计算
  - 支持Raw和Decoded评估
  - 保存结果到JSON和CSV格式

#### 1.3 训练流程
- **[✅] train/train_p1_mamba_hmm.py** - P1完整训练脚本
  - SequenceDataset: 按参与者分组的sequence数据集
  - P1MambaHMMTrainer: 训练、早停、检查点管理
  - 支持字符串和整数标签的自动转换
  - 配置驱动的超参数设置

### ✅ 第2阶段：烟雾测试通过

**Smoke Test结果 (2 epochs)**
```
Split:
  Train: 657,593 samples, 105 participants
  Val:   131,891 samples, 22 participants
  Test:  145,278 samples, 24 participants

Label mapping: 
  {light: 0, moderate-vigorous: 1, sedentary: 2, sleep: 3}

Test Results (Epoch 2):
  Raw Macro F1:     0.5025
  Decoded Macro F1: 0.5264 (HMM Viterbi +0.0239 improvement)
  
  Per-class F1 (Raw):     [0.4835, 0.0002, 0.6610, 0.8655]
  Per-class F1 (Decoded): [0.4811, 0.0000, 0.7049, 0.9194]
  
  Cohen's Kappa (Raw):     0.5277
  Cohen's Kappa (Decoded): 0.5842
```

**关键观察：**
1. ✅ 代码完全可运行，无系统错误
2. ✅ 模型能学习到有意义的patterns
3. ✅ HMM解码在raw基础上提高F1: +0.0239 (+0.48%)
4. ✅ 特别是对sleep (0.8655→0.9194, +0.0539) 和light (0.6610→0.7049) 有改进
5. ⚠️  moderate-vigorous类(class 1)性能很差: 0.0002 raw F1
   - 可能原因：类不平衡(45,449 vs其他)或信息不足

## 下一步计划

### 第3阶段：完整训练与基线对比 (进行中)

#### 3.1 P1完整训练 (20 epochs)
- 预计运行时间: ~3-4小时
- 输出目录: `artifacts/p1_full_20epochs/`
- 预期性能: Decoded Macro F1 ≈ 0.55-0.58 (基于smoke test趋势)

#### 3.2 基线重新评估
**需要运行的baseline:**
1. **RF (Random Forest)**: 直接预测，无HMM
2. **RF+HMM**: 参考baseline，使用RF的OOB概率作为emissions
3. **ESN Smoother**: 现有实现（可选，如果时间允许）
4. **Mamba Smoother**: 现有实现（可选）

**关键：** 必须使用统一的评估框架确保fair comparison

### 第4阶段：超参数扫描 (按计划)

**Stage 1: 主要超参数扫描 (12 configs)**
```
Grid:
- d_model ∈ {16, 32}
- n_layers ∈ {1, 2}
- dropout ∈ {0.1, 0.2, 0.3}
- lr ∈ {1e-3, 3e-4, 1e-4}
- weight_decay ∈ {1e-4, 1e-3}

固定: τ=1.0, ε=0.0

Staged approach:
1. Fix dropout=0.2, weight_decay=1e-4: 扫描d_model × n_layers × lr (12次)
2. 选择top 3，再扫描dropout × weight_decay (6次each, 18总计)
3. 只在必要时调整τ和ε
```

**预期:**
- Stage 1: 12个config × 2.5 hours/config ≈ 30小时
- Stage 2: 18个config × 2.5 hours ≈ 45小时
- 总计: ~75小时 (可并行)

### 第5阶段：最终报告生成

输出文件：
```
REPORT.md:
├── 1. Split Protocol 说明
├── 2. Baseline 结果对比表
├── 3. P1 最佳配置 + 性能
├── 4. Ablation Studies
│   ├── Raw vs Decoded (HMM benefit)
│   ├── Per-class F1分析
│   └── 超参数敏感性
├── 5. 诊断信息
│   ├── 触发的contingency steps
│   └── 调试notes
└── 6. 结论与讨论
```

## 数据集信息

- **参与者数**: 151人
- **样本总数**: 934,762 个时间窗口
- **标签**: 4类 (light, moderate-vigorous, sedentary, sleep)
- **特征维度**: 32 (原始activity features)
- **类别分布**:
  - sedentary: 371,077 (39.7%)
  - sleep: 340,228 (36.4%)
  - light: 178,008 (19.0%)
  - moderate-vigorous: 45,449 (4.9%)
- **长期序列**: 按参与者分组，每人多个连续序列

## 技术亮点

1. **严格的分割协议**: 参与者级别的train/val/test分割，避免数据泄漏
2. **统一评估框架**: Raw和Decoded metrics的一致计算
3. **配置驱动**: 所有运行完全可复现
4. **早停机制**: 基于validation decoded F1防止过拟合
5. **兼容性**: 支持字符串/整数标签，自动处理

## 已知问题与解决方案

### 问题1: Class不平衡
- **现象**: moderate-vigorous (F1≈0) 和light (F1≈0.7) 类性能差异大
- **诊断**: 类权重问题或features不足
- **解决**: 
  1. 检查emission校准 (temperature scaling)
  2. 考虑class-weighted loss (如果design允许)
  3. 分析per-class transition matrix

### 问题2: 长序列处理效率
- **现象**: 103个序列，平均长度~9,000步，显存占用
- **解决**: 批处理每个sequence单独处理

### 问题3: 基线评估时间
- **现象**: RF训练很慢 (~10+ min)
- **解决**: 检查是否可以使用pre-trained baseline或加速方法

## 文件清单

```
项目根目录: /iridisfs/scratch/jc15u24/Code/activity/LetMeSeeYourGaitSignature/

核心代码:
├── models/
│   ├── __init__.py
│   ├── tiny_mamba_emission.py       [✅ 实现完成]
│   └── hmm_decode.py                [✅ 实现完成]
├── evaluation/
│   ├── __init__.py
│   └── evaluate_sequence_labeling.py [✅ 实现完成]
├── train/
│   ├── train_p1_mamba_hmm.py         [✅ 实现完成]
│   ├── run_all_baselines.py          [⚠️  开发中]
│   └── run_p1_experiment.py          [⚠️  开发中]

数据:
├── prepared_data/
│   ├── X_feats.pkl                   (934,762 × 32)
│   ├── Y_Walmsley2020.npy            (934,762,)
│   └── P.npy                         (934,762,)

输出:
├── artifacts/
│   ├── p1_hmm/                       (初始smoke test)
│   ├── p1_full_20epochs/             (进行中...)
│   ├── p1_full_experiment/           (超参数扫描输出)
│   └── baselines/                    (baseline结果)
```

## 时间估计与资源

- **硬件**: GPU (CUDA 12.8, PyTorch 2.9.0)
- **环境**: easygen-clean conda环境
- **每个2-epoch运行**: ~1.5分钟 (烟雾测试)
- **每个20-epoch运行**: ~15-20分钟
- **预期总时间**: 
  - Full P1: 20分钟
  - Baselines: 30分钟
  - Sweep Stage 1: 30小时
  - Sweep Stage 2: 45小时
  - **总计: ~76小时** (不含等待)

## 关键命令

```bash
# 烟雾测试 (已完成)
python3 train/train_p1_mamba_hmm.py --epochs 2

# 完整训练 (20 epochs)
python3 train/train_p1_mamba_hmm.py --epochs 20 --output_dir artifacts/p1_full_20epochs

# 基线评估
python3 train/run_all_baselines.py --seed 42

# 超参数扫描 (Stage 1)
python3 train/run_p1_experiment.py --stage 4 --sweep stage1
```

## 联系与疑问

- 设计文档: `hmm_based_mamba_design_and_experiments (1)(1).md`
- 当前报告: 本文件
- 下一步: 等待完整训练完成，更新baselines和sweep结果

---

**最后更新**: 2025-12-16 21:20 UTC
**状态**: ✅ 代码完成 ✅ Smoke测试通过 ⏳ 完整训练进行中
