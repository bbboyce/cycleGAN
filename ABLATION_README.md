# 🧪 CycleGAN 消融实验框架

## ✨ 功能概览

这个框架提供了完整的消融实验（Ablation Study）工具，用于系统地理解 CycleGAN 中不同损失函数组件的贡献。

### 📦 包含的文件

| 文件 | 说明 |
|------|------|
| `ablation_study.py` | 核心实验框架，支持自动化运行多个实验 |
| `ablation_configs.json` | 实验配置文件，定义所有消融实验 |
| `ABLATION_STUDY.md` | 详细的实验指南（100+ 页） |
| `ABLATION_STUDY_ANALYSIS.ipynb` | Jupyter notebook，用于结果分析与可视化 |
| `ablation_quickstart.sh` | 快速启动脚本 |
| `models/cycle_gan_ablation_model.py` | 增强的 CycleGAN 模型，支持灵活的损失配置 |

---

## 🚀 5 分钟快速开始

### 1. 生成配置

```bash
python ablation_study.py --gen-config
```

这会创建 `ablation_configs.json`，包含预设的消融实验。

### 2. 查看实验列表

```bash
python ablation_study.py --list-experiments
```

输出：
```
✅ Full CycleGAN (Baseline)
   All loss components enabled

1. No Cycle Loss
   Remove cycle consistency loss (lambda_A=0, lambda_B=0)

2. No Identity Loss
   Remove identity loss (lambda_identity=0)

3. Cycle Loss Only
   Only cycle loss, no GAN loss or identity loss

...
```

### 3. 运行所有实验

```bash
python ablation_study.py --dataroot ./datasets/handbags
```

### 4. 分析结果（可选）

```bash
jupyter notebook ABLATION_STUDY_ANALYSIS.ipynb
```

---

## 🧪 预设实验方案

### 方案 A: 基础消融（推荐）

测试三个主要损失函数的独立贡献：

| 实验 | 配置 | 目的 |
|------|------|------|
| Baseline | 完整模型 | 基准 |
| No Cycle Loss | λ_A=0, λ_B=0 | 测试循环一致性的重要性 |
| No Identity Loss | λ_identity=0 | 测试恒等映射的影响 |
| Cycle Loss Only | 仅循环损失 | 隔离循环损失效果 |

**运行时间**: ~40-60 小时（8×V100 GPU）

### 方案 B: 权重敏感性分析

研究超参数对性能的影响：

| 实验 | λ_A | λ_B | 目的 |
|------|-----|-----|------|
| Baseline | 10 | 10 | 基准 |
| Light Constraint | 5 | 5 | 减弱约束 |
| Medium Constraint | 10 | 10 | 标准约束 |
| Strong Constraint | 20 | 20 | 强约束 |

**运行时间**: ~30-45 小时（4×V100 GPU）

### 方案 C: 最小配置

快速验证模型可学性：

```bash
# 仅循环损失
python train.py --dataroot ./datasets/handbags \
  --name ablation_cycle_only \
  --model cycle_gan \
  --lambda_A 10 \
  --lambda_B 10 \
  --lambda_identity 0 \
  --n_epochs 50

# 仅 GAN 损失
python train.py --dataroot ./datasets/handbags \
  --name ablation_gan_only \
  --model cycle_gan \
  --lambda_A 0 \
  --lambda_B 0 \
  --lambda_identity 0 \
  --n_epochs 50
```

**运行时间**: ~5-10 小时（1×GPU）

---

## 📊 实验结果示例

### 预期发现

#### 1. 循环一致性的重要性 ⭐⭐⭐⭐⭐

**移除循环损失**：
- ❌ 严重的图像失真
- ❌ 无法保留内容
- ❌ 收敛不稳定
- **结论**: 循环损失是必要的

#### 2. 恒等映射的影响 ⭐⭐⭐

**移除恒等损失**：
- △ FID 略微增加
- ✓ 颜色转换更激进
- ✓ 细节更丰富
- **结论**: 依赖任务，灰度↔彩色时重要

#### 3. 权重平衡 ⭐⭐⭐⭐

**权重分析**：
- λ ↑↑ (λ_A=20): 保守但稳定
- λ = 10: 最优平衡
- λ ↓↓ (λ_A=5): 激进但失真

---

## 💻 使用示例

### 例 1: 快速验证

```bash
bash ablation_quickstart.sh
# 选择选项 1（运行所有实验）
```

### 例 2: 单个实验

```bash
python train.py \
  --dataroot ./datasets/horses2zebras \
  --name ablation_no_identity \
  --model cycle_gan \
  --lambda_identity 0.0 \
  --n_epochs 100
```

### 例 3: 自定义配置

编辑 `ablation_configs.json`:

```json
{
  "experiments": [
    {
      "name": "Custom Experiment",
      "lambda_A": 15.0,
      "lambda_B": 15.0,
      "lambda_identity": 0.3
    }
  ]
}
```

然后运行：

```bash
python ablation_study.py --dataroot ./datasets/facades
```

### 例 4: 使用增强模型

```bash
python train.py \
  --dataroot ./datasets/handbags \
  --name ablation_test \
  --model cycle_gan_ablation \
  --ablation_mode no_cycle \
  --n_epochs 50
```

**可用的消融模式**:
- `full` - 完整模型
- `no_cycle` - 移除循环损失
- `no_identity` - 移除恒等映射损失
- `no_gan` - 移除 GAN 损失
- `cycle_only` - 仅循环损失
- `cycle_and_gan` - 循环 + GAN
- `identity_only` - 仅恒等映射

---

## 📈 评估指标

### 1. 自动生成

```bash
# FID (Fréchet Inception Distance)
python -m pytorch_fid ./results/test_latest/images \
  ./datasets/handbags/testB
```

### 2. 在 Notebook 中计算

```python
# ABLATION_STUDY_ANALYSIS.ipynb 的第 6 节
from evaluation_metrics import compute_fid, compute_is, compute_lpips

fid_score = compute_fid(real_images_dir, fake_images_dir)
is_score = compute_is(fake_images_dir)
```

### 3. 自定义脚本

```python
import lpips
loss_fn = lpips.LPIPS(net='vgg')
distance = loss_fn(img1, img2)
```

---

## 📋 推荐实验流程

```
Week 1: 基础消融
├── Day 1: 生成配置，测试环境
├── Day 2-4: 运行基础实验 (Full, No Cycle, No Identity)
└── Day 5-7: 初步分析结果

Week 2: 参数敏感性
├── Day 1-3: 运行权重变体实验
├── Day 4-5: 详细分析
└── Day 6-7: 生成报告

Week 3: 定量评估 + 论文撰写
├── Day 1-2: 计算所有指标 (FID, IS, LPIPS)
├── Day 3-5: 生成对比图表
└── Day 6-7: 撰写论文相关章节
```

---

## 🔧 故障排除

### 问题 1: 运行很慢

**原因**: GPU 显存不足或 CPU 计算
**解决**:
```bash
python train.py --batch_size 1 --n_threads 2
```

### 问题 2: 训练不收敛

**原因**: 学习率不合适、损失权重极端
**解决**:
```bash
python train.py --lr 0.0001 --lambda_A 10 --lambda_B 10
```

### 问题 3: 内存溢出

**原因**: 图像太大或 batch size 过大
**解决**:
```bash
python train.py --crop_size 128 --batch_size 1 --load_size 143
```

### 问题 4: 找不到数据集

**原因**: 数据集路径不正确
**解决**:
```bash
# 下载数据集
bash datasets/download_cyclegan_dataset.sh handbags

# 验证路径
ls -la ./datasets/handbags/trainA
```

---

## 📚 关键论文与资源

### 原始论文
- **CycleGAN** (Zhu et al. 2017): https://arxiv.org/abs/1703.10593
- **Pix2Pix** (Isola et al. 2016): https://arxiv.org/abs/1611.05957

### 消融研究
- **Melis et al.** (2019): https://arxiv.org/abs/1902.05820
- **Survey of Ablation Studies** in Vision Domain

### 评估指标
- **FID**: Heusel et al. (2017)
- **IS**: Salimans et al. (2016)
- **LPIPS**: Zhang et al. (2018)

---

## 📝 输出文件结构

```
checkpoints/
├── full_cyclegan/
│   ├── loss_log.txt
│   ├── loss.pkl
│   └── ...
├── no_cycle_loss/
├── no_identity_loss/
└── ...

results_ablation/
├── ablation_report_*.md
├── training_curves_comparison.png
├── hyperparameter_comparison.png
├── ablation_comparison.csv
└── ...
```

---

## 🎯 最佳实践

✅ **DO**:
- 使用相同的随机种子便于重现
- 运行多次取平均值
- 详细记录每个实验
- 在相同硬件上运行对比
- 使用标准化的评估指标

❌ **DON'T**:
- 在不同数据集上对比结果
- 改变优化器或学习率
- 混用不同的数据预处理
- 忽略标准差或置信区间
- 过度解释单次运行的结果

---

## 📞 常见问题

**Q: 消融实验需要多长时间？**
A: 取决于数据集和硬件。单个 GPU 上，典型的 200 周期实验需要 2-4 小时。

**Q: 如何加速实验？**
A: 使用多 GPU、减小图像尺寸或使用更多并行工作进程。

**Q: 结果不稳定怎么办？**
A: 运行多次（3-5 次）取平均值，检查学习率和损失权重。

**Q: 能用小数据集测试吗？**
A: 可以，但用完整数据集训练至少一次以验证最终结果。

---

## 📞 获取帮助

- 📖 [详细指南](ABLATION_STUDY.md)
- 💻 [框架代码](ablation_study.py)
- 🔧 [模型代码](models/cycle_gan_ablation_model.py)
- 📊 [分析 Notebook](ABLATION_STUDY_ANALYSIS.ipynb)

---

**Created**: 2026-04-08  
**Framework Version**: 1.0  
**Maintained by**: CycleGAN Research Team

🎉 **Happy Ablation Study!**
