# CycleGAN 消融实验（Ablation Study）指南

## 📋 概述

消融实验是一种系统的方法来理解模型中每个组件对最终性能的贡献。在 CycleGAN 中，我们将测试以下损失函数组件：

1. **GAN 损失** (Generator Adversarial Loss)
   - 使生成图像看起来更真实
   - 公式：`L_GAN(G, D, A, B) = E[logD(B)] + E[log(1-D(G(A)))]`

2. **循环一致性损失** (Cycle Consistency Loss)
   - 确保 A→B→A 和 B→A→B 的循环映射能恢复原始图像
   - 公式：`L_cycle(G, F) = ||F(G(A)) - A|| + ||G(F(B)) - B||`

3. **恒等映射损失** (Identity Loss)
   - 在两个域颜色接近时，保持图像原样
   - 公式：`L_identity = ||G(B) - B|| + ||F(A) - A||`

---

## 🚀 快速开始

### 1. 查看可用的实验配置

```bash
cd /Users/leon/Desktop/cycleGAN
python ablation_study.py --list-experiments
```

### 2. 生成默认配置文件

```bash
python ablation_study.py --gen-config
```

这会创建 `ablation_configs.json` 文件，包含：
- ✅ **Full CycleGAN (Baseline)** - 所有损失函数都启用
- ❌ **No Cycle Loss** - 移除循环一致性损失
- ❌ **No Identity Loss** - 移除恒等映射损失
- ✅ **Cycle Loss Only** - 仅保留循环一致性损失
- 📊 **Half/Double Cycle Weight** - 改变循环损失权重

---

## 🧪 实验配置详解

### 配置文件结构

```json
{
  "base_experiment": {
    "name": "Full CycleGAN (Baseline)",
    "lambda_A": 10.0,        // A→B→A 循环损失权重
    "lambda_B": 10.0,        // B→A→B 循环损失权重
    "lambda_identity": 0.5   // 恒等映射损失权重
  },
  "experiments": [...]
}
```

### 关键超参数说明

| 参数 | 含义 | 典型值 | 影响 |
|------|------|--------|------|
| `lambda_A` | A→B→A 循环损失权重 | 10.0 | ↑ 更好的纹理保留，速度慢 |
| `lambda_B` | B→A→B 循环损失权重 | 10.0 | ↑ 更好的结构保留，可能过度约束 |
| `lambda_identity` | 恒等映射损失权重 | 0.5 | ↑ 保留颜色，↓ 更多变换 |

---

## 📊 预设实验方案

### 方案 1：基础对比
```
实验1: Full CycleGAN (baseline)
实验2: No Cycle Loss
实验3: No Identity Loss
```

**预期结果：**
- 无循环损失：图像失真明显，无法准确转换
- 无恒等损失：在相近域之间转换时会改变颜色
- 基础模型：平衡效果

### 方案 2：损失权重敏感性
```
实验1: Full (λ_A=10, λ_B=10)
实验2: Half (λ_A=5, λ_B=5)
实验3: Double (λ_A=20, λ_B=20)
实验4: Cycle Only
```

**预期结果：**
- 权重↑：更好的一致性，但可能欠饱和
- 权重↓：转换更激进，但可能失真
- 仅循环：纹理保留好，但缺乏现实感

### 方案 3：最小化配置
```
实验1: Full (All losses)
实验2: Cycle + GAN (No identity)
实验3: Cycle Only
实验4: GAN Only (Unlikely to work)
```

**预期结果：**
- 循环+GAN：接近完整模型
- 仅循环：保守但稳定
- 仅GAN：可能不稳定

---

## 💻 运行实验

### 方法 1：运行所有实验

```bash
python ablation_study.py --dataroot ./datasets/handbags
```

### 方法 2：使用自定义配置

编辑 `ablation_configs.json`，然后运行：

```bash
python ablation_study.py --config ablation_configs.json --dataroot ./datasets/horses2zebras
```

### 方法 3：使用修改后的模型

在 `train.py` 中使用新模型：

```bash
# 使用消融模型训练
python train.py \
  --dataroot ./datasets/handbags \
  --name ablation_no_cycle \
  --model cycle_gan_ablation \
  --ablation_mode no_cycle \
  --lambda_A 0.0 \
  --lambda_B 0.0 \
  --n_epochs 200
```

### 可用的消融模式

```python
--ablation_mode full              # 完整模型（默认）
--ablation_mode no_cycle         # 移除循环损失
--ablation_mode no_identity      # 移除恒等映射损失
--ablation_mode no_gan           # 移除 GAN 损失
--ablation_mode cycle_only       # 仅保留循环损失
--ablation_mode cycle_and_gan    # 循环+GAN，无恒等映射
--ablation_mode identity_only    # 仅保留恒等映射损失
```

---

## 📈 评估指标

### 1. 定量指标

#### FID (Fréchet Inception Distance)
```bash
python -m pytorch_fid path/to/real_images path/to/generated_images
```
- 越低越好
- 衡量生成图像分布与真实图像分布的距离

#### IS (Inception Score)
- 衡量图像质量和多样性
- 越高越好

#### LPIPS (Learned Perceptual Image Patch Similarity)
```bash
# 安装：pip install lpips
python -c "
import lpips
loss_fn = lpips.LPIPS(net='vgg')
# 计算距离
"
```

### 2. 定性指标

- **视觉质量**：生成图像看起来有多真实
- **内容保留**：原始图像的内容是否得到保留
- **风格转换**：目标域的特征是否转移
- **一致性**：多次生成相同输入的结果是否一致

### 3. 训练指标

监测以下损失曲线：
- `loss_G`：生成器总损失
- `loss_D_A` / `loss_D_B`：判别器损失
- `loss_cycle_A` / `loss_cycle_B`：循环一致性损失
- `loss_idt_A` / `loss_idt_B`：恒等映射损失

---

## 📊 结果分析

### 生成报告

```bash
# 自动生成 Markdown 报告
python ablation_study.py --dataroot ./datasets/handbags
```

输出位置：`./results_ablation/ablation_report_*.md`

### 可视化结果

```python
from ablation_study import AblationStudy

study = AblationStudy()
study.visualize_results()
```

生成柱状图对比不同实验的超参数设置。

---

## 🔍 常见观察

### 循环一致性损失的重要性
- **移除循环损失后**：图像会发生意想不到的转换，甚至失真
- **结论**：循环一致性是 CycleGAN 的核心，不能移除

### 恒等映射损失的影响
- **移除恒等损失后**：图像颜色会改变，即使源和目标域相似
- **结论**：对于颜色域转换（如灰度→彩色）很重要

### GAN 损失的作用
- **移除 GAN 损失后**：图像纹理生成差，但保持结构
- **结论**：GAN 损失提供现实感，循环损失提供内容保留

### 权重平衡
- **权重全为 0**：不会学习任何东西
- **权重不平衡**：可能某一部分损失主导训练
- **建议**：从默认值（λ_A=10, λ_B=10, λ_identity=0.5）开始

---

## 📝 记录实验

创建实验日志文件 `ABLATION_LOG.md`：

```markdown
# 消融实验日志

## 实验 1：Full CycleGAN (Baseline)
- 日期：2026-04-08
- 数据集：handbags
- 配置：λ_A=10, λ_B=10, λ_identity=0.5
- 最终 FID：45.3
- 观察：基准模型性能

## 实验 2：No Cycle Loss
- 日期：2026-04-08
- 数据集：handbags
- 配置：λ_A=0, λ_B=0, λ_identity=0.5
- 最终 FID：89.2（差）
- 观察：图像严重失真，模型无法收敛

## 实验 3：No Identity Loss
- 日期：2026-04-08
- 数据集：handbags
- 配置：λ_A=10, λ_B=10, λ_identity=0.0
- 最终 FID：48.1
- 观察：FID 略有增加，颜色转换更激进
```

---

## 💡 最佳实践

### 1. 系统方法
- 逐次移除一个组件
- 保持其他参数不变
- 使用相同的随机种子便于重现

### 2. 充分训练
- 每个实验运行足够长的周期（200+ 轮）
- 使用相同的数据集和预处理
- 在相同的硬件上运行

### 3. 多次运行
- 由于深度学习的随机性，推荐运行 3-5 次
- 取平均值和标准差
- 报告置信区间

### 4. 控制变量
- 修改实验：一次改变一个参数
- 交叉组合实验：系统测试多个参数

---

## 🐛 常见问题

### Q1：为什么某个实验没有收敛？
**A:** 可能是：
- 损失权重设置过高或过低
- 学习率不适合新配置
- 数据集不适合（缺少某个域）

**解决方案**：
```bash
python train.py ... --lr 0.0001 --n_epochs 300
```

### Q2：如何比较不同实验的结果？
**A:** 使用标准化的评估指标：
- 在相同的测试集上评估
- 使用相同的指标（FID、LPIPS 等）
- 绘制损失曲线对比

### Q3：实验需要多久？
**A:** 取决于：
- 数据集大小
- GPU 性能
- 训练周期数

**典型时间**（一个 GPU）：
- 100 周期：2-4 小时
- 200 周期：4-8 小时

---

## 📚 参考文献

1. **CycleGAN 原始论文**
   - Zhu et al. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
   - https://arxiv.org/abs/1703.10593

2. **消融研究指南**
   - Melis et al. "Ablation Study"
   - https://arxiv.org/abs/1902.05820

3. **评估指标**
   - FID: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
   - IS: Salimans et al., "Improved Techniques for Training GANs"
   - LPIPS: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"

---

## 📞 获取帮助

如遇问题，请检查：
1. ✅ 数据集路径是否正确
2. ✅ GPU 内存是否充足
3. ✅ 所有依赖包是否已安装
4. ✅ 模型参数是否合理

```bash
# 检查环境
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

---

**Happy Ablation Study! 🎉**
