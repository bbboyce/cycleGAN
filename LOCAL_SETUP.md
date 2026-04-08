# 🚀 本地环境配置指南

这个指南将帮助您在本地机器上配置并运行CycleGAN和pix2pix模型。

## 📋 系统要求

- **操作系统**: macOS, Linux 或 Windows
- **Python**: 3.7+
- **GPU** (推荐但非必需): NVIDIA GPU with CUDA 11.8+ 或 Apple Silicon (M系列)
- **磁盘空间**: 至少 10GB (包括数据集和模型)
- **内存**: 至少 8GB RAM (16GB 推荐)

## ✅ 第一步：环境配置

### 1.1 创建虚拟环境

```bash
# 进入项目目录
cd /Users/leon/Desktop/cycleGAN

# 创建虚拟环境 (如果还没有)
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate  # Windows
```

### 1.2 安装依赖

```bash
# 升级pip
pip install --upgrade pip

# 方法1: 使用requirements.txt
pip install -r requirements.txt

# 方法2: 使用package安装
pip install -e .

# 方法3: 使用conda (如果已安装conda)
conda env create -f environment.yml
conda activate pytorch-img2img
```

### 1.3 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
python -c "from PIL import Image; print('PIL: OK')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

## 📦 第二步：下载数据集

### 2.1 可用的预处理数据集

#### CycleGAN 数据集

```bash
# 下载CycleGAN数据集 (选择一个)
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
bash ./datasets/download_cyclegan_dataset.sh apple2orange
bash ./datasets/download_cyclegan_dataset.sh summer2winter_yosemite
bash ./datasets/download_cyclegan_dataset.sh monet_photo
bash ./datasets/download_cyclegan_dataset.sh cezanne
bash ./datasets/download_cyclegan_dataset.sh ukiyoe
bash ./datasets/download_cyclegan_dataset.sh vangogh
bash ./datasets/download_cyclegan_dataset.sh maps
bash ./datasets/download_cyclegan_dataset.sh cityscapes
bash ./datasets/download_cyclegan_dataset.sh facades
bash ./datasets/download_cyclegan_dataset.sh iphone2dslr_flower
```

#### pix2pix 数据集

```bash
# 下载pix2pix数据集 (选择一个)
bash ./datasets/download_pix2pix_dataset.sh facades
bash ./datasets/download_pix2pix_dataset.sh cityscapes
bash ./datasets/download_pix2pix_dataset.sh maps
bash ./datasets/download_pix2pix_dataset.sh edges2handbags
bash ./datasets/download_pix2pix_dataset.sh edges2shoes
bash ./datasets/download_pix2pix_dataset.sh night2day
bash ./datasets/download_pix2pix_dataset.sh BtoA
```

### 2.2 快速开始 (推荐)

```bash
# 对于CycleGAN，下载最小的数据集
bash ./datasets/download_cyclegan_dataset.sh horse2zebra

# 对于pix2pix，下载较小的数据集
bash ./datasets/download_pix2pix_dataset.sh facades
```

### 2.3 数据集位置

下载的数据集将位于：
```
./datasets/
├── horse2zebra/
│   ├── trainA/    # 源域训练图像
│   ├── trainB/    # 目标域训练图像
│   ├── testA/     # 源域测试图像
│   └── testB/     # 目标域测试图像
└── facades/
    ├── train/
    └── test/
```

## 🎯 第三步：下载预训练模型 (可选)

### 3.1 CycleGAN 预训练模型

```bash
# 下载预训练模型 (选择一个)
bash ./scripts/download_cyclegan_model.sh horse2zebra
bash ./scripts/download_cyclegan_model.sh apple2orange
bash ./scripts/download_cyclegan_model.sh vangogh
bash ./scripts/download_cyclegan_model.sh monet_photo
```

### 3.2 pix2pix 预训练模型

```bash
# 下载预训练模型 (选择一个)
bash ./scripts/download_pix2pix_model.sh facades_label2photo
bash ./scripts/download_pix2pix_model.sh sat2map
bash ./scripts/download_pix2pix_model.sh map2sat
bash ./scripts/download_pix2pix_model.sh edges2shoes
bash ./scripts/download_pix2pix_model.sh day2night
```

### 3.3 预训练模型位置

```
./checkpoints/
├── horse2zebra_pretrained/
│   └── latest_net_G.pt
├── facades_label2photo_pretrained/
│   └── latest_net_G.pt
└── ...
```

## 🏋️ 第四步：训练模型

### 4.1 基本训练命令

**CycleGAN:**
```bash
python train.py \
  --dataroot ./datasets/horse2zebra \
  --name horse2zebra_cyclegan \
  --model cycle_gan \
  --display_freq 100 \
  --print_freq 100
```

**pix2pix:**
```bash
python train.py \
  --dataroot ./datasets/facades \
  --name facades_pix2pix \
  --model pix2pix \
  --direction BtoA \
  --display_freq 100 \
  --print_freq 100
```

### 4.2 常用训练参数

```bash
# 批次大小
--batch_size 1          # 默认1 (GPU内存不足时)
--batch_size 4          # 更快的训练 (需要更多内存)

# 学习率计划
--n_epochs 200          # 总训练周期数 (默认200)
--n_epochs_decay 200    # 学习率衰减周期 (默认200)

# 显示和保存频率
--display_freq 100      # 每100个迭代显示一次
--print_freq 100        # 每100个迭代打印一次
--save_latest_freq 5000 # 每5000个迭代保存最新模型

# 多GPU训练
--gpu_ids 0,1           # 使用GPU 0和1

# Weights & Biases 日志
--use_wandb             # 启用wandb日志
```

### 4.3 多GPU DDP训练 (推荐用于大规模训练)

```bash
# 使用torchrun进行DDP训练
torchrun --nproc_per_node=4 train.py \
  --dataroot ./datasets/horse2zebra \
  --name horse2zebra_cyclegan_ddp \
  --model cycle_gan
```

## 🧪 第五步：测试模型

### 5.1 使用预训练模型进行测试

**测试CycleGAN模型:**
```bash
python test.py \
  --dataroot ./datasets/horse2zebra \
  --name horse2zebra_pretrained \
  --model cycle_gan
```

**测试pix2pix模型:**
```bash
python test.py \
  --dataroot ./datasets/facades \
  --name facades_label2photo_pretrained \
  --model pix2pix \
  --direction BtoA
```

### 5.2 测试单张图像

```bash
# 准备单张图像到文件夹，例如 ./test_images/
mkdir -p ./test_images

# 将图像放入 ./test_images/

# 运行测试
python test.py \
  --dataroot ./test_images \
  --name facades_label2photo_pretrained \
  --model test \
  --no_dropout
```

### 5.3 查看测试结果

```bash
# 结果保存位置
./results/{model_name}/test_latest/

# 查看生成的图像
open ./results/horse2zebra_pretrained/test_latest/images/  # macOS
# 或
ls ./results/horse2zebra_pretrained/test_latest/images/    # Linux
```

## 📊 第六步：配置和选项

### 6.1 重要选项解释

```bash
# 数据加载
--dataroot           # 数据集路径
--dataset_mode       # 数据集模式: aligned, unaligned, single
--serial_batches     # 顺序加载 (不随机打乱)
--num_threads        # 加载数据的线程数

# 模型选择
--model              # 模型类型: cycle_gan, pix2pix, test, colorization
--netG               # 生成器架构: resnet_6blocks, resnet_9blocks, unet_256, unet_128
--netD               # 判别器架构: basic, n_layers, pixel

# 训练参数
--lr                 # 初始学习率 (默认0.0002)
--beta1              # Adam优化器的beta1 (默认0.5)
--pool_size          # 图像池大小 (默认50)
--cut_param          # CUT论文的参数

# 硬件配置
--gpu_ids            # GPU ID列表,-1表示CPU
--num_test           # 测试图像数量
--use_wandb          # 使用Weights & Biases
```

### 6.2 模型架构选择

```bash
# CycleGAN - 更多参数，需要更多显存
python train.py ... --netG resnet_9blocks --netD n_layers

# CycleGAN - 更少参数，更快训练
python train.py ... --netG resnet_6blocks --netD basic

# pix2pix - U-Net生成器
python train.py ... --netG unet_256 --netD n_layers
```

## 🔧 故障排除

### 问题1: "CUDA out of memory"

**解决方案:**
```bash
# 减少批次大小
python train.py ... --batch_size 1

# 使用更小的网络
python train.py ... --netG resnet_6blocks

# 使用CPU (慢很多)
python train.py ... --gpu_ids -1
```

### 问题2: 数据集下载缓慢

```bash
# 手动下载数据集，然后解压到 ./datasets/
# 访问: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

# 或使用国内镜像源
```

### 问题3: 模块导入错误

```bash
# 验证安装
python -c "import data; import models; import options; import util"

# 重新安装
pip uninstall -y pytorch-cyclegan-pix2pix
pip install -e .
```

### 问题4: 显存不足 (Apple Silicon M系列)

```bash
# 使用MPS (Metal Performance Shaders)
python train.py ... --gpu_ids 0  # 自动检测MPS
```

## 📈 监控训练进度

### 6.1 使用Weights & Biases监控

```bash
# 安装wandb (如果还没安装)
pip install wandb

# 训练时启用wandb
python train.py ... --use_wandb

# 在浏览器打开https://wandb.ai查看实时指标
```

### 6.2 使用Tensorboard (可选)

```bash
# 安装tensorboard
pip install tensorboard

# 查看训练日志
tensorboard --logdir ./checkpoints/{model_name}
```

## 💾 保存和恢复训练

### 7.1 继续训练

```bash
# 从最后一个检查点继续训练
python train.py ... --continue_train

# 指定特定的周期
python train.py ... --continue_train --epoch_count 100
```

### 7.2 模型检查点位置

```
./checkpoints/{model_name}/
├── latest_net_G.pt        # 最新生成器
├── latest_net_D.pt        # 最新判别器
├── {epoch}_net_G.pt       # 特定周期的生成器
└── {epoch}_net_D.pt       # 特定周期的判别器
```

## 📝 完整示例工作流

### 示例：快速启动CycleGAN

```bash
# 1. 激活虚拟环境
cd /Users/leon/Desktop/cycleGAN
source venv/bin/activate

# 2. 下载数据集
bash ./datasets/download_cyclegan_dataset.sh horse2zebra

# 3. 训练模型 (10个周期快速测试)
python train.py \
  --dataroot ./datasets/horse2zebra \
  --name horse2zebra_test \
  --model cycle_gan \
  --n_epochs 10 \
  --n_epochs_decay 10 \
  --batch_size 1 \
  --display_id -1

# 4. 测试模型
python test.py \
  --dataroot ./datasets/horse2zebra \
  --name horse2zebra_test \
  --model cycle_gan

# 5. 查看结果
open ./results/horse2zebra_test/test_latest/images/
```

### 示例：pix2pix with 预训练模型

```bash
# 1. 下载数据集和预训练模型
bash ./datasets/download_pix2pix_dataset.sh facades
bash ./scripts/download_pix2pix_model.sh facades_label2photo

# 2. 测试预训练模型
python test.py \
  --dataroot ./datasets/facades \
  --direction BtoA \
  --model pix2pix \
  --name facades_label2photo_pretrained

# 3. 查看结果
open ./results/facades_label2photo_pretrained/test_latest/
```

## 🎓 资源和文档

- [原始论文 - CycleGAN](https://arxiv.org/pdf/1703.10593.pdf)
- [原始论文 - pix2pix](https://arxiv.org/pdf/1611.07004.pdf)
- [数据集详情](./docs/datasets.md)
- [训练技巧](./docs/tips.md)
- [常见问题解答](./docs/qa.md)
- [项目主页](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## ✨ 快速命令参考

```bash
# 激活环境
source venv/bin/activate

# 下载数据集
bash ./datasets/download_cyclegan_dataset.sh horse2zebra

# 训练模型
python train.py --dataroot ./datasets/horse2zebra --name h2z --model cycle_gan

# 测试模型
python test.py --dataroot ./datasets/horse2zebra --name h2z --model cycle_gan

# 看日志
tail -f ./checkpoints/h2z/loss_log.txt

# 查看帮助
python train.py --help
python test.py --help
```

---
**提示**: 第一次运行时可能会下载一些大文件，请耐心等待。建议在有稳定网络连接的地方进行数据集下载。
