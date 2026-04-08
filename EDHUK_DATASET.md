# 📸 EdUHK 雨天/晴天数据集准备指南

本指南将帮助您收集500张香港教育大学（EdUHK）的雨天和晴天照片，用于训练CycleGAN模型。

## 📚 目录

1. [快速开始](#快速开始)
2. [数据集结构](#数据集结构)
3. [图像采集方法](#图像采集方法)
4. [数据集验证](#数据集验证)
5. [CycleGAN训练配置](#cyclegan训练配置)
6. [常见问题](#常见问题)

---

## 🚀 快速开始

### 方法1: 自动下载 (Bing Image Search)

**安装依赖:**
```bash
pip install bing-image-downloader requests pillow
```

**运行下载:**
```bash
cd /Users/leon/Desktop/cycleGAN

# 下载500张图像 (雨天和晴天各250张)
python prepare_edhuk_dataset.py --bing --num-images 250

# 验证数据集
python prepare_edhuk_dataset.py --validate
```

### 方法2: 手动下载

```bash
# 显示详细指南
python prepare_edhuk_dataset.py --manual
```

然后按照指南从Unsplash、Pexels等网站手动下载。

---

## 📁 数据集结构

CycleGAN模型实现支持**未配对数据集**，所以您需要以下结构:

```
datasets/edhuk_weather/
├── trainA/           # 训练集 - 雨天图像 (200-250张)
│   ├── rain_001.jpg
│   ├── rain_002.jpg
│   └── ...
├── trainB/           # 训练集 - 晴天图像 (200-250张)
│   ├── sunny_001.jpg
│   ├── sunny_002.jpg
│   └── ...
├── testA/            # 测试集 - 雨天图像 (50张)
│   └── ...
├── testB/            # 测试集 - 晴天图像 (50张)
│   └── ...
└── dataset_info.json # 数据集元数据
```

### 为什么这样分类?

- **trainA/trainB**: 用于训练，各需要200-300张高质量图像
- **testA/testB**: 用于测试/验证，各需要50-100张
- **未配对**: 雨天和晴天图像不需要一一对应，这是CycleGAN的优势

---

## 📥 图像采集方法

### 方法A: 使用爬虫脚本 (推荐)

**推荐原因:**
- 自动批量下载
- 快速省时
- 质量检查

**步骤:**

```bash
# 1. 安装依赖
pip install bing-image-downloader requests pillow

# 2. 运行脚本
python prepare_edhuk_dataset.py --bing --num-images 250

# 3. 等待下载完成 (通常10-30分钟)
# 状态会实时显示进度

# 4. 验证结果
python prepare_edhuk_dataset.py --validate
```

### 方法B: 使用API接口 (需要注册)

#### **Unsplash API** (推荐)

官网: https://unsplash.com/developers

**步骤:**
1. 访问 https://unsplash.com/developers
2. 创建应用并获取 Access Key
3. 运行: `python prepare_edhuk_dataset.py --unsplash --api-key YOUR_KEY`

#### **Pexels API**

官网: https://www.pexels.com/api/

**步骤:**
1. 访问 https://www.pexels.com/api/
2. 获取API Key
3. 运行: `python prepare_edhuk_dataset.py --pexels --api-key YOUR_KEY`

### 方法C: 手动下载 (需要更多时间)

**推荐网站:**

1. **Unsplash** (https://unsplash.com)
   - 搜索: "rain rainy weather"
   - 搜索: "sunny clear weather"
   - 优点: 高质量无版权

2. **Pexels** (https://www.pexels.com)
   - 搜索同上
   - 优点: 完全免费

3. **Pixabay** (https://pixabay.com)
   - 搜索同上
   - 优点: 图像库大

**详细步骤:**

```bash
# 1. 下载图像到本地文件夹
# 2. 创建目录结构
mkdir -p datasets/edhuk_weather/{trainA,trainB,testA,testB}

# 3. 将雨天图像移到trainA和testA
cp ./downloaded/rain_*.jpg ./datasets/edhuk_weather/trainA/
cp ./downloaded/rain_*.jpg ./datasets/edhuk_weather/testA/ (选50-100张)

# 4. 将晴天图像移到trainB和testB
cp ./downloaded/sunny_*.jpg ./datasets/edhuk_weather/trainB/
cp ./downloaded/sunny_*.jpg ./datasets/edhuk_weather/testB/ (选50-100张)

# 5. 验证结果
python prepare_edhuk_dataset.py --validate
```

---

## ✅ 数据集验证

### 验证步骤

```bash
# 检查文件数量和图像质量
python prepare_edhuk_dataset.py --validate
```

**预期输出:**
```
📋 验证数据集...
============================================================
  trainA     : 250 张图像
  trainB     : 250 张图像
  testA      :  50 张图像
  testB      :  50 张图像
============================================================
  总计:       600 张图像

📊 检查图像质量...
  trainA     : 平均分辨率 1920x1080
  trainB     : 平均分辨率 1920x1080
  testA      : 平均分辨率 1920x1080
  testB      : 平均分辨率 1920x1080
```

### 手动检查

```bash
# 进入数据集目录
cd datasets/edhuk_weather

# 查看目录结构
ls -lh

# 检查每个子目录的文件数
for dir in trainA trainB testA testB; do
  echo "$dir: $(ls $dir | wc -l) files"
done

# 查看图像分辨率 (macOS)
file trainA/*.jpg | head -10
```

---

## 🎯 CycleGAN 训练配置

一旦数据集准备好，即可开始训练。

### 训练命令

```bash
# 激活虚拟环境
source venv/bin/activate

# 基础训练 (快速测试)
python train.py \
  --dataroot ./datasets/edhuk_weather \
  --name edhuk_rain2sunny \
  --model cycle_gan \
  --n_epochs 50 \
  --batch_size 1 \
  --display_freq 100

# 完整训练 (生产环境)
python train.py \
  --dataroot ./datasets/edhuk_weather \
  --name edhuk_rain2sunny_full \
  --model cycle_gan \
  --n_epochs 200 \
  --n_epochs_decay 200 \
  --batch_size 4 \
  --display_freq 100 \
  --print_freq 100 \
  --save_latest_freq 5000 \
  --netG resnet_9blocks \
  --gpu_ids 0
```

### 多GPU训练 (DDP)

```bash
# 使用4个GPU训练
torchrun --nproc_per_node=4 train.py \
  --dataroot ./datasets/edhuk_weather \
  --name edhuk_rain2sunny_ddp \
  --model cycle_gan \
  --batch_size 8
```

### 常用参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `n_epochs` | 训练周期数 | 200 |
| `n_epochs_decay` | 学习率衰减周期 | 200 |
| `batch_size` | 批次大小 | 1-4 (根据GPU内存) |
| `lr` | 初始学习率 | 0.0002 |
| `netG` | 生成器架构 | resnet_9blocks |
| `netD` | 判别器架构 | n_layers |

---

## 🧪 测试和推理

### 使用训练好的模型进行测试

```bash
# 在测试集上运行
python test.py \
  --dataroot ./datasets/edhuk_weather \
  --name edhuk_rain2sunny_full \
  --model cycle_gan

# 结果保存到: ./results/edhuk_rain2sunny_full/test_latest/
```

### 单张图像转换

```bash
# 准备一张图像
mkdir -p ./single_images
cp your_image.jpg ./single_images/

# 运行转换
python test.py \
  --dataroot ./single_images \
  --name edhuk_rain2sunny_full \
  --model test \
  --no_dropout
```

---

## 📊 数据集质量检查清单

在训练前检查以下项目:

### ✅ 数量检查
- [ ] trainA: 200-300张雨天图像
- [ ] trainB: 200-300张晴天图像
- [ ] testA: 50-100张雨天图像
- [ ] testB: 50-100张晴天图像

### ✅ 质量检查
- [ ] 所有图像分辨率 >= 256x256 (推荐 512x512+)
- [ ] 图像格式为 JPG/PNG/BMP
- [ ] 没有损坏或不完整的图像
- [ ] 没有文本水印

### ✅ 内容检查
- [ ] 雨天图像明确显示降雨
- [ ] 晴天图像明确显示晴朗天气
- [ ] 图像显示类似的地点/场景 (建议是EdUHK校园)
- [ ] 图像包含多样的时间、角度、对象

### ✅ 数据集多样性
- [ ] 不同时间拍摄 (早晨、中午、傍晚)
- [ ] 不同角度/视角
- [ ] 不同天气强度 (小雨到暴雨)
- [ ] 不同场景 (室外、建筑、植被等)

---

## ❓ 常见问题

### Q1: 如何获得更多高质量图像?

**A:** 使用多个搜索关键词变体：
- "Hong Kong rain weather"
- "rainy day urban photography"
- "Hong Kong street rain"
- "weather conditions city"

### Q2: 下载速度太慢怎么办?

**A:** 
- 使用VPN或代理
- 尝试不同的下载方法 (Bing, Unsplash, Pexels)
- 分批下载 (每次100-200张)

### Q3: 图像大小不一致会影响训练吗?

**A:** CycleGAN会自动调整大小到指定分辨率 (默认256x256)，但最好保持原始分辨率一致。

### Q4: 可以混合使用来自不同来源的图像吗?

**A:** 可以的。但要确保：
- 所有图像质量相似
- 没有版权问题
- 主题相关 (雨天/晴天)

### Q5: 需要多少张图像?

**A:** 最少：
- 200张训练数据 (各天气类型)
- 50张测试数据 (各天气类型)

最佳：
- 300-500张训练数据
- 100张测试数据

### Q6: 训练需要多长时间?

**A:** 
```
GPU (RTX 3090):     ~4-6小时 (200周期)
GPU (RTX 2080):     ~8-12小时
GPU (M1 Pro/Max):   ~6-10小时 (MPS)
CPU:                ~24-48小时 (不推荐)
```

### Q7: 如何评估模型质量?

**A:** 检查以下指标：
1. **视觉质量**: 转换结果看起来自然吗?
2. **内容保存**: 物体和结构是否保持?
3. **风格转换**: 是否成功改变了天气条件?
4. **训练稳定性**: 查看loss曲线是否稳定下降

---

## 🔗 相关资源

### 官方文档
- [CycleGAN 论文](https://arxiv.org/pdf/1703.10593.pdf)
- [PyTorch CycleGAN 实现](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [数据集最佳实践](./docs/datasets.md)

### 推荐数据源
- [Unsplash](https://unsplash.com)
- [Pexels](https://www.pexels.com)
- [Pixabay](https://pixabay.com)
- [Google Dataset Search](https://datasetsearch.research.google.com/)

### 数据集工具
- [Dataset Utils](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/datasets)

---

## 📝 快速参考命令

```bash
# 创建目录结构
mkdir -p datasets/edhuk_weather/{trainA,trainB,testA,testB}

# 自动下载 (Bing)
python prepare_edhuk_dataset.py --bing --num-images 250

# 验证数据集
python prepare_edhuk_dataset.py --validate

# 训练模型
python train.py \
  --dataroot ./datasets/edhuk_weather \
  --name edhuk_rain2sunny \
  --model cycle_gan \
  --n_epochs 200

# 测试模型
python test.py \
  --dataroot ./datasets/edhuk_weather \
  --name edhuk_rain2sunny \
  --model cycle_gan

# 查看帮助
python prepare_edhuk_dataset.py --help
```

---

**✨ 现在您已准备好收集数据并训练CycleGAN模型了！**
