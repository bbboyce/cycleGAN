#!/bin/bash
# CycleGAN 消融实验 - 快速开始脚本
# Usage: bash ablation_quickstart.sh

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   CycleGAN 消融实验 - 快速开始                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}\n"

# 1. 检查环境
echo -e "${YELLOW}📋 检查环境...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 未安装${NC}"
    exit 1
fi

python3 --version
echo -e "${GREEN}✅ Python 版本检查通过${NC}\n"

# 2. 检查虚拟环境
echo -e "${YELLOW}🔍 检查虚拟环境...${NC}"

if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 创建虚拟环境...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate
echo -e "${GREEN}✅ 虚拟环境已激活${NC}\n"

# 3. 生成消融配置
echo -e "${YELLOW}⚙️  生成消融实验配置...${NC}"

if [ ! -f "ablation_configs.json" ]; then
    python ablation_study.py --gen-config
    echo -e "${GREEN}✅ 配置已生成: ablation_configs.json${NC}"
else
    echo -e "${GREEN}✅ 配置文件已存在${NC}"
fi
echo ""

# 4. 列出实验
echo -e "${YELLOW}📊 可用的消融实验:${NC}"
python ablation_study.py --list-experiments
echo ""

# 5. 检查数据集
echo -e "${YELLOW}🔍 检查数据集...${NC}"

DATASETS=(
    "./datasets/handbags"
    "./datasets/horses2zebras" 
    "./datasets/facades"
)

FOUND_DATASET=""
for dataset in "${DATASETS[@]}"; do
    if [ -d "$dataset" ]; then
        FOUND_DATASET="$dataset"
        echo -e "${GREEN}✅ 找到数据集: $dataset${NC}"
        break
    fi
done

if [ -z "$FOUND_DATASET" ]; then
    echo -e "${RED}❌ 未找到数据集${NC}"
    echo -e "${YELLOW}请下载数据集:${NC}"
    echo "  bash datasets/download_cyclegan_dataset.sh handbags"
    echo ""
    read -p "按 Enter 继续或 Ctrl+C 退出..." -t 5 || true
    FOUND_DATASET="./datasets/handbags"
fi
echo ""

# 6. 开始实验
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   🚀 开始消融实验                                      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}\n"

echo -e "${YELLOW}选择运行方式:${NC}"
echo "  1. 运行所有实验 (推荐)"
echo "  2. 预览配置并退出"
echo "  3. 自定义命令"
echo ""

read -p "请选择 (1-3): " choice

case $choice in
    1)
        echo -e "${YELLOW}⏳ 开始运行消融实验...${NC}"
        echo -e "${YELLOW}数据集: $FOUND_DATASET${NC}\n"
        python ablation_study.py --dataroot "$FOUND_DATASET"
        ;;
    2)
        echo -e "${GREEN}✅ 配置已设置，运行以下命令开始实验:${NC}"
        echo "  python ablation_study.py --dataroot $FOUND_DATASET"
        exit 0
        ;;
    3)
        echo -e "${YELLOW}输入自定义命令 (例如: python train.py --name exp1 --model cycle_gan_ablation --ablation_mode no_cycle --dataroot ./datasets/handbags):${NC}"
        read -p "> " custom_cmd
        eval "$custom_cmd"
        ;;
    *)
        echo -e "${RED}❌ 无效选择${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   📊 开始分析结果                                      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}\n"

# 7. 提示下一步
echo -e "${GREEN}✅ 实验完成！${NC}\n"
echo -e "${YELLOW}📄 后续步骤:${NC}"
echo "  1. 打开 Jupyter Notebook 查看结果:"
echo "     jupyter notebook ABLATION_STUDY_ANALYSIS.ipynb"
echo ""
echo "  2. 查看生成的报告:"
echo "     cat results_ablation/ablation_report_*.md"
echo ""
echo "  3. 查看训练曲线对比:"
echo "     open results_ablation/training_curves_comparison.png"
echo ""
echo -e "${YELLOW}📚 查看详细指南:${NC}"
echo "  cat ABLATION_STUDY.md"
echo ""
echo -e "${GREEN}🎉 祝你消融实验顺利！${NC}"
