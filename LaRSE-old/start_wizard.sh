#!/bin/bash

# ============================================================
# LaRSE ViT-L-14 启动向导
# 这是你唯一需要运行的脚本！
# ============================================================

clear

cat << 'EOF'
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          LaRSE ViT-L-14 升级 - 启动向导 🚀              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
EOF

echo ""
echo "欢迎！这个向导会帮你完成所有设置。"
echo ""

# ============================================================
# 检查当前位置
# ============================================================
if [ ! -f "train.py" ]; then
    echo "❌ 错误：请在 LaRSE 项目根目录运行此脚本"
    echo ""
    echo "正确做法："
    echo "  cd /data/cjh/LaRSE"
    echo "  bash start_wizard.sh"
    exit 1
fi

echo "✅ 位置正确：$(pwd)"
echo ""

# ============================================================
# 检查环境
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Step 1/5: 检查环境"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Conda 环境未激活"
    echo ""
    echo "请先运行："
    echo "  conda activate /data/cjh/envs/LARSE-clean"
    echo ""
    read -p "激活后按回车继续..."
fi

echo "✅ Conda 环境: $CONDA_DEFAULT_ENV"

# 检查 Python
python_version=$(python --version 2>&1)
echo "✅ Python: $python_version"

# 检查 PyTorch
pytorch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✅ PyTorch: $pytorch_version"
else
    echo "❌ PyTorch 未安装"
    exit 1
fi

# 检查 CUDA
cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$cuda_available" = "True" ]; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "✅ CUDA 可用: $gpu_name"
else
    echo "❌ CUDA 不可用"
    exit 1
fi

echo ""

# ============================================================
# 检查文件
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Step 2/5: 检查文件"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

files=(
    "train.py"
    "train.sh"
    "train_comparison.sh"
    "test.sh"
    "modules/models/lseg_blocks.py"
    "modules/models/lseg_net.py"
    "modules/models/lseg_vit.py"
)

all_files_ok=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file 不存在"
        all_files_ok=false
    fi
done

if [ "$all_files_ok" = false ]; then
    echo ""
    echo "❌ 文件不完整，请先复制所有文件到服务器"
    exit 1
fi

echo ""

# ============================================================
# 检查数据集
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Step 3/5: 检查数据集"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

data_path="/home/heda/zyy/seg/datasets/buff1w"
if [ -d "$data_path" ]; then
    echo "✅ 数据集路径存在: $data_path"

    if [ -d "$data_path/train" ]; then
        train_count=$(ls "$data_path/train" 2>/dev/null | wc -l)
        echo "   训练集: $train_count 文件"
    fi

    if [ -d "$data_path/val" ]; then
        val_count=$(ls "$data_path/val" 2>/dev/null | wc -l)
        echo "   验证集: $val_count 文件"
    fi
else
    echo "⚠️  数据集路径不存在: $data_path"
    echo ""
    read -p "输入正确的数据集路径: " custom_data_path
    data_path="$custom_data_path"
fi

echo ""

# ============================================================
# 测试模型
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Step 4/5: 测试模型"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "正在测试 ViT-L-14 模型..."

python3 << 'PYTEST'
import torch
import torch.nn as nn
from modules.models.lseg_net import LSeg

try:
    head = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    model = LSeg(
        labels=['building', 'road', 'tree', 'water', 'background'],
        head=head,
        features=256,
        backbone='clip_vitl14_384',
        readout='project',
        arch_option=0,
        block_depth=0,
    )

    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model(x)

    print(f"✅ 模型测试成功: {x.shape} → {output.shape}")
    exit(0)
except Exception as e:
    print(f"❌ 模型测试失败: {e}")
    exit(1)
PYTEST

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 模型测试失败，请检查代码"
    exit 1
fi

echo ""

# ============================================================
# 选择训练方案
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Step 5/5: 选择训练方案"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "请选择训练方案："
echo ""
echo "  1) 仅训练 ViT-L-14 (快速, 约 8-12 小时)"
echo "  2) 对比实验 (ViT-B-32 + ViT-L-14, 约 16-24 小时, 推荐用于论文)"
echo "  3) 退出"
echo ""
read -p "输入选项 [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 开始训练 ViT-L-14"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "训练命令："
        echo "  bash train.sh"
        echo ""
        echo "日志将保存到: training_vitl14.log"
        echo "Checkpoints: checkpoints/larse_vitl14/"
        echo ""
        read -p "按回车开始训练..."

        chmod +x train.sh
        bash train.sh 2>&1 | tee training_vitl14.log
        ;;

    2)
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 开始对比实验训练"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "训练命令："
        echo "  bash train_comparison.sh"
        echo ""
        echo "将依次训练："
        echo "  1. ViT-B-32 (baseline)"
        echo "  2. ViT-L-14 (upgraded)"
        echo ""
        echo "日志将保存到: training_comparison.log"
        echo "Checkpoints:"
        echo "  - checkpoints/larse_vitb32_baseline/"
        echo "  - checkpoints/larse_vitl14_upgraded/"
        echo ""
        read -p "按回车开始训练..."

        chmod +x train_comparison.sh
        bash train_comparison.sh 2>&1 | tee training_comparison.log
        ;;

    3)
        echo ""
        echo "已退出。你可以稍后手动运行："
        echo "  bash train.sh              # 仅 ViT-L-14"
        echo "  bash train_comparison.sh   # 对比实验"
        exit 0
        ;;

    *)
        echo ""
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 训练完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "下一步："
echo "  1. 查看训练日志"
echo "  2. 运行测试脚本"
echo "  3. 分析结果"
echo ""
echo "测试命令："
if [ "$choice" = "1" ]; then
    echo "  bash test.sh"
elif [ "$choice" = "2" ]; then
    echo "  bash test_comparison.sh"
fi
echo ""
