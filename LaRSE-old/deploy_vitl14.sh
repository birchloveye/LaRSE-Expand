#!/bin/bash

# ============================================================
# LaRSE ViT-L-14 升级 - 一键部署脚本
# ============================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "LaRSE ViT-L-14 升级部署脚本"
echo "=========================================="
echo ""

# ============================================================
# Step 1: 检查环境
# ============================================================
echo "Step 1/4: 检查环境..."

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: Python3 未安装"
    exit 1
fi
echo "✅ Python3: $(python3 --version)"

# 检查 pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ 错误: pip3 未安装"
    exit 1
fi
echo "✅ pip3 已安装"

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA 可用"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  警告: CUDA 不可用，仅能在 CPU 上运行"
fi

echo ""

# ============================================================
# Step 2: 安装依赖
# ============================================================
echo "Step 2/4: 检查并安装依赖..."

# 检查并安装 huggingface-hub
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "正在安装 huggingface-hub..."
    pip3 install huggingface_hub
else
    echo "✅ huggingface-hub 已安装"
fi

# 检查 PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    echo "❌ 错误: PyTorch 未安装，请先安装 PyTorch"
    exit 1
fi
echo "✅ PyTorch 已安装: $(python3 -c 'import torch; print(torch.__version__)')"

# 检查 open_clip
if ! python3 -c "import open_clip" 2>/dev/null; then
    echo "❌ 错误: open_clip 未安装，请运行: pip install open-clip-torch"
    exit 1
fi
echo "✅ open_clip 已安装"

# 检查 timm
if ! python3 -c "import timm" 2>/dev/null; then
    echo "❌ 错误: timm 未安装，请运行: pip install timm"
    exit 1
fi
echo "✅ timm 已安装"

echo ""

# ============================================================
# Step 3: 下载权重文件
# ============================================================
echo "Step 3/4: 下载 RemoteCLIP ViT-L-14 权重..."

# 确定权重路径
if [ -n "$CLIP_CACHE_DIR" ]; then
    WEIGHTS_DIR="$CLIP_CACHE_DIR"
    echo "使用自定义路径: $WEIGHTS_DIR"
else
    WEIGHTS_DIR="$HOME/.cache/clip"
    echo "使用默认路径: $WEIGHTS_DIR"
fi

# 创建目录
mkdir -p "$WEIGHTS_DIR"

# 权重文件路径
WEIGHTS_FILE="$WEIGHTS_DIR/RemoteCLIP-ViT-L-14.pt"

# 检查权重是否已存在
if [ -f "$WEIGHTS_FILE" ]; then
    FILE_SIZE=$(du -h "$WEIGHTS_FILE" | cut -f1)
    echo "✅ 权重文件已存在: $WEIGHTS_FILE ($FILE_SIZE)"

    read -p "是否重新下载？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "跳过下载"
    else
        echo "重新下载权重文件..."
        rm -f "$WEIGHTS_FILE"
        python3 << EOF
from huggingface_hub import hf_hub_download
print("正在从 HuggingFace 下载 RemoteCLIP-ViT-L-14.pt (约 900MB)...")
path = hf_hub_download(
    repo_id="chendelong/RemoteCLIP",
    filename="RemoteCLIP-ViT-L-14.pt",
    local_dir="$WEIGHTS_DIR"
)
print(f"✅ 下载完成: {path}")
EOF
    fi
else
    echo "正在下载权重文件..."
    python3 << EOF
from huggingface_hub import hf_hub_download
print("正在从 HuggingFace 下载 RemoteCLIP-ViT-L-14.pt (约 900MB)...")
print("这可能需要几分钟，请耐心等待...")
path = hf_hub_download(
    repo_id="chendelong/RemoteCLIP",
    filename="RemoteCLIP-ViT-L-14.pt",
    local_dir="$WEIGHTS_DIR"
)
print(f"✅ 下载完成: {path}")
EOF
fi

# 验证文件完整性
if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "❌ 错误: 权重文件下载失败"
    exit 1
fi

FILE_SIZE_BYTES=$(stat -f%z "$WEIGHTS_FILE" 2>/dev/null || stat -c%s "$WEIGHTS_FILE" 2>/dev/null)
FILE_SIZE_MB=$((FILE_SIZE_BYTES / 1024 / 1024))

if [ $FILE_SIZE_MB -lt 800 ]; then
    echo "❌ 错误: 权重文件大小异常 (${FILE_SIZE_MB}MB)，预期约 900MB"
    echo "请删除文件后重新运行: rm $WEIGHTS_FILE"
    exit 1
fi

echo "✅ 权重文件验证通过: ${FILE_SIZE_MB}MB"
echo ""

# ============================================================
# Step 4: 测试模型加载
# ============================================================
echo "Step 4/4: 测试模型加载..."

# 设置环境变量
export CLIP_CACHE_DIR="$WEIGHTS_DIR"

# 测试模型
python3 << 'EOF'
import torch
import sys

try:
    print("正在加载模型...")
    from modules.models.lseg_net import LSeg

    # 创建模型
    model = LSeg(
        head=None,
        features=256,
        backbone="clip_vitl14_384",
        readout="project",
    )
    model.labels = ["building", "road", "tree", "water"]

    print("✅ 模型创建成功")

    # 测试前向传播
    print("正在测试前向传播...")
    x = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        output = model(x)

    print(f"✅ 前向传播成功")
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output.shape}")
    print("")
    print("=" * 50)
    print("🎉 所有测试通过！ViT-L-14 升级成功！")
    print("=" * 50)

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 部署完成！"
    echo "=========================================="
    echo ""
    echo "现在可以开始训练："
    echo ""
    echo "  python train.py \\"
    echo "    --backbone clip_vitl14_384 \\"
    echo "    --batch_size 8 \\"
    echo "    --max_epochs 80 \\"
    echo "    --exp_name larse_vitl14"
    echo ""
    echo "如果使用自定义权重路径："
    echo ""
    echo "  export CLIP_CACHE_DIR=$WEIGHTS_DIR"
    echo "  python train.py --backbone clip_vitl14_384 ..."
    echo ""
    echo "=========================================="
else
    echo ""
    echo "❌ 部署失败，请查看上面的错误信息"
    exit 1
fi
