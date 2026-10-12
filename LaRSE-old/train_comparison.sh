#!/bin/bash

# ============================================================
# LaRSE 对比实验训练脚本
# ViT-B-32 (baseline) vs ViT-L-14 (upgraded)
# ============================================================

export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "LaRSE 对比实验训练"
echo "=========================================="
echo ""

# ============================================================
# 实验 1: ViT-B-32 (baseline)
# ============================================================
echo "实验 1/2: 训练 ViT-B-32 baseline..."
echo ""

python train.py \
  --backbone clip_vitb32_384 \
  --dataset buff1w \
  --data_path /home/heda/zyy/seg/datasets \
  --batch_size 8 \
  --base_lr 0.004 \
  --max_epochs 80 \
  --gpus 1 \
  --precision 16 \
  --exp_name larse_vitb32_baseline \
  --widehead \
  --no-scaleinv \
  --arch_option 0 \
  --block_depth 0 \
  --activation lrelu

echo ""
echo "✅ ViT-B-32 训练完成"
echo ""

# ============================================================
# 实验 2: ViT-L-14 (upgraded)
# ============================================================
echo "实验 2/2: 训练 ViT-L-14 upgraded..."
echo ""

python train.py \
  --backbone clip_vitl14_384 \
  --dataset buff1w \
  --data_path /home/heda/zyy/seg/datasets \
  --batch_size 8 \
  --base_lr 0.004 \
  --max_epochs 80 \
  --gpus 1 \
  --precision 16 \
  --exp_name larse_vitl14_upgraded \
  --widehead \
  --no-scaleinv \
  --arch_option 0 \
  --block_depth 0 \
  --activation lrelu

echo ""
echo "✅ ViT-L-14 训练完成"
echo ""

# ============================================================
# 完成
# ============================================================
echo "=========================================="
echo "✅ 对比实验训练完成！"
echo "=========================================="
echo ""
echo "Checkpoints:"
echo "  - ViT-B-32: checkpoints/larse_vitb32_baseline/checkpoint_LARSE.ckpt"
echo "  - ViT-L-14: checkpoints/larse_vitl14_upgraded/checkpoint_LARSE.ckpt"
echo ""
echo "下一步：运行测试对比"
echo "  bash test_comparison.sh"
echo ""
