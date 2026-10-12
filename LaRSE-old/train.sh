#!/bin/bash

# ============================================================
# LaRSE 训练脚本 - ViT-L-14
# ============================================================

export CUDA_VISIBLE_DEVICES=0

python train.py \
  --backbone clip_vitl14_384 \
  --dataset buff1w \
  --data_path /home/heda/zyy/seg/datasets \
  --batch_size 8 \
  --base_lr 0.004 \
  --max_epochs 80 \
  --gpus 1 \
  --precision 16 \
  --exp_name larse_vitl14 \
  --widehead \
  --no-scaleinv \
  --arch_option 0 \
  --block_depth 0 \
  --activation lrelu

# ============================================================
# 参数说明
# ============================================================
# --backbone: 使用的视觉编码器
#   - clip_vitl14_384: ViT-L-14 (新，768维)
#   - clip_vitb32_384: ViT-B-32 (旧，512维)
#   - clip_vitl16_384: ViT-L-16
#
# --dataset: 数据集名称
#   - buff1w: BUFF1W 建筑功能分类数据集
#
# --data_path: 数据集根目录
#
# --batch_size: 批次大小（根据 GPU 显存调整）
#   - 8: 适合 16GB+ 显存
#   - 4: 适合 8-16GB 显存
#
# --base_lr: 基础学习率
#
# --max_epochs: 最大训练轮数
#
# --precision: 训练精度
#   - 16: 混合精度（更快，省显存）
#   - 32: 全精度
#
# --exp_name: 实验名称（用于保存 checkpoint 和 logs）
#
# --widehead: 使用宽 head（LaRSE 默认）
#
# --no-scaleinv: 不使用 scale invariant（LaRSE 默认）
#
# --arch_option: 架构选项
#   - 0: 基础架构（默认）
#   - 1: bottleneck block
#   - 2: depthwise block
#
# ============================================================
