#!/usr/bin/env python3
"""
LaRSE 训练脚本
支持 ViT-L-14 和其他 CLIP backbone
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser

from modules.lseg_module import LSegModule


def main():
    # ============================================================
    # 参数解析
    # ============================================================
    parser = ArgumentParser()

    # Trainer 参数
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes")
    parser.add_argument("--accelerator", type=str, default="gpu", help="accelerator type")
    parser.add_argument("--precision", type=int, default=16, help="16 for mixed precision, 32 for full precision")

    # 实验名称
    parser.add_argument("--exp_name", type=str, default="larse_experiment", help="experiment name")
    parser.add_argument("--log_dir", type=str, default="logs", help="log directory")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="checkpoint directory")

    # 添加模型特定参数
    parser = LSegModule.add_model_specific_args(parser)

    args = parser.parse_args()

    # ============================================================
    # 打印配置
    # ============================================================
    print("=" * 60)
    print("LaRSE 训练配置")
    print("=" * 60)
    print(f"实验名称: {args.exp_name}")
    print(f"Backbone: {args.backbone}")
    print(f"数据集: {args.dataset}")
    print(f"数据路径: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"学习率: {args.base_lr}")
    print(f"最大 epochs: {args.max_epochs}")
    print(f"GPU 数量: {args.gpus}")
    print(f"精度: {args.precision}")
    print("=" * 60)
    print()

    # ============================================================
    # 创建模型
    # ============================================================
    print("正在创建模型...")
    model = LSegModule(
        data_path=args.data_path,
        dataset=args.dataset,
        batch_size=args.batch_size,
        base_lr=args.base_lr,
        max_epochs=args.max_epochs,
        **vars(args)
    )
    print("✅ 模型创建成功")
    print()

    # ============================================================
    # 配置 Callbacks
    # ============================================================
    checkpoint_dir = os.path.join(args.ckpt_dir, args.exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        # 保存最佳模型
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="checkpoint_LARSE",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        # 学习率监控
        LearningRateMonitor(logging_interval="step"),
    ]

    # ============================================================
    # 配置 Logger
    # ============================================================
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.exp_name,
    )

    # ============================================================
    # 配置 Trainer
    # ============================================================
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.gpus,
        num_nodes=args.num_nodes,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=1.0,  # 每个 epoch 验证一次
        gradient_clip_val=0.5,
    )

    # ============================================================
    # 开始训练
    # ============================================================
    print("=" * 60)
    print("开始训练...")
    print("=" * 60)
    print()

    trainer.fit(model)

    print()
    print("=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)
    print(f"最佳模型保存在: {checkpoint_dir}/checkpoint_LARSE.ckpt")
    print()


if __name__ == "__main__":
    main()
