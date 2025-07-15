"""
模型相关工具模块
包含模型创建、加载、保存等功能
"""

import logging
import torch
from typing import Dict, Any, Optional
from transformers import AutoModelForTokenClassification
from pathlib import Path

# 获取日志记录器，但不配置基本设置（让主脚本来配置）
logger = logging.getLogger(__name__)


def create_model(
    model_config, tag2id: Dict[str, int], id2tag: Dict[int, str]
) -> AutoModelForTokenClassification:
    """
    创建NER模型

    Args:
        model_config: 模型配置
        tag2id: 标签到ID的映射
        id2tag: ID到标签的映射

    Returns:
        初始化的模型

    Raises:
        ValueError: 配置参数无效
        Exception: 模型创建失败
    """
    if not tag2id or not id2tag:
        raise ValueError("标签映射不能为空")

    if len(tag2id) != len(id2tag):
        raise ValueError("标签映射不一致")

    try:
        logger.info(f"正在创建模型: {model_config.model_name}")
        logger.info(f"标签数量: {model_config.num_labels}")

        model = AutoModelForTokenClassification.from_pretrained(
            model_config.model_name,
            num_labels=model_config.num_labels,
            id2label=id2tag,
            label2id=tag2id,
        )

        logger.info("模型创建成功")
        logger.info(f"模型结构:\n{model}")

        return model

    except Exception as e:
        logger.error(f"创建模型失败: {e}")
        raise


def save_model(
    model: AutoModelForTokenClassification,
    save_path: str,
    tag2id: Optional[Dict[str, int]] = None,
    id2tag: Optional[Dict[int, str]] = None,
) -> None:
    """
    保存训练好的模型

    Args:
        model: 要保存的模型
        save_path: 保存路径
        tag2id: 标签到ID的映射（可选）
        id2tag: ID到标签的映射（可选）

    Raises:
        ValueError: 输入参数无效
        Exception: 保存失败
    """
    if not model:
        raise ValueError("模型不能为空")

    if not save_path:
        raise ValueError("保存路径不能为空")

    try:
        # 确保保存目录存在
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"正在保存模型到: {save_path}")

        # 保存模型
        model.save_pretrained(save_path)

        # 如果提供了标签映射，也一起保存
        if tag2id and id2tag:
            import json

            mappings = {
                "tag2id": tag2id,
                "id2tag": {str(k): v for k, v in id2tag.items()},  # JSON要求key为字符串
                "num_labels": len(tag2id),
            }

            mapping_file = save_path / "label_mappings.json"
            with open(mapping_file, "w", encoding="utf-8") as f:
                json.dump(mappings, f, ensure_ascii=False, indent=2)

            logger.info(f"标签映射已保存到: {mapping_file}")

        logger.info("模型保存完成")

    except Exception as e:
        logger.error(f"保存模型失败: {e}")
        raise


def load_model(model_path: str, device: Optional[str] = None) -> tuple:
    """
    加载训练好的模型

    Args:
        model_path: 模型路径
        device: 设备类型（cpu/cuda）

    Returns:
        tuple: (model, tag2id, id2tag) 如果有标签映射文件的话

    Raises:
        FileNotFoundError: 模型文件不存在
        Exception: 加载失败
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    try:
        logger.info(f"正在加载模型: {model_path}")

        # 自动检测设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"使用设备: {device}")

        # 加载模型
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model.to(device)

        # 尝试加载标签映射
        tag2id, id2tag = None, None
        mapping_file = model_path / "label_mappings.json"

        if mapping_file.exists():
            import json

            logger.info("发现标签映射文件，正在加载...")

            with open(mapping_file, "r", encoding="utf-8") as f:
                mappings = json.load(f)

            tag2id = mappings.get("tag2id", {})
            id2tag_str = mappings.get("id2tag", {})
            # 将字符串key转换回整数
            id2tag = {int(k): v for k, v in id2tag_str.items()}

            logger.info(f"加载标签映射: {len(tag2id)} 个标签")

        logger.info("模型加载完成")

        return model, tag2id, id2tag

    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise


def get_model_info(model: AutoModelForTokenClassification) -> Dict[str, Any]:
    """
    获取模型信息

    Args:
        model: 模型实例

    Returns:
        模型信息字典
    """
    if not model:
        raise ValueError("模型不能为空")

    try:
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info = {
            "model_type": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "num_labels": (
                model.config.num_labels if hasattr(model.config, "num_labels") else None
            ),
            "model_size_mb": total_params * 4 / (1024 * 1024),  # 假设float32
        }

        # 获取设备信息
        if hasattr(model, "device"):
            info["device"] = str(model.device)

        logger.info("模型信息获取完成")
        return info

    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        raise


def freeze_layers(
    model: AutoModelForTokenClassification,
    freeze_embeddings: bool = True,
    freeze_encoder_layers: int = 0,
) -> None:
    """
    冻结模型层

    Args:
        model: 模型实例
        freeze_embeddings: 是否冻结embedding层
        freeze_encoder_layers: 冻结的编码器层数（从底层开始）

    Raises:
        ValueError: 输入参数无效
    """
    if not model:
        raise ValueError("模型不能为空")

    if freeze_encoder_layers < 0:
        raise ValueError("冻结层数不能为负数")

    try:
        frozen_params = 0

        # 冻结embedding层
        if freeze_embeddings:
            if hasattr(model, "bert"):
                for param in model.bert.embeddings.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
                logger.info("已冻结embedding层")
            elif hasattr(model, "roberta"):
                for param in model.roberta.embeddings.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
                logger.info("已冻结embedding层")

        # 冻结编码器层
        if freeze_encoder_layers > 0:
            encoder = None
            if hasattr(model, "bert"):
                encoder = model.bert.encoder
            elif hasattr(model, "roberta"):
                encoder = model.roberta.encoder

            if encoder and hasattr(encoder, "layer"):
                num_layers = len(encoder.layer)
                freeze_count = min(freeze_encoder_layers, num_layers)

                for i in range(freeze_count):
                    for param in encoder.layer[i].parameters():
                        param.requires_grad = False
                        frozen_params += param.numel()

                logger.info(f"已冻结前 {freeze_count} 个编码器层")

        logger.info(f"总共冻结了 {frozen_params:,} 个参数")

    except Exception as e:
        logger.error(f"冻结模型层失败: {e}")
        raise


def print_model_summary(model: AutoModelForTokenClassification) -> None:
    """
    打印模型摘要信息

    Args:
        model: 模型实例
    """
    try:
        info = get_model_info(model)

        print("\n" + "=" * 50)
        print("模型摘要信息")
        print("=" * 50)
        print(f"模型类型: {info.get('model_type', 'Unknown')}")
        print(f"总参数数: {info.get('total_parameters', 0):,}")
        print(f"可训练参数: {info.get('trainable_parameters', 0):,}")
        print(f"冻结参数: {info.get('frozen_parameters', 0):,}")
        print(f"标签数量: {info.get('num_labels', 'Unknown')}")
        print(f"模型大小: {info.get('model_size_mb', 0):.2f} MB")
        print(f"设备: {info.get('device', 'Unknown')}")
        print("=" * 50 + "\n")

    except Exception as e:
        logger.error(f"打印模型摘要失败: {e}")
