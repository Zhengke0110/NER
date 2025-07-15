"""
重构后的NER训练主文件
使用模块化的工具函数进行训练
"""

import logging
import sys
import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置Hugging Face镜像

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
)
from utils.dataUtils import prepare_datasets
from utils.modelUtils import create_model, print_model_summary
from utils.trainingUtils import (
    create_compute_metrics_fn,
    create_training_arguments,
    create_trainer,
    run_training,
)

# 确保logs目录存在
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# 设置日志 - 同时输出到控制台和文件
log_handlers = [
    logging.FileHandler("logs/training.log", encoding="utf-8"),
    logging.StreamHandler(sys.stdout)
]

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=log_handlers,
    force=True  # 强制重新配置日志
)
logger = logging.getLogger(__name__)


def main():
    """主训练函数"""
    try:
        logger.info("开始NER模型训练流程")

        # 从config.json加载配置
        logger.info("加载配置文件...")
        data_config = DataConfig.from_json_with_fallback("config.json", "data")
        model_config = ModelConfig.from_json_with_fallback("config.json", "model")
        training_config = TrainingConfig.from_json_with_fallback(
            "config.json", "training"
        )

        logger.info("配置加载完成")
        logger.info(f"数据配置: {data_config.to_dict()}")
        logger.info(f"模型配置: {model_config.to_dict()}")

        # 准备数据集
        logger.info("准备数据集...")
        train_dataset, val_dataset, tag2id, id2tag, label_list = prepare_datasets(
            data_config, model_config
        )

        logger.info(f"训练集样本数: {len(train_dataset)}")
        logger.info(f"验证集样本数: {len(val_dataset)}")
        logger.info(f"标签列表: {label_list}")

        # 创建模型
        logger.info("创建模型...")
        model = create_model(model_config, tag2id, id2tag)

        # 打印模型摘要
        print_model_summary(model)

        # 创建评估函数
        logger.info("创建评估函数...")
        compute_metrics_fn = create_compute_metrics_fn(label_list)

        # 创建训练参数
        logger.info("创建训练参数...")
        training_args = create_training_arguments(training_config)

        # 创建Trainer
        logger.info("创建Trainer...")
        trainer = create_trainer(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics_fn=compute_metrics_fn,
        )

        # 运行训练
        logger.info("开始训练...")
        results = run_training(
            trainer=trainer,
            training_config=training_config,
            model_config=model_config,
            tag2id=tag2id,
            id2tag=id2tag,
        )

        # 打印最终结果
        logger.info("=" * 60)
        logger.info("训练完成! 最终结果:")
        logger.info("=" * 60)
        for metric, value in results["final_metrics"].items():
            logger.info(f"{metric}: {value:.4f}")
        logger.info(f"模型已保存到: {results['model_save_path']}")
        logger.info("=" * 60)

        return results

    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
