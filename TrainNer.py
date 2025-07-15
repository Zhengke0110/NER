"""
重构后的NER训练主文件
使用模块化的工具函数进行训练
"""

import logging
import sys
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置Hugging Face镜像

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAINING_CONFIG,
)
from utils.dataUtils import prepare_datasets
from utils.modelUtils import create_model, print_model_summary
from utils.trainingUtils import (
    create_compute_metrics_fn,
    create_training_arguments,
    create_trainer,
    run_training,
)

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """主训练函数"""
    try:
        logger.info("开始NER模型训练流程")

        # 准备数据集
        logger.info("准备数据集...")
        train_dataset, val_dataset, tag2id, id2tag, label_list = prepare_datasets(
            DEFAULT_DATA_CONFIG, DEFAULT_MODEL_CONFIG
        )

        logger.info(f"训练集样本数: {len(train_dataset)}")
        logger.info(f"验证集样本数: {len(val_dataset)}")
        logger.info(f"标签列表: {label_list}")

        # 创建模型
        logger.info("创建模型...")
        model = create_model(DEFAULT_MODEL_CONFIG, tag2id, id2tag)

        # 打印模型摘要
        print_model_summary(model)

        # 创建评估函数
        logger.info("创建评估函数...")
        compute_metrics_fn = create_compute_metrics_fn(label_list)

        # 创建训练参数
        logger.info("创建训练参数...")
        training_args = create_training_arguments(DEFAULT_TRAINING_CONFIG)

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
            training_config=DEFAULT_TRAINING_CONFIG,
            model_config=DEFAULT_MODEL_CONFIG,
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
