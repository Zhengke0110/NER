"""
训练相关工具模块
包含评估指标、训练参数、Trainer设置等功能
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from transformers import TrainingArguments, Trainer
import evaluate

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_compute_metrics_fn(label_list: List[str], metric_name: str = "seqeval"):
    """
    创建评估指标计算函数
    
    Args:
        label_list: 标签列表
        metric_name: 评估指标名称
        
    Returns:
        评估函数
        
    Raises:
        ValueError: 输入参数无效
        Exception: 加载评估指标失败
    """
    if not label_list:
        raise ValueError("标签列表不能为空")
    
    try:
        logger.info(f"正在加载评估指标: {metric_name}")
        metric = evaluate.load(metric_name)
        logger.info("评估指标加载成功")
        
        def compute_metrics(eval_pred):
            """
            计算评估指标
            
            Args:
                eval_pred: 包含predictions和labels的元组
                
            Returns:
                评估指标字典
            """
            try:
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=2)

                # 移除忽略的标签(-100)并转换为标签名称
                true_predictions = []
                true_labels = []
                
                for prediction, label in zip(predictions, labels):
                    true_pred = []
                    true_label = []
                    
                    for p, l in zip(prediction, label):
                        if l != -100:  # 忽略padding和特殊token
                            if 0 <= p < len(label_list) and 0 <= l < len(label_list):
                                true_pred.append(label_list[p])
                                true_label.append(label_list[l])
                            else:
                                logger.warning(f"标签索引超出范围: pred={p}, label={l}")
                    
                    if true_pred and true_label:
                        true_predictions.append(true_pred)
                        true_labels.append(true_label)

                if not true_predictions or not true_labels:
                    logger.warning("没有有效的预测结果用于评估")
                    return {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "accuracy": 0.0,
                    }

                # 计算评估指标
                results = metric.compute(predictions=true_predictions, references=true_labels)
                
                return {
                    "precision": results.get("overall_precision", 0.0),
                    "recall": results.get("overall_recall", 0.0),
                    "f1": results.get("overall_f1", 0.0),
                    "accuracy": results.get("overall_accuracy", 0.0),
                }
                
            except Exception as e:
                logger.error(f"计算评估指标时出错: {e}")
                return {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "accuracy": 0.0,
                }
        
        return compute_metrics
        
    except Exception as e:
        logger.error(f"创建评估函数失败: {e}")
        raise


def create_training_arguments(training_config) -> TrainingArguments:
    """
    创建训练参数
    
    Args:
        training_config: 训练配置
        
    Returns:
        训练参数对象
        
    Raises:
        ValueError: 配置参数无效
        Exception: 创建训练参数失败
    """
    if not training_config:
        raise ValueError("训练配置不能为空")
    
    try:
        logger.info("正在创建训练参数")
        
        # 验证关键参数
        if training_config.num_train_epochs <= 0:
            raise ValueError("训练轮数必须大于0")
        
        if training_config.per_device_train_batch_size <= 0:
            raise ValueError("训练批次大小必须大于0")
        
        training_args = TrainingArguments(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.num_train_epochs,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            learning_rate=training_config.learning_rate,
            warmup_steps=training_config.warmup_steps,
            weight_decay=training_config.weight_decay,
            logging_dir=training_config.logging_dir,
            logging_steps=training_config.logging_steps,
            save_strategy=training_config.save_strategy,
            save_steps=training_config.save_steps,
            save_total_limit=training_config.save_total_limit,
            eval_strategy=training_config.eval_strategy,
            eval_steps=training_config.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=None,  # 禁用wandb等日志记录
        )
        
        logger.info("训练参数创建成功")
        return training_args
        
    except Exception as e:
        logger.error(f"创建训练参数失败: {e}")
        raise


def create_trainer(model, training_args, train_dataset, eval_dataset, 
                   compute_metrics_fn, tokenizer=None) -> Trainer:
    """
    创建Trainer对象
    
    Args:
        model: 模型实例
        training_args: 训练参数
        train_dataset: 训练数据集
        eval_dataset: 验证数据集
        compute_metrics_fn: 评估函数
        tokenizer: tokenizer实例（可选）
        
    Returns:
        Trainer对象
        
    Raises:
        ValueError: 输入参数无效
        Exception: 创建Trainer失败
    """
    if not all([model, training_args, train_dataset, eval_dataset, compute_metrics_fn]):
        raise ValueError("所有必需参数都不能为空")
    
    try:
        logger.info("正在创建Trainer")
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_fn,
            tokenizer=tokenizer,
        )
        
        logger.info("Trainer创建成功")
        return trainer
        
    except Exception as e:
        logger.error(f"创建Trainer失败: {e}")
        raise


class TrainingCallback:
    """自定义训练回调类"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        初始化回调
        
        Args:
            patience: 早停耐心值
            min_delta: 最小改进阈值
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """评估时的回调"""
        if logs is None:
            return
        
        current_score = logs.get("eval_f1", 0)
        
        if self.best_score is None:
            self.best_score = current_score
        elif current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            logger.info(f"Early stopping at epoch {state.epoch}")
            control.should_training_stop = True
            self.stopped_epoch = state.epoch


def log_training_info(trainer: Trainer, training_config) -> None:
    """
    记录训练信息
    
    Args:
        trainer: Trainer对象
        training_config: 训练配置
    """
    try:
        logger.info("="*50)
        logger.info("训练信息")
        logger.info("="*50)
        logger.info(f"输出目录: {training_config.output_dir}")
        logger.info(f"训练轮数: {training_config.num_train_epochs}")
        logger.info(f"训练批次大小: {training_config.per_device_train_batch_size}")
        logger.info(f"验证批次大小: {training_config.per_device_eval_batch_size}")
        logger.info(f"学习率: {training_config.learning_rate}")
        logger.info(f"权重衰减: {training_config.weight_decay}")
        logger.info(f"预热步数: {training_config.warmup_steps}")
        logger.info(f"日志记录步数: {training_config.logging_steps}")
        logger.info(f"保存策略: {training_config.save_strategy}")
        logger.info(f"保存步数: {training_config.save_steps}")
        logger.info(f"评估策略: {training_config.eval_strategy}")
        logger.info(f"评估步数: {training_config.eval_steps}")
        
        if hasattr(trainer, 'train_dataset'):
            logger.info(f"训练样本数: {len(trainer.train_dataset)}")
        if hasattr(trainer, 'eval_dataset'):
            logger.info(f"验证样本数: {len(trainer.eval_dataset)}")
        
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"记录训练信息失败: {e}")


def run_training(trainer: Trainer, training_config, model_config, 
                 tag2id: Dict[str, int], id2tag: Dict[int, str]) -> Dict[str, Any]:
    """
    运行完整的训练流程
    
    Args:
        trainer: Trainer对象
        training_config: 训练配置
        model_config: 模型配置
        tag2id: 标签到ID映射
        id2tag: ID到标签映射
        
    Returns:
        训练结果字典
        
    Raises:
        Exception: 训练过程中出现错误
    """
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.modelUtils import save_model
        
        logger.info("开始训练...")
        log_training_info(trainer, training_config)
        
        # 执行训练
        train_result = trainer.train()
        
        logger.info("训练完成，开始最终评估...")
        
        # 执行最终评估
        eval_result = trainer.evaluate()
        
        logger.info("评估完成，正在保存模型...")
        
        # 保存模型
        save_path = training_config.get_checkpoint_path(
            model_config.model_name.split('/')[-1], 
            training_config.num_train_epochs
        )
        
        save_model(trainer.model, save_path, tag2id, id2tag)
        
        # 整理结果
        results = {
            "train_result": train_result,
            "eval_result": eval_result,
            "model_save_path": save_path,
            "final_metrics": {
                "train_loss": train_result.training_loss,
                "eval_precision": eval_result.get("eval_precision", 0),
                "eval_recall": eval_result.get("eval_recall", 0),
                "eval_f1": eval_result.get("eval_f1", 0),
                "eval_accuracy": eval_result.get("eval_accuracy", 0),
            }
        }
        
        logger.info("训练流程完成!")
        logger.info(f"最终评估结果: {results['final_metrics']}")
        
        return results
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise


def predict_single_text(model, tokenizer, text: List[str], 
                       id2tag: Dict[int, str], device: str = "cpu") -> List[str]:
    """
    对单个文本进行NER预测
    
    Args:
        model: 训练好的模型
        tokenizer: tokenizer实例
        text: 分词后的文本列表
        id2tag: ID到标签的映射
        device: 设备类型
        
    Returns:
        预测的标签列表
        
    Raises:
        ValueError: 输入参数无效
        Exception: 预测失败
    """
    if not all([model, tokenizer, text, id2tag]):
        raise ValueError("所有参数都不能为空")
    
    if not isinstance(text, list):
        raise ValueError("文本必须是分词后的列表")
    
    try:
        import torch
        
        # 设置模型为评估模式
        model.eval()
        model.to(device)
        
        # 编码文本
        encoding = tokenizer(
            text,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # 解码预测结果
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        predictions = predictions[0].cpu().numpy()
        
        # 提取真实token的预测结果
        predicted_labels = []
        word_ids = encoding.word_ids(batch_index=0)
        
        previous_word_id = None
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id != previous_word_id:
                predicted_labels.append(id2tag.get(predictions[i], "O"))
            previous_word_id = word_id
        
        return predicted_labels
        
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise
