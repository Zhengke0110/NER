"""
验证相关工具函数
包含NER预测、数据读取、结果保存等功能
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import torch
import numpy as np
from transformers import AutoModelForTokenClassification, BertTokenizerFast

from .convert import get_token
from .config import ValidationConfig

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path: str, tokenizer_path: str):
    """
    加载模型和tokenizer

    Args:
        model_path: 模型路径
        tokenizer_path: tokenizer路径

    Returns:
        tuple: (model, tokenizer)
    """
    logger.info("正在加载模型和tokenizer...")
    try:
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        logger.info("模型和tokenizer加载成功")
        logger.info(f"模型配置: {model.config}")
        logger.info(f"标签映射: {model.config.id2label}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"模型或tokenizer加载失败: {e}")
        raise


def predict_ner(text: str, model, tokenizer) -> List[Dict[str, Any]]:
    """
    对文本进行NER预测，提取命名实体

    Args:
        text: 输入文本
        model: 训练好的模型
        tokenizer: tokenizer实例

    Returns:
        预测结果列表，每个元素包含实体信息
    """
    logger.info(f"开始处理文本: {text}")

    try:
        # 文本分词
        input_char = get_token(text)
        logger.info(f"分词结果: {input_char}")

        # tokenizer编码
        input_tensor = tokenizer(
            input_char,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            max_length=512,
            return_tensors="pt",
        )

        input_tokens = input_tensor.tokens()
        offsets = input_tensor["offset_mapping"]
        ignore_mask = offsets[0, :, 1] == 0

        logger.info(f"输入tokens: {input_tokens}")

        # 移除offset_mapping以避免模型报错
        input_tensor.pop("offset_mapping")

        # 模型预测
        model.eval()
        with torch.no_grad():
            outputs = model(**input_tensor)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
        predictions = outputs.logits.argmax(dim=-1)[0].tolist()

        logger.info(f"预测结果: {predictions}")

        # 处理预测结果
        results = []
        idx = 0

        while idx < len(predictions):
            if ignore_mask[idx]:
                idx += 1
                continue

            pred = predictions[idx]
            label = model.config.id2label[pred]

            if label != "O":
                # 移除B-或I-前缀
                entity_type = label[2:]
                start = idx
                end = start + 1

                # 收集所有相关token的分数
                all_scores = [probabilities[start][predictions[start]]]

                # 处理连续的I-标签
                while (
                    end < len(predictions)
                    and not ignore_mask[end]
                    and model.config.id2label[predictions[end]] == f"I-{entity_type}"
                ):
                    all_scores.append(probabilities[end][predictions[end]])
                    end += 1
                    idx += 1

                # 计算平均置信度
                score = np.mean(all_scores).item()

                # 提取实体词
                entity_tokens = input_tokens[start:end]
                entity_word = "".join(entity_tokens).replace("##", "")

                entity_info = {
                    "entity_group": entity_type,
                    "score": score,
                    "word": entity_word,
                    "tokens": entity_tokens,
                    "start": start,
                    "end": end,
                }

                results.append(entity_info)
                logger.info(f"发现实体: {entity_info}")

            idx += 1

        logger.info(f"总共发现 {len(results)} 个实体")
        return results

    except Exception as e:
        logger.error(f"预测过程中发生错误: {e}")
        raise


def batch_predict(texts: List[str], model, tokenizer) -> List[List[Dict[str, Any]]]:
    """
    批量进行NER预测

    Args:
        texts: 文本列表
        model: 训练好的模型
        tokenizer: tokenizer实例

    Returns:
        每个文本的预测结果列表
    """
    logger.info(f"开始批量处理 {len(texts)} 个文本")
    all_results = []

    for i, text in enumerate(texts, 1):
        logger.info(f"处理第 {i}/{len(texts)} 个文本")
        try:
            results = predict_ner(text, model, tokenizer)
            all_results.append(results)
        except Exception as e:
            logger.error(f"处理第 {i} 个文本时出错: {e}")
            all_results.append([])

    return all_results


def save_results(
    results: List[Dict[str, Any]], output_file: str, config: ValidationConfig = None
):
    """
    保存预测结果到文件

    Args:
        results: 预测结果列表
        output_file: 输出文件路径，如果为None则使用配置中的默认路径
        config: 验证配置对象
    """
    if output_file is None and config:
        output_file = config.get_output_path(config.single_output_file)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {output_file}")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")


def save_batch_results(
    all_results: List[List[Dict[str, Any]]],
    texts: List[str],
    output_file: str,
    mode: str = "batch",
):
    """
    保存批量预测结果

    Args:
        all_results: 所有文本的预测结果
        texts: 输入文本列表
        output_file: 输出文件路径
        mode: 处理模式
    """
    batch_output = {
        "mode": mode,
        "total_texts": len(texts),
        "timestamp": str(datetime.now()),
        "results": [],
    }

    for i, (text, results) in enumerate(zip(texts, all_results)):
        batch_output["results"].append(
            {
                "id": i + 1,
                "text": text,
                "entities": results,
                "entity_count": len(results),
            }
        )

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(batch_output, f, ensure_ascii=False, indent=2)
        logger.info(f"批量结果已保存到: {output_file}")
    except Exception as e:
        logger.error(f"保存批量结果失败: {e}")


def read_test_data_from_file(
    test_file: str, validation_config: ValidationConfig = None
) -> List[str]:
    """
    从文件读取测试数据，如果文件不存在则使用默认测试文本

    Args:
        test_file: 测试文件路径
        validation_config: 验证配置对象

    Returns:
        测试文本列表
    """
    # 首先尝试从指定的验证集文件读取
    if test_file and os.path.exists(test_file):
        logger.info(f"从验证集文件读取数据: {test_file}")
        try:
            test_texts = []
            with open(test_file, "r", encoding="utf-8") as f:
                current_text = []
                for line in f:
                    line = line.strip()
                    if not line:  # 空行表示一个句子结束
                        if current_text:
                            # 处理BIO格式，提取文本部分
                            if "\t" in current_text[0] or " " in current_text[0]:
                                # BIO格式：词 标签
                                text = "".join(
                                    [token.split()[0] for token in current_text]
                                )
                            else:
                                # 纯文本格式
                                text = "".join(current_text)
                            test_texts.append(text)
                            current_text = []
                    else:
                        current_text.append(line)

                # 处理最后一个句子
                if current_text:
                    if "\t" in current_text[0] or " " in current_text[0]:
                        text = "".join([token.split()[0] for token in current_text])
                    else:
                        text = "".join(current_text)
                    test_texts.append(text)

            logger.info(f"从验证集文件读取到 {len(test_texts)} 个测试样本")
            return test_texts

        except Exception as e:
            logger.error(f"读取验证集文件失败: {e}")
    else:
        logger.warning(f"验证集文件不存在: {test_file}")

    # 如果无法从文件读取，使用默认测试文本
    if validation_config and validation_config.default_test_texts:
        logger.info("使用配置中的默认测试文本")
        return validation_config.default_test_texts
    else:
        logger.info("使用内置默认测试文本")
        return [
            "2009年高考在北京的报名费是2009元",
            "2020年研究生考试在上海进行",
            "明年的公务员考试将在广州举办",
            "2018年高考报名费用是100元",
            "去年的期末考试在深圳大学举行",
        ]


def get_test_data(
    validation_config: ValidationConfig, test_file: str = None
) -> List[str]:
    """
    获取测试数据，优先使用指定的验证集文件，文件不存在时使用默认测试文本

    Args:
        validation_config: 验证配置对象
        test_file: 可选的测试文件路径

    Returns:
        测试文本列表
    """
    # 优先使用用户指定的测试文件
    if test_file:
        return read_test_data_from_file(test_file, validation_config)

    # 尝试使用配置中的验证集路径
    val_file_path = validation_config.get_val_file_path()
    if val_file_path:
        return read_test_data_from_file(val_file_path, validation_config)

    # 如果验证集文件不存在，使用默认测试文本
    logger.info("验证集文件不存在，使用默认测试文本")
    return read_test_data_from_file(None, validation_config)


def display_single_results(results: List[Dict[str, Any]]):
    """
    显示单个文本的预测结果

    Args:
        results: 预测结果列表
    """
    print("\n=== 预测结果 ===")
    if results:
        for i, result in enumerate(results, 1):
            print(f"{i}. 实体: {result['word']}")
            print(f"   类型: {result['entity_group']}")
            print(f"   置信度: {result['score']:.4f}")
            print(f"   位置: {result['start']}-{result['end']}")
            print(f"   Tokens: {result['tokens']}")
            print()
    else:
        print("未发现任何实体")


def display_batch_results(texts: List[str], all_results: List[List[Dict[str, Any]]]):
    """
    显示批量预测结果

    Args:
        texts: 输入文本列表
        all_results: 所有文本的预测结果
    """
    print("\n=== 批量预测结果 ===")
    for i, (text, results) in enumerate(zip(texts, all_results), 1):
        print(f"\n{i}. 文本: {text}")
        if results:
            for j, result in enumerate(results, 1):
                print(
                    f"   {j}. [{result['entity_group']}] {result['word']} (置信度: {result['score']:.4f})"
                )
        else:
            print("   未发现实体")


def interactive_validation_loop(model, tokenizer):
    """
    交互式验证循环

    Args:
        model: 训练好的模型
        tokenizer: tokenizer实例
    """
    logger.info("进入交互式验证模式")
    print("\n=== 交互式NER验证 ===")
    print("输入文本进行实体识别，输入 'quit' 退出")

    while True:
        try:
            text = input("\n请输入文本: ").strip()

            if text.lower() in ["quit", "exit", "q"]:
                print("退出验证")
                break

            if not text:
                print("请输入有效文本")
                continue

            # 进行预测
            results = predict_ner(text, model, tokenizer)

            # 显示结果
            if results:
                print(f"\n发现 {len(results)} 个实体:")
                for i, result in enumerate(results, 1):
                    print(
                        f"{i}. [{result['entity_group']}] {result['word']} (置信度: {result['score']:.4f})"
                    )
            else:
                print("未发现任何实体")

        except KeyboardInterrupt:
            print("\n\n用户中断，退出验证")
            break
        except Exception as e:
            logger.error(f"处理过程中出错: {e}")
            print(f"处理出错: {e}")
