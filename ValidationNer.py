import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置Hugging Face镜像

import torch
import numpy as np

# 导入自定义工具模块
from utils.convert import get_token
from utils.config import DataConfig, ValidationConfig
from transformers import AutoModelForTokenClassification, BertTokenizerFast

# 加载配置
validation_config = ValidationConfig.from_json_with_fallback(
    "config.json", "validation"
)

# 配置日志
log_handlers = [
    logging.FileHandler(validation_config.log_file, encoding="utf-8"),
    logging.StreamHandler(),
]

logging.basicConfig(
    level=getattr(logging, validation_config.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=log_handlers,
    force=True,  # 强制重新配置
)

# 确保日志立即写入文件
for handler in log_handlers:
    if isinstance(handler, logging.FileHandler):
        handler.setLevel(getattr(logging, validation_config.log_level))

logger = logging.getLogger(__name__)

# 验证配置
if not validation_config.validate_paths():
    logger.error("配置验证失败，退出程序")
    exit(1)

# 模型和tokenizer配置
MODEL_PATH = validation_config.model_path
TOKENIZER_PATH = validation_config.tokenizer_path

# 加载模型和tokenizer
logger.info("正在加载模型和tokenizer...")
try:
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
    logger.info("模型和tokenizer加载成功")
    logger.info(f"模型配置: {model.config}")
    logger.info(f"标签映射: {model.config.id2label}")
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


def save_results(results: List[Dict[str, Any]], output_file: str = None):
    """
    保存预测结果到文件

    Args:
        results: 预测结果列表
        output_file: 输出文件路径，如果为None则使用配置中的默认路径
    """
    if output_file is None:
        output_file = validation_config.get_output_path(
            validation_config.single_output_file
        )

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {output_file}")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")


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


def interactive_validation():
    """
    交互式验证模式
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


def validate_test_data(test_file: str = None):
    """
    使用测试数据集进行验证

    Args:
        test_file: 测试文件路径，如果为None则使用配置中的测试文件
    """
    try:
        # 加载数据配置
        data_config = DataConfig()

        if test_file is None:
            test_file = data_config.get_file_path("test")

        logger.info(f"使用测试文件: {test_file}")

        if not os.path.exists(test_file):
            logger.warning(f"测试文件不存在: {test_file}")
            return

        # 读取测试数据
        test_texts = []
        with open(test_file, "r", encoding="utf-8") as f:
            current_text = []
            for line in f:
                line = line.strip()
                if not line:  # 空行表示一个句子结束
                    if current_text:
                        text = "".join([token.split("\t")[0] for token in current_text])
                        test_texts.append(text)
                        current_text = []
                else:
                    current_text.append(line)

            # 处理最后一个句子
            if current_text:
                text = "".join([token.split("\t")[0] for token in current_text])
                test_texts.append(text)

        logger.info(f"读取到 {len(test_texts)} 个测试样本")

        # 批量预测
        all_results = batch_predict(
            test_texts[:5], model, tokenizer
        )  # 只处理前5个样本作为示例

        # 保存批量结果
        batch_output = {
            "test_file": test_file,
            "total_samples": len(test_texts),
            "processed_samples": len(all_results),
            "results": [],
        }

        for i, (text, results) in enumerate(zip(test_texts[:5], all_results)):
            batch_output["results"].append(
                {"sample_id": i + 1, "text": text, "entities": results}
            )

        with open("batch_validation_results.json", "w", encoding="utf-8") as f:
            json.dump(batch_output, f, ensure_ascii=False, indent=2)

        logger.info("批量验证结果已保存到: batch_validation_results.json")

    except Exception as e:
        logger.error(f"测试数据验证失败: {e}")


if __name__ == "__main__":
    import argparse

    # 命令行参数解析
    parser = argparse.ArgumentParser(description="NER模型验证工具")
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "interactive", "test"],
        default="single",
        help="验证模式",
    )
    parser.add_argument("--text", type=str, help="单个文本验证时的输入文本")
    parser.add_argument("--test-file", type=str, help="测试文件路径")
    parser.add_argument(
        "--output", type=str, default="validation_results.json", help="输出文件路径"
    )

    args = parser.parse_args()

    logger.info(f"验证模式: {args.mode}")

    try:
        if args.mode == "single":
            # 单个文本验证
            test_text = args.text or "2009年高考在北京的报名费是2009元"
            logger.info(f"单个文本验证: {test_text}")

            results = predict_ner(test_text, model, tokenizer)

            # 打印结果
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

            # 保存结果
            output_path = args.output
            if args.output == "validation_results.json":  # 如果是默认值，使用配置路径
                output_path = validation_config.get_output_path(
                    validation_config.single_output_file
                )
            save_results(results, output_path)

        elif args.mode == "batch":
            # 批量文本验证
            test_texts = validation_config.default_test_texts

            logger.info(f"批量验证 {len(test_texts)} 个文本")
            all_results = batch_predict(test_texts, model, tokenizer)

            # 显示结果
            print("\n=== 批量预测结果 ===")
            for i, (text, results) in enumerate(zip(test_texts, all_results), 1):
                print(f"\n{i}. 文本: {text}")
                if results:
                    for j, result in enumerate(results, 1):
                        print(
                            f"   {j}. [{result['entity_group']}] {result['word']} (置信度: {result['score']:.4f})"
                        )
                else:
                    print("   未发现实体")

            # 保存批量结果
            batch_output = {
                "mode": "batch",
                "total_texts": len(test_texts),
                "timestamp": str(datetime.now()),
                "results": [],
            }

            for i, (text, results) in enumerate(zip(test_texts, all_results)):
                batch_output["results"].append(
                    {
                        "id": i + 1,
                        "text": text,
                        "entities": results,
                        "entity_count": len(results),
                    }
                )

            # 使用配置中的输出路径
            output_path = validation_config.get_output_path(
                validation_config.batch_output_file
            )
            if args.output != "validation_results.json":  # 如果用户指定了输出文件
                output_path = args.output

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(batch_output, f, ensure_ascii=False, indent=2)

            logger.info(f"批量结果已保存到: {output_path}")

        elif args.mode == "interactive":
            # 交互式验证
            interactive_validation()

        elif args.mode == "test":
            # 测试数据集验证
            validate_test_data(args.test_file)

        logger.info("验证完成")

    except Exception as e:
        logger.error(f"验证过程中发生错误: {e}")
        raise
