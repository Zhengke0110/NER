"""
数据处理工具模块
包含数据读取、标签编码、数据集创建等功能
"""

import re
import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from transformers import BertTokenizerFast
from torch.utils.data import Dataset

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_data(file_path: str, encoding: str = "UTF-8") -> Tuple[List[List[str]], List[List[str]]]:
    """
    读取NER标注数据文件
    
    Args:
        file_path: 数据文件路径
        encoding: 文件编码格式
        
    Returns:
        tuple: (token_docs, tag_docs) - 分词结果和标签结果
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误
        UnicodeDecodeError: 编码错误
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        logger.info(f"正在读取数据文件: {file_path}")
        
        raw_text = file_path.read_text(encoding=encoding).strip()
        if not raw_text:
            raise ValueError(f"数据文件为空: {file_path}")
        
        # 按空行分割文档
        raw_docs = re.split(r'\n\t?\n', raw_text)
        
        token_docs = []
        tag_docs = []
        
        for doc_idx, doc in enumerate(raw_docs):
            if not doc.strip():
                continue
                
            tokens = []
            tags = []
            
            for line_idx, line in enumerate(doc.split('\n')):
                line = line.strip()
                if not line:
                    continue
                    
                # 分割token和tag
                parts = line.split(' ')
                if len(parts) != 2:
                    logger.warning(f"第{doc_idx+1}个文档，第{line_idx+1}行格式错误: {line}")
                    continue
                    
                token, tag = parts
                tokens.append(token)
                tags.append(tag)
            
            if tokens and tags:
                token_docs.append(tokens)
                tag_docs.append(tags)
        
        logger.info(f"成功读取 {len(token_docs)} 个文档")
        return token_docs, tag_docs
        
    except UnicodeDecodeError as e:
        logger.error(f"文件编码错误: {e}")
        raise
    except Exception as e:
        logger.error(f"读取文件时发生错误: {e}")
        raise


def create_label_mappings(train_tags: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    """
    创建标签映射
    
    Args:
        train_tags: 训练集标签
        
    Returns:
        tuple: (tag2id, id2tag, label_list)
        
    Raises:
        ValueError: 当标签为空时
    """
    if not train_tags:
        raise ValueError("训练标签不能为空")
    
    # 收集所有唯一标签
    unique_tags = set()
    for doc_tags in train_tags:
        unique_tags.update(doc_tags)
    
    if not unique_tags:
        raise ValueError("没有找到任何标签")
    
    # 创建映射
    unique_tags = sorted(list(unique_tags))  # 保证顺序一致
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    label_list = list(unique_tags)
    
    logger.info(f"创建标签映射，共 {len(unique_tags)} 个标签: {unique_tags}")
    
    return tag2id, id2tag, label_list


def encode_tags(tags, encodings, tag2id):
    """
    修复后的标签编码函数
    
    Args:
        tags: 原始标签列表，每个元素是一个文档的标签列表
        encodings: tokenizer编码结果
        tag2id: 标签到ID的映射
    
    Returns:
        编码后的标签列表
    """
    if not tags or not hasattr(encodings, 'offset_mapping'):
        raise ValueError("无效的输入参数")
    
    if not tag2id:
        raise ValueError("标签映射不能为空")
    
    # 将标签转换为ID
    try:
        labels = [[tag2id[tag] for tag in doc] for doc in tags]
    except KeyError as e:
        logger.error(f"发现未知标签: {e}")
        raise ValueError(f"发现未知标签: {e}")
    
    encoded_labels = []
    
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # 创建全由-100组成的数组（用于忽略loss计算）
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        
        # 追踪当前处理到的原始token位置
        token_idx = 0
        
        for i, (start, end) in enumerate(doc_offset):
            # 跳过特殊token：[CLS], [SEP], [PAD]等
            # 这些token的offset是(0, 0)
            if start == 0 and end == 0:
                continue
            
            # 只为每个原始token的第一个subword分配标签
            # BERT会将一个中文字符分为一个token，但英文单词可能被分为多个subword
            # 我们只为第一个subword分配标签，其他subword保持-100
            if start == 0 and end > 0:
                # 这是一个真实token的开始
                if token_idx < len(doc_labels):
                    doc_enc_labels[i] = doc_labels[token_idx]
                    token_idx += 1
        
        # 检查是否所有标签都被处理
        if token_idx < len(doc_labels):
            logger.warning(f"标签数量({len(doc_labels)})超过有效token数量({token_idx})")
        
        encoded_labels.append(doc_enc_labels.tolist())
    
    logger.info(f"成功编码 {len(encoded_labels)} 个文档的标签")
    return encoded_labels
def create_tokenizer(model_name: str) -> BertTokenizerFast:
    """
    创建tokenizer
    
    Args:
        model_name: 模型名称
        
    Returns:
        tokenizer实例
        
    Raises:
        Exception: 创建tokenizer失败
    """
    try:
        logger.info(f"正在创建tokenizer: {model_name}")
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        logger.info("tokenizer创建成功")
        return tokenizer
    except Exception as e:
        logger.error(f"创建tokenizer失败: {e}")
        raise


def encode_texts(texts: List[List[str]], tokenizer: BertTokenizerFast, 
                 max_length: int = 512, return_offsets: bool = True) -> Any:
    """
    使用tokenizer编码文本
    
    Args:
        texts: 文本序列
        tokenizer: tokenizer实例
        max_length: 最大长度
        return_offsets: 是否返回offset mapping
        
    Returns:
        编码结果
        
    Raises:
        ValueError: 输入参数无效
    """
    if not texts:
        raise ValueError("文本序列不能为空")
    
    if not tokenizer:
        raise ValueError("tokenizer不能为空")
    
    try:
        logger.info(f"正在编码 {len(texts)} 个文本序列")
        
        encodings = tokenizer(
            texts,
            is_split_into_words=True,
            return_offsets_mapping=return_offsets,
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        logger.info("文本编码完成")
        return encodings
        
    except Exception as e:
        logger.error(f"文本编码失败: {e}")
        raise


class NerDataset(Dataset):
    """NER数据集类"""
    
    def __init__(self, encodings: Any, labels: List[List[int]]):
        """
        初始化数据集
        
        Args:
            encodings: tokenizer编码结果
            labels: 编码后的标签
            
        Raises:
            ValueError: 输入参数无效
        """
        if not encodings or not labels:
            raise ValueError("编码结果和标签不能为空")
        
        if len(encodings['input_ids']) != len(labels):
            raise ValueError(f"编码结果数量 {len(encodings['input_ids'])} 与标签数量 {len(labels)} 不匹配")
        
        self.encodings = encodings
        self.labels = labels
        
        logger.info(f"创建NER数据集，包含 {len(labels)} 个样本")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        if idx >= len(self.labels):
            raise IndexError(f"索引 {idx} 超出范围")
        
        try:
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        except Exception as e:
            logger.error(f"获取样本 {idx} 时出错: {e}")
            raise

    def __len__(self) -> int:
        """获取数据集大小"""
        return len(self.labels)


def prepare_datasets(data_config, model_config) -> Tuple[NerDataset, NerDataset, Dict[str, int], Dict[int, str], List[str]]:
    """
    准备训练和验证数据集
    
    Args:
        data_config: 数据配置
        model_config: 模型配置
        
    Returns:
        tuple: (train_dataset, val_dataset, tag2id, id2tag, label_list)
    """
    try:
        # 读取数据
        train_texts, train_tags = read_data(data_config.get_file_path("train"), data_config.encoding)
        val_texts, val_tags = read_data(data_config.get_file_path("val"), data_config.encoding)
        
        # 创建标签映射
        tag2id, id2tag, label_list = create_label_mappings(train_tags)
        
        # 更新模型配置中的标签数量
        model_config.num_labels = len(label_list)
        
        # 创建tokenizer
        tokenizer = create_tokenizer(model_config.model_name)
        
        # 编码文本
        train_encodings = encode_texts(train_texts, tokenizer, data_config.max_length)
        val_encodings = encode_texts(val_texts, tokenizer, data_config.max_length)
        
        # 编码标签
        train_labels = encode_tags(train_tags, train_encodings, tag2id)
        val_labels = encode_tags(val_tags, val_encodings, tag2id)
        
        # 移除offset_mapping（训练时不需要）
        train_encodings.pop("offset_mapping", None)
        val_encodings.pop("offset_mapping", None)
        
        # 创建数据集
        train_dataset = NerDataset(train_encodings, train_labels)
        val_dataset = NerDataset(val_encodings, val_labels)
        
        logger.info("数据集准备完成")
        return train_dataset, val_dataset, tag2id, id2tag, label_list
        
    except Exception as e:
        logger.error(f"准备数据集时发生错误: {e}")
        raise
