import json
import os
from typing import List
import logging

# 获取日志记录器，但不配置基本设置（让主脚本来配置）
logger = logging.getLogger(__name__)

# 使用最常用的BIO标注方法
# B-X 代表实体X的开头，I-X代表实体X的内部，O代表不属于任何实体类型
def get_token(text: str) -> List[str]:
    """
    将文本分词，英文字母作为单词处理，其他字符单独处理
    
    Args:
        text: 输入文本
        
    Returns:
        分词后的列表
        
    Raises:
        ValueError: 当输入为空或None时
    """
    if not text:
        raise ValueError("输入文本不能为空")
    
    english_letters = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    output = []
    buffer = ''
    
    for char in text:
        if char in english_letters:
            buffer += char
        else:
            if buffer:
                output.append(buffer)
                buffer = ''
            if char.strip():  # 忽略空白字符
                output.append(char)
    
    if buffer:
        output.append(buffer)
    
    return output

def json2bio(input_file: str, output_file: str, split_by: str = 's') -> None:
    """
    将JSON格式的标注数据转换为BIO格式
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出BIO格式文件路径
        split_by: 分割方式（暂未使用）
        
    Raises:
        FileNotFoundError: 当输入文件不存在时
        ValueError: 当文件格式错误时
        IOError: 当文件读写出错时
    """
    # 参数验证
    if not input_file or not output_file:
        raise ValueError("输入和输出文件路径不能为空")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            line_count = 0
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 解析JSON
                    annotations = json.loads(line)
                    
                    # 验证必要字段
                    if 'text' not in annotations or 'label' not in annotations:
                        logger.warning(f"第{line_num}行缺少必要字段: {line}")
                        continue
                    
                    # 预处理文本
                    text = annotations['text'].replace('\n', ' ')
                    if not text.strip():
                        logger.warning(f"第{line_num}行文本为空")
                        continue
                    
                    # 分词
                    tokens = get_token(text.replace(' ', ','))
                    if not tokens:
                        logger.warning(f"第{line_num}行分词结果为空")
                        continue
                    
                    # 初始化标签
                    labels = ['O'] * len(tokens)
                    
                    # 处理实体标注
                    for entity in annotations['label']:
                        if not isinstance(entity, (list, tuple)) or len(entity) < 3:
                            logger.warning(f"第{line_num}行实体标注格式错误: {entity}")
                            continue
                        
                        try:
                            start_pos = int(entity[0])
                            end_pos = int(entity[1])
                            entity_type = str(entity[2]).strip()
                            
                            # 边界检查
                            if start_pos < 0 or end_pos < 0 or start_pos >= len(tokens):
                                logger.warning(f"第{line_num}行实体位置超出范围: {entity}")
                                continue
                            
                            if start_pos > end_pos:
                                logger.warning(f"第{line_num}行实体起始位置大于结束位置: {entity}")
                                continue
                            
                            if not entity_type:
                                logger.warning(f"第{line_num}行实体类型为空: {entity}")
                                continue
                            
                            # 调整结束位置，确保不超出范围
                            end_pos = min(end_pos, len(tokens))
                            
                            # 设置BIO标签
                            labels[start_pos] = f'B-{entity_type}'
                            for pos in range(start_pos + 1, end_pos):
                                labels[pos] = f'I-{entity_type}'
                                
                        except (ValueError, IndexError) as e:
                            logger.warning(f"第{line_num}行处理实体时出错: {entity}, 错误: {e}")
                            continue
                    
                    # 写入BIO格式
                    for token, label in zip(tokens, labels):
                        f_out.write(f'{token} {label}\n')
                    f_out.write('\n')  # 句子间空行分隔
                    
                    line_count += 1
                    
                except json.JSONDecodeError as e:
                    logger.error(f"第{line_num}行JSON解析错误: {e}")
                    continue
                except Exception as e:
                    logger.error(f"第{line_num}行处理时发生未知错误: {e}")
                    continue
            
            logger.info(f"转换完成，共处理{line_count}行数据，输出到: {output_file}")
            
    except IOError as e:
        raise IOError(f"文件操作错误: {e}")
    except Exception as e:
        raise Exception(f"转换过程中发生错误: {e}")