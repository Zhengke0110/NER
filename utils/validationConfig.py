"""
模型验证配置
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class ValidationConfig:
    """验证配置类"""
    
    # 模型相关配置
    model_path: str = "./output/checkpoint-1000"
    tokenizer_path: str = "bert-base-chinese"
    
    # 日志配置
    log_file: str = "validation.log"
    log_level: str = "INFO"
    
    # 输出配置
    output_dir: str = "./validation_output"
    single_output_file: str = "single_validation_results.json"
    batch_output_file: str = "batch_validation_results.json"
    
    # 推理配置
    max_length: int = 512
    device: str = "cpu"  # 或 "cuda" 如果有GPU
    
    # 测试样本
    default_test_texts: list = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.default_test_texts is None:
            self.default_test_texts = [
                "2009年高考在北京的报名费是2009元",
                "2020年研究生考试在上海进行", 
                "明年的公务员考试将在广州举办",
                "2018年高考报名费用是100元",
                "去年的期末考试在深圳大学举行",
                "今年中考将在杭州市举行",
                "2022年考研报名时间是10月份",
                "期中考试安排在教学楼进行"
            ]
        
        # 确保输出目录存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def get_output_path(self, filename: str) -> str:
        """获取输出文件的完整路径"""
        return os.path.join(self.output_dir, filename)
    
    def validate_paths(self) -> bool:
        """验证路径是否有效"""
        errors = []
        
        # 检查模型路径
        if not os.path.exists(self.model_path):
            errors.append(f"模型路径不存在: {self.model_path}")
        
        # 检查输出目录
        output_path = Path(self.output_dir)
        if not output_path.exists():
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"无法创建输出目录: {e}")
        
        if errors:
            for error in errors:
                print(f"配置错误: {error}")
            return False
        
        return True
