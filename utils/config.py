"""
配置管理模块
包含训练、模型、数据相关的配置参数
"""

import os
import json
import logging
from dataclasses import dataclass, fields
from typing import Optional, Dict, Any, Union, List
from pathlib import Path

# 配置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    # 只有当logger还没有handlers时才添加，避免重复添加
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class DataConfig:
    """数据相关配置"""

    data_dir: str = "D:\\NLP_Project\\NER\\data"
    train_file: str = "train.txt"
    val_file: str = "test.txt"  # 注意：原代码中val用的是test.txt
    test_file: str = "val.txt"  # 注意：原代码中test用的是val.txt
    max_length: int = 512
    encoding: str = "UTF-8"

    @classmethod
    def from_json(
        cls, json_path: Union[str, Path], section: str = "data"
    ) -> "DataConfig":
        """从JSON文件加载配置

        Args:
            json_path: JSON配置文件路径
            section: 配置文件中的节名，默认为"data"

        Returns:
            DataConfig实例

        Raises:
            FileNotFoundError: 配置文件不存在
            KeyError: 配置文件中缺少指定节
            ValueError: 配置格式错误
        """
        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {json_path}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {e}")

        if section not in config_data:
            raise KeyError(f"配置文件中缺少 '{section}' 节")

        section_data = config_data[section]

        # 获取类的字段名
        field_names = {field.name for field in fields(cls)}

        # 过滤出有效的配置项
        valid_config = {k: v for k, v in section_data.items() if k in field_names}

        # 验证路径是否存在（对于data_dir）
        if "data_dir" in valid_config:
            data_dir = Path(valid_config["data_dir"])
            if not data_dir.exists():
                logger.warning(f"数据目录不存在: {data_dir}")

        return cls(**valid_config)

    @classmethod
    def from_json_with_fallback(
        cls, json_path: Union[str, Path], section: str = "data"
    ) -> "DataConfig":
        """从JSON文件加载配置，如果失败则使用默认配置

        Args:
            json_path: JSON配置文件路径
            section: 配置文件中的节名，默认为"data"

        Returns:
            DataConfig实例
        """
        try:
            return cls.from_json(json_path, section)
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.warning(f"加载配置失败，使用默认配置: {e}")
            return cls()

    def get_file_path(self, file_type: str) -> str:
        """获取文件完整路径"""
        file_mapping = {
            "train": self.train_file,
            "val": self.val_file,
            "test": self.test_file,
        }

        if file_type not in file_mapping:
            raise ValueError(
                f"不支持的文件类型: {file_type}. 支持的类型: {list(file_mapping.keys())}"
            )

        file_path = os.path.join(self.data_dir, file_mapping[file_type])

        # 验证文件是否存在
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")

        return file_path

    def validate(self) -> bool:
        """验证配置是否有效

        Returns:
            bool: 配置是否有效
        """
        errors = []

        # 检查数据目录
        if not os.path.exists(self.data_dir):
            errors.append(f"数据目录不存在: {self.data_dir}")

        # 检查文件
        for file_type in ["train", "val", "test"]:
            file_path = os.path.join(self.data_dir, getattr(self, f"{file_type}_file"))
            if not os.path.exists(file_path):
                errors.append(f"{file_type}文件不存在: {file_path}")

        # 检查参数范围
        if self.max_length <= 0:
            errors.append(f"max_length必须大于0: {self.max_length}")

        if errors:
            logger.error("配置验证失败:")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "data_dir": self.data_dir,
            "train_file": self.train_file,
            "val_file": self.val_file,
            "test_file": self.test_file,
            "max_length": self.max_length,
            "encoding": self.encoding,
        }


@dataclass
class ModelConfig:
    """模型相关配置"""

    model_name: str = "bert-base-chinese"
    num_labels: int = 7
    dropout: float = 0.1

    @classmethod
    def from_json(
        cls, json_path: Union[str, Path], section: str = "model"
    ) -> "ModelConfig":
        """从JSON文件加载配置"""
        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {json_path}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {e}")

        if section not in config_data:
            raise KeyError(f"配置文件中缺少 '{section}' 节")

        section_data = config_data[section]
        field_names = {field.name for field in fields(cls)}
        valid_config = {k: v for k, v in section_data.items() if k in field_names}

        return cls(**valid_config)

    @classmethod
    def from_json_with_fallback(
        cls, json_path: Union[str, Path], section: str = "model"
    ) -> "ModelConfig":
        """从JSON文件加载配置，如果失败则使用默认配置"""
        try:
            return cls.from_json(json_path, section)
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.warning(f"加载模型配置失败，使用默认配置: {e}")
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于模型初始化"""
        return {"num_labels": self.num_labels, "dropout": self.dropout}


@dataclass
class TrainingConfig:
    """训练相关配置"""

    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoint"
    logging_dir: str = "./logs"

    num_train_epochs: int = 1000
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8

    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01

    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 1
    eval_strategy: str = "steps"
    eval_steps: int = 1000

    @classmethod
    def from_json(
        cls, json_path: Union[str, Path], section: str = "training"
    ) -> "TrainingConfig":
        """从JSON文件加载配置"""
        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {json_path}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {e}")

        if section not in config_data:
            raise KeyError(f"配置文件中缺少 '{section}' 节")

        section_data = config_data[section]
        field_names = {field.name for field in fields(cls)}
        valid_config = {k: v for k, v in section_data.items() if k in field_names}

        return cls(**valid_config)

    @classmethod
    def from_json_with_fallback(
        cls, json_path: Union[str, Path], section: str = "training"
    ) -> "TrainingConfig":
        """从JSON文件加载配置，如果失败则使用默认配置"""
        try:
            return cls.from_json(json_path, section)
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.warning(f"加载训练配置失败，使用默认配置: {e}")
            return cls()

    def get_checkpoint_path(self, model_name: str, epochs: int) -> str:
        """获取模型保存路径"""
        os.makedirs(os.path.join(self.checkpoint_dir, "model"), exist_ok=True)
        return os.path.join(self.checkpoint_dir, "model", f"{model_name}-{epochs}epoch")


@dataclass
class ValidationConfig:
    """验证配置类"""

    # 模型相关配置
    model_path: str = "./output/checkpoint-1000"
    tokenizer_path: str = "bert-base-chinese"

    # 验证集文件路径
    val_file: Optional[str] = "./data/val.txt"

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
    default_test_texts: Optional[List[str]] = None

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
                "期中考试安排在教学楼进行",
            ]

        # 确保输出目录存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_json(
        cls, json_path: Union[str, Path], section: str = "validation"
    ) -> "ValidationConfig":
        """从JSON文件加载配置"""
        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {json_path}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {e}")

        if section not in config_data:
            raise KeyError(f"配置文件中缺少 '{section}' 节")

        section_data = config_data[section]
        field_names = {field.name for field in fields(cls)}
        valid_config = {k: v for k, v in section_data.items() if k in field_names}

        return cls(**valid_config)

    @classmethod
    def from_json_with_fallback(
        cls, json_path: Union[str, Path], section: str = "validation"
    ) -> "ValidationConfig":
        """从JSON文件加载配置，如果失败则使用默认配置"""
        try:
            return cls.from_json(json_path, section)
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.warning(f"加载验证配置失败，使用默认配置: {e}")
            return cls()

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
                logger.error(f"配置错误: {error}")
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "val_file": self.val_file,
            "log_file": self.log_file,
            "log_level": self.log_level,
            "output_dir": self.output_dir,
            "single_output_file": self.single_output_file,
            "batch_output_file": self.batch_output_file,
            "max_length": self.max_length,
            "device": self.device,
            "default_test_texts": self.default_test_texts,
        }

    def get_val_file_path(self) -> Optional[str]:
        """
        获取验证集文件的完整路径

        Returns:
            验证集文件路径，如果不存在则返回None
        """
        if self.val_file:
            # 如果是相对路径，转换为绝对路径
            if not os.path.isabs(self.val_file):
                val_path = os.path.join(os.getcwd(), self.val_file)
            else:
                val_path = self.val_file

            # 检查文件是否存在
            if os.path.exists(val_path):
                return val_path
            else:
                logger.warning(f"验证集文件不存在: {val_path}")
                return None
        return None


# 默认配置实例
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_VALIDATION_CONFIG = ValidationConfig()
