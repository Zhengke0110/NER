# NER 命名实体识别项目

基于 BERT 的中文命名实体识别（NER）模型训练和验证系统。

## 项目结构

```
NER/
├── config.json                 # 配置文件
├── TrainNer.py                # 模型训练主文件
├── ValidationNer.py           # 模型验证主文件
├── DataConvert.py             # 数据格式转换工具
├── data/                      # 数据目录
│   ├── train_BIO.txt         # 训练数据（BIO格式）
│   ├── test.txt              # 测试数据
│   └── val.txt               # 验证数据
├── label/                     # 原始标注数据
│   └── admin.jsonl           # JSONL格式标注数据
├── utils/                     # 工具模块
│   ├── config.py             # 配置管理
│   ├── convert.py            # 数据转换工具
│   ├── dataUtils.py          # 数据处理工具
│   ├── modelUtils.py         # 模型相关工具
│   ├── trainingUtils.py      # 训练相关工具
│   └── validationUtils.py    # 验证相关工具
├── output/                    # 模型输出目录
├── checkpoint/               # 检查点目录
├── logs/                     # 日志目录
└── validation_output/        # 验证结果输出目录
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- Transformers
- Datasets
- NumPy
- 建议使用 GPU 进行训练

## 安装依赖

```bash
pip install torch transformers datasets numpy
```

## 配置说明

项目使用 `config.json` 统一管理配置，主要包含以下部分：

### 数据配置 (data)

- `data_dir`: 数据目录路径
- `train_file`: 训练文件名
- `val_file`: 验证文件名
- `test_file`: 测试文件名
- `max_length`: 最大序列长度
- `encoding`: 文件编码

### 模型配置 (model)

- `model_name`: 预训练模型名称（如：bert-base-chinese）
- `num_labels`: 标签数量
- `dropout`: Dropout 率

### 训练配置 (training)

- `output_dir`: 模型输出目录
- `num_train_epochs`: 训练轮数
- `per_device_train_batch_size`: 训练批次大小
- `learning_rate`: 学习率
- `save_steps`: 保存步数间隔

### 验证配置 (validation)

- `model_path`: 模型路径
- `tokenizer_path`: 分词器路径
- `output_dir`: 验证结果输出目录

## 使用方法

### 1. 数据准备

#### 数据格式转换

如果您有 JSONL 格式的标注数据，可以使用数据转换工具：

```bash
python DataConvert.py
```

该工具会将 `label/admin.jsonl` 转换为 BIO 格式的训练数据 `data/train_BIO.txt`。

#### 数据格式说明

BIO 格式数据示例：

```
词 O
汇 O
阅 O
读 O
是 O
关 O
键 O
2 B-year
0 I-year
0 I-year
9 I-year
年 I-year
高 B-exam
考 I-exam
在 O
北 B-location
京 I-location
```

支持的实体类型：

- `B-exam / I-exam`: 考试类型
- `B-location / I-location`: 地点
- `B-year / I-year`: 年份
- `O`: 非实体

**重要提示**：确保所有数据文件（训练、验证、测试）使用相同的标签体系，避免标签不一致导致的训练错误。

### 2. 模型训练

```bash
python TrainNer.py
```

训练过程会：

1. 加载配置文件
2. 准备训练和验证数据集
3. 创建 BERT-based NER 模型
4. 开始训练并保存检查点
5. 输出训练日志和模型评估结果

训练完成后，模型将保存在 `output/` 目录中。

### 3. 模型验证

ValidationNer.py 支持四种不同的验证模式，通过 `--mode` 参数指定：

#### 模式 1：单句验证模式 (single)

对单个句子进行实体识别：

```bash
# 使用默认测试文本
python ValidationNer.py --mode single

# 指定自定义文本
python ValidationNer.py --mode single --text "2009年高考在北京的报名费是2009元"

# 指定输出文件
python ValidationNer.py --mode single --text "明年考研在清华大学举行" --output my_result.json
```

**输出示例：**

```
=== 预测结果 ===
1. 实体: 2009年
   类型: year
   置信度: 0.9856
   位置: 0-5
   Tokens: ['2009', '年']

2. 实体: 高考
   类型: exam
   置信度: 0.9721
   位置: 5-7
   Tokens: ['高', '考']

3. 实体: 北京
   类型: location
   置信度: 0.9634
   位置: 8-10
   Tokens: ['北', '京']
```

#### 模式 2：批量验证模式 (batch)

对预设的多个测试文本进行批量验证：

```bash
# 批量验证配置文件中的默认测试文本
python ValidationNer.py --mode batch

# 指定输出文件
python ValidationNer.py --mode batch --output batch_results.json
```

批量模式会处理配置文件中 `default_test_texts` 数组里的所有文本，包括：

- "2009 年高考在北京的报名费是 2009 元"
- "2020 年研究生考试在上海进行"
- "明年的公务员考试将在广州举办"
- 等 8 个预设测试样本

**输出示例：**

```
=== 批量预测结果 ===

1. 文本: 2009年高考在北京的报名费是2009元
   1. [year] 2009年 (置信度: 0.9856)
   2. [exam] 高考 (置信度: 0.9721)
   3. [location] 北京 (置信度: 0.9634)

2. 文本: 2020年研究生考试在上海进行
   1. [year] 2020年 (置信度: 0.9801)
   2. [exam] 研究生考试 (置信度: 0.9678)
   3. [location] 上海 (置信度: 0.9543)
```

#### 模式 3：交互式验证模式 (interactive)

提供持续的交互式命令行界面，可以连续输入文本进行实时验证：

```bash
python ValidationNer.py --mode interactive
```

**使用示例：**

```
=== 交互式NER验证 ===
输入文本进行实体识别，输入 'quit' 退出

请输入文本: 2021年计算机二级考试在深圳举行

发现 3 个实体:
1. [year] 2021年 (置信度: 0.9823)
2. [exam] 计算机二级考试 (置信度: 0.9567)
3. [location] 深圳 (置信度: 0.9445)

请输入文本: quit
退出验证
```

交互式模式的优点：

- 无需重复启动程序
- 适合快速测试多个不同文本
- 实时查看识别结果
- 支持 `quit`、`exit`、`q` 命令退出

#### 模式 4：测试数据集验证模式 (test)

对完整的测试数据集文件进行验证和评估：

```bash
# 使用配置文件中的测试文件
python ValidationNer.py --mode test

# 指定自定义测试文件
python ValidationNer.py --mode test --test-file data/custom_test.txt

# 指定输出路径
python ValidationNer.py --mode test --test-file data/val.txt --output test_results.json
```

测试模式功能：

- 读取 BIO 格式的测试文件
- 对所有测试样本进行批量预测
- 生成详细的验证报告
- 计算整体性能指标（如果有标准答案）

#### 验证结果文件

验证结果将保存在 `validation_output/` 目录中：

- **single 模式**: `single_validation_results.json` - 单句验证结果
- **batch 模式**: `batch_validation_results.json` - 批量验证结果
- **test 模式**: `test_validation_results.json` - 测试集验证结果
- **interactive 模式**: 不自动保存文件，但可在交互中查看结果

#### 通用参数说明

- `--mode`: 验证模式，可选值：`single`、`batch`、`interactive`、`test`
- `--text`: 单句模式下的输入文本
- `--test-file`: 测试模式下的数据文件路径
- `--output`: 自定义输出文件路径（默认使用配置文件中的路径）

## 常见问题

### 1. 标签不一致错误

```
ERROR: 发现未知标签: 'B-place'
```

**解决方案**：确保所有数据文件使用相同的标签体系。检查训练数据和验证数据的标签是否一致。

### 2. CUDA 内存不足

**解决方案**：

- 减小 `per_device_train_batch_size`
- 减小 `max_length`
- 使用梯度累积

### 3. 模型加载失败

**解决方案**：

- 检查 `model_path` 是否正确
- 确认模型文件完整性
- 验证配置文件中的路径设置

## 监控和日志

- 训练日志：查看 `logs/` 目录
- 验证日志：查看 `validation.log`
- 模型检查点：查看 `output/` 目录
- 验证结果：查看 `validation_output/` 目录

## 自定义配置

您可以通过修改 `config.json` 来调整：

- 模型超参数
- 数据路径
- 训练策略
- 验证设置
