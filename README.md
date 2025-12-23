# LoRA 模型微调与量化感知训练

## 项目概述

本项目实现了基于 LoRA (Low-Rank Adaptation) 的大语言模型微调技术，并结合了 QAT (Quantization-Aware Training) 进行量化感知训练。该方案允许在资源受限的环境下高效微调大模型，同时支持后续的低比特推理部署。

## 核心功能

### ✅ 模型训练与微调
- **LoRA 适配器注入**：仅训练低秩分解参数，大幅减少可训练参数量
- **FP16 半精度训练**：支持混合精度训练，降低内存占用
- **自动设备分配**：支持多 GPU 训练，自动分配模型权重
- **随机种子固定**：保证训练过程的可复现性

### ✅ 量化感知训练 (QAT)
- **激活伪量化**：对子层输入进行 INT8 伪量化模拟
- **STE 直通估计**：使用 Straight-Through Estimator 进行梯度回传
- **灵活的量化目标**：支持对注意力层、前馈层等不同组件进行量化
- **动态量化比例**：基于张量最大值动态计算量化比例

### ✅ 完整的训练流程
- **数据集处理**：支持 Hugging Face 数据集加载与预处理
- **Causal LM 训练**：支持因果语言模型的生成式微调
- **训练配置灵活**：支持命令行参数调整，方便实验
- **推理测试**：训练完成后自动进行推理测试

## 技术栈

| 组件 | 版本/说明 |
|------|----------|
| PyTorch | 深度学习框架 |
| Transformers | Hugging Face 模型库 |
| PEFT | 参数高效微调库 |
| Datasets | Hugging Face 数据集库 |
| Accelerate | 加速训练库 |

## 安装与环境配置

###  安装依赖

使用 pip 安装：

```bash
pip install torch transformers peft accelerate datasets
```

或使用 conda 环境：

```bash
conda env create -f environment.yml
conda activate lora-env
```

## 快速开始

### 基本使用示例

```bash
python lora.py --model_name_or_path <model-path> --output_dir <output-path> --epochs 3 --batch_size 2
```

### 命令行参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name_or_path` | `./model/Qwen1.5-0.5B` | 原始模型路径或 Hugging Face 模型名 |
| `--output_dir` | `./joint-fp16-out` | 训练结果输出目录 |
| `--epochs` | 1 | 训练轮数 |
| `--batch_size` | 1 | 每设备训练批大小 |
| `--grad_accum` | 4 | 梯度累积步数 |
| `--lr` | 2e-4 | 学习率 |
| `--max_length` | 512 | 最大序列长度 |
| `--seed` | 42 | 随机种子 |
| `--qat_targets` | `attn,ffn` | 量化目标层，可选值：`attn`, `ffn` |

## 实现原理

### 1. LoRA 适配器机制

LoRA 通过低秩分解减少可训练参数：
- 将全连接层权重分解为两个低秩矩阵
- 仅训练低秩矩阵参数，冻结原始模型权重
- 训练完成后可与原始模型合并，不影响推理速度

```python
lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
```

### 2. 量化感知训练

使用伪量化技术模拟低比特推理环境：
- 自定义 `FakeQuantSTE` 函数实现量化模拟
- 前向传播：将输入量化到 INT8 再反量化回 FP16
- 反向传播：使用 STE 忽略量化误差，直接传递梯度

```python
class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        # 量化：round(x / scale) → 反量化：q * scale
        scale = torch.clamp(scale, min=1e-8)
        q = torch.clamp(torch.round(x / scale), -127, 127)
        return q * scale
    
    @staticmethod
    def backward(ctx, grad_output):
        # STE：直接传递梯度
        return grad_output, None
```

### 3. 模型权重冻结与参数计算

- 仅训练 LoRA 注入的低秩参数
- 原始模型权重完全冻结，不参与训练
- 可训练参数占比通常小于 1%

## 项目结构

```
P07/
├── lora.py                    # 主训练脚本
├── ex1_load.py                # 模型加载示例
├── ex1_packge.py              # 模型打包示例
├── ex1_qat_train.py           # QAT 训练示例
├── ex2_qlora_finetune.py      # QLoRA 微调示例
├── ex3_qkd_distillation_fp16.py  # 知识蒸馏示例
├── ex4_joint_qat_qlora_train_fp16.py  # 联合训练示例
├── environment.yml            # Conda 环境配置
├── joint-fp16-out/            # 训练输出目录
└── README.md                  # 项目说明文档
```

## 训练流程

1. **模型加载**：使用 FP16 精度加载预训练模型
2. **LoRA 注入**：配置并注入 LoRA 适配器
3. **QAT 包装**：对指定层进行激活量化包装
4. **数据准备**：加载、预处理并分词数据集
5. **训练配置**：设置训练参数与优化器
6. **模型训练**：执行微调训练
7. **推理测试**：使用测试样本验证模型效果
8. **模型保存**：保存训练后的 LoRA 权重与配置

## 性能指标

### 参数量对比

| 模型规模 | 全量微调参数量 | LoRA 微调参数量 | 参数量减少比例 |
|----------|----------------|-----------------|----------------|
| 0.5B     | 500M           | 约 1M           | 99.8%          |
| 1.8B     | 1.8B           | 约 3.6M         | 99.8%          |
| 7B       | 7B             | 约 14M          | 99.8%          |

### 内存占用

- FP16 模型加载：约 1GB per 1B 参数
- LoRA 微调：额外增加约 100MB-500MB 内存
- QAT 训练：内存占用与 FP16 训练相当

## 推理与部署

### 模型合并

训练完成后，可以将 LoRA 权重与原始模型合并，生成完整模型：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 加载原始模型
base_model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载 LoRA 模型
peft_model = PeftModel.from_pretrained(base_model, "lora_output_path")

# 合并模型
merged_model = peft_model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("merged_output_path")
```

### 量化部署

经过 QAT 训练的模型可以直接导出为 INT8 模型进行部署：
- 支持 ONNX Runtime 或 TensorRT 进行低比特推理
- 量化后的模型大小约为 FP16 模型的 1/2
- 推理速度可提升 2-4 倍（取决于硬件支持）

## 应用场景

1. **资源受限环境下的模型微调**
2. **大模型的领域适配**
3. **低比特推理部署准备**
4. **模型压缩与加速**
5. **多任务学习与迁移学习**

## 实验示例

### 情感分析任务微调

使用 `lansinuote/ChnSentiCorp` 中文情感分析数据集进行微调：

```bash
python lora.py --model_name_or_path Qwen1.5-0.5B --output_dir sentiment-lora --epochs 3 --batch_size 2
```

### 不同量化目标对比

```bash
# 仅量化注意力层
python lora.py --qat_targets attn --output_dir lora-attn-qat

# 仅量化前馈层  
python lora.py --qat_targets ffn --output_dir lora-ffn-qat

# 量化所有层
python lora.py --qat_targets attn,ffn --output_dir lora-all-qat
```

## 注意事项

1. **模型选择**：确保使用支持因果语言建模的模型架构
2. **内存管理**：对于较大模型，建议使用梯度累积或模型并行
3. **学习率调整**：LoRA 微调通常需要比全量微调整更高的学习率
4. **量化精度**：QAT 训练仅支持激活量化，权重量化需在部署阶段进行
5. **数据集大小**：建议使用足够大的数据集，避免过拟合

## 未来改进方向

- [ ] 支持权重量化感知训练
- [ ] 实现 GPTQ 等量化技术集成
- [ ] 支持 LoRA 与其他 PEFT 方法结合
- [ ] 提供更丰富的模型评估指标
- [ ] 支持多模态模型微调

## 参考资料

1. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. [PEFT: Parameter-Efficient Fine-Tuning of Billion-Scale Models on Low-Resource Hardware](https://arxiv.org/abs/2106.09685)
3. [Quantization-Aware Training for Deep Neural Networks](https://arxiv.org/abs/1712.05877)
4. [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/index)
5. [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)

## 许可证

本项目采用 MIT 许可证，详情请查看 LICENSE 文件。

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

**版本**: 1.0.0  
**更新日期**: 2025-12-09  
**作者**: LoRA 项目团队