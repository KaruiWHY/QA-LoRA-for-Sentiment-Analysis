# LoRA 模型微调与量化感知训练

## 项目概述


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


## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

**版本**: 1.0.0  
**更新日期**: 2025-12-09  
