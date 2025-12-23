# -*- coding: utf-8 -*-                          # 指定文件编码为 UTF-8，确保中文注释与字符串不会乱码
"""
ex4_joint_qat_qlora_train_fp16.py
---------------------------------
联合训练：QAT（激活伪量化） + LoRA（FP16 可训练适配器）
适配 AMD ROCm / CUDA，不依赖 bitsandbytes。

✅ 功能要点
- 模型主干：FP16，全权重冻结；
- 适配器：LoRA 注入，仅训练 LoRA 参数；
- QAT：对子层输入做激活伪量化 (FakeQuant + STE)，模拟低比特推理。

依赖：
    pip install torch transformers peft accelerate datasets
"""

import os                                          # 文件与路径操作
import torch                                       # PyTorch 主库
import torch.nn as nn                              # 神经网络组件
import torch.nn.functional as F                    # 常用函数库 (如激活、损失)
import random, argparse, json                      # 随机数、命令行参数解析、JSON 保存
from typing import List, Tuple                     # 类型注解工具
from datasets import Dataset,load_dataset                       # Hugging Face 数据集工具
from transformers import (                         # 导入 Transformers 模块
    AutoTokenizer,                                 # 自动加载分词器
    AutoModelForCausalLM,                          # 自动加载因果语言模型
    Trainer,                                       # 训练器封装类
    TrainingArguments,                             # 训练配置参数
    DataCollatorForLanguageModeling,               # 数据整理器（自动填充、对齐）
)
from peft import LoraConfig, get_peft_model        # 导入 PEFT 库的 LoRA 模块
from peft.tuners.lora.layer import Linear as LoRALinear  # 访问 LoRA 的线性层实现

# ======================================================
# 1  实用函数
# ======================================================
def set_seed(seed=42):                             # 固定随机种子，保证可复现性
    random.seed(seed)                              # Python 随机数种子
    torch.manual_seed(seed)                        # PyTorch CPU 随机种子
    torch.cuda.manual_seed_all(seed)               # GPU 随机种子（多卡情况）

def print_trainable_parameters(model):             # 打印模型可训练参数比例
    trainable, total = 0, 0
    for p in model.parameters():                   # 遍历所有参数
        total += p.numel()                         # 累加参数总数
        if p.requires_grad:                        # 判断是否可训练
            trainable += p.numel()
    print(f"🧮 可训练参数: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# ======================================================
# 2 FakeQuant + STE (仅激活)
# ======================================================
class FakeQuantSTE(torch.autograd.Function):       # 定义自定义伪量化函数（支持反传）
    @staticmethod
    def forward(ctx, x, scale):                    # 前向传播：模拟量化 + 反量化
        scale = torch.clamp(scale, min=1e-8)       # 防止 scale 过小
        q = torch.clamp(torch.round(x / scale), -127, 127)  # 量化到 INT8 区间
        return q * scale                           # 反量化回 FP16 空间
    @staticmethod
    def backward(ctx, grad_output):                # 反向传播：STE（直通估计）
        return grad_output, None                   # 忽略 scale 梯度，只保留 x 的梯度

def calc_scale_tensorwise(x, eps=1e-8):            # 计算张量级 scale 值
    max_abs = x.detach().abs().amax()              # 获取绝对值最大值
    return torch.clamp(max_abs / 127.0, min=eps)   # 映射到 INT8 动态范围

class QActWrapper(nn.Module):                      # 激活伪量化包装器
    """对输入激活进行INT8伪量化（Tensor级），保持权重不动"""
    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.sub = submodule                       # 保存原子层
    def forward(self, x, *args, **kwargs):          # 前向时先量化输入
        scale = calc_scale_tensorwise(x)            # 计算量化比例
        x_q = FakeQuantSTE.apply(x, scale)          # 执行量化仿真
        return self.sub(x_q, *args, **kwargs)       # 调用原子层继续前传

# ======================================================
# 3 QAT 选择逻辑
# ======================================================
ATTN_KEYS = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 注意力层关键词
FFN_KEYS  = ["up_proj", "gate_proj", "down_proj", "w1", "w2", "w3"]  # 前馈层关键词

def should_wrap(name: str, qat_targets: List[str]) -> bool:  # 判断是否应量化该层
    lname = name.lower()
    want_attn = "attn" in qat_targets
    want_ffn  = "ffn" in qat_targets
    if want_attn and any(k in lname for k in ATTN_KEYS): return True
    if want_ffn and any(k in lname for k in FFN_KEYS): return True
    return False

def get_parent_by_name(model, name) -> Tuple[nn.Module, str]:  # 获取模块父级对象
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        if not hasattr(parent, p): return None, None
        parent = getattr(parent, p)
    return parent, parts[-1]

def wrap_for_qat_after_lora(model, qat_targets):   # 包裹指定层的激活伪量化逻辑
    """仅包裹 LoRA base_layer + 普通 Linear"""
    replaced = []                                  # 记录被替换的层
    for name, module in list(model.named_modules()):  # 遍历所有子层
        if isinstance(module, LoRALinear) and should_wrap(name, qat_targets):  # 针对 LoRA 子层
            if hasattr(module, "base_layer"):
                module.base_layer = QActWrapper(module.base_layer)  # 包裹激活量化
                replaced.append(f"{name}.base_layer")
        elif "lora" in name.lower():               # 跳过 LoRA 自身定义层
            continue
        elif should_wrap(name, qat_targets):       # 对普通 Linear 层
            parent, key = get_parent_by_name(model, name)
            if parent and isinstance(getattr(parent, key), nn.Module):
                setattr(parent, key, QActWrapper(getattr(parent, key)))  # 替换为量化包装层
                replaced.append(name)
    print(f"🔧 已包裹 QAT 激活伪量化子层: {len(replaced)} 层")
    for n in replaced[:20]:                        # 打印前 20 层包裹信息
        print("  •", n)
    if len(replaced) > 20: print("  • ... (省略)")

# ======================================================
# 4 构建数据集
# ======================================================        

def preprocess_senti(example, tokenizer):
    # 将 label (0 或 1) 转换为文本
    label_text = "positive" if example["label"] == 1 else "negative"

    # 构建提示词 (prompt) 和 响应 (response)
    prompt = f"分析以下评论的情感：\n评论：{example['text']}\n情感："
    response = f"{label_text}{tokenizer.eos_token}"  # 加上结束符

    # Causal LM 训练：将 prompt 和 response 拼接在一起
    full_text = prompt + response

    # 1. 编码完整文本
    tokenized = tokenizer(full_text, truncation=True, max_length=512, padding="max_length")

    # 2. 编码提示词 (用于计算标签)
    prompt_tokenized = tokenizer(prompt, truncation=True, max_length=512, padding="max_length")
    prompt_len = sum(prompt_tokenized.attention_mask)

    # 3. 创建标签
    # 目标：让模型只学习预测 response 部分
    # 方法：将 prompt 部分的 token 对应的 labels 设为 -100 (忽略)
    labels = tokenized["input_ids"][:]
    labels[:prompt_len] = [-100] * prompt_len

    tokenized["labels"] = labels
    return tokenized


# ======================================================
# 5 主流程
# ======================================================
def main():
    parser = argparse.ArgumentParser()             # 创建命令行解析器
    parser.add_argument("--model_name_or_path", type=str,
                        default=r"Qwen/Qwen1.5-1.8B-Chat",
                        help="原始HF模型路径") # 模型路径或名称
    parser.add_argument("--output_dir", default="./joint-fp16-out")  # 输出目录
    parser.add_argument("--epochs", type=int, default=1)        # 训练轮数
    parser.add_argument("--batch_size", type=int, default=1)    # 批大小
    parser.add_argument("--grad_accum", type=int, default=4)    # 梯度累计步数
    parser.add_argument("--lr", type=float, default=2e-4)       # 学习率
    parser.add_argument("--max_length", type=int, default=512)  # 最大序列长度
    parser.add_argument("--seed", type=int, default=42)         # 随机种子
    parser.add_argument("--qat_targets", type=str, default="attn,ffn")  # 指定量化目标
    args = parser.parse_args()                    # 解析参数

    set_seed(args.seed)                           # 固定随机种子
    os.makedirs(args.output_dir, exist_ok=True)   # 创建输出目录

    print(f"🔹 加载 FP16 模型：{args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained( # 加载基础语言模型
        args.model_name_or_path,
        torch_dtype=torch.float16,                # 使用半精度权重
        device_map="auto",                        # 自动分配设备（支持多 GPU）
        trust_remote_code=True,                   # 允许自定义模型代码
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)  # 加载分词器
    if tokenizer.pad_token is None:               # 若无 pad_token 则补齐
        tokenizer.pad_token = tokenizer.eos_token

    # 注入 LoRA 模块
    print(">>>注入 LoRA 适配器 ...")
    lora_cfg = LoraConfig(                        # 定义 LoRA 参数配置
        r=8, lora_alpha=16, lora_dropout=0.05,    # 低秩分解、缩放与dropout
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],      # 在所有线性层注入 LoRA
        bias="none", task_type="CAUSAL_LM"        # 语言建模任务
    )
    model = get_peft_model(model, lora_cfg)       # 将 LoRA 模块注入模型
    print_trainable_parameters(model)             # 打印可训练参数占比

    # 包裹激活伪量化模块
    qat_targets = [s.strip().lower() for s in args.qat_targets.split(",")]
    wrap_for_qat_after_lora(model, qat_targets)

    # 数据准备
    dataset_name = "lansinuote/ChnSentiCorp"                            # 数据集名称
    print(f">>>正在加载数据集: {dataset_name}...")                  
    dataset = load_dataset(dataset_name)                                # 构建数据集
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42) # 切分测试数据集

    tokenized_dataset = dataset.map(lambda x: preprocess_senti(x, tokenizer))

    collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
    )
    # 训练配置
    train_args = TrainingArguments(
        output_dir=args.output_dir,               # 输出目录
        num_train_epochs=args.epochs,             # 训练轮数
        per_device_train_batch_size=args.batch_size,  # 每卡批次大小
        gradient_accumulation_steps=args.grad_accum,  # 梯度累积步数
        learning_rate=args.lr,                    # 学习率
        fp16=True,                                # 启用半精度训练
        logging_steps=5,                          # 日志间隔
        save_steps=100,                           # 模型保存步数
        evaluation_strategy="no",                 # 不启用评估
        save_total_limit=1,                       # 最多保存1份权重
        report_to="none",                         # 不上传至wandb
        dataloader_num_workers=0,                 # 单线程
    )

    trainer = Trainer(                            # 构建 Hugging Face Trainer
        model=model,
        args=train_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    print(">>>启动 FP16 + LoRA + QAT 联合训练 ...")
    trainer.train()                               # 开始训练
    print(">>>训练完成，保存模型中 ...")
    model.save_pretrained(args.output_dir)        # 保存模型权重
    tokenizer.save_pretrained(args.output_dir)    # 保存分词器

    # 简单推理测试
    prompts = [
        "分析以下评论的情感：\n评论:我真的是受够了！\n情感：",
        "分析以下评论的情感：\n评论:这里的氛围真的不错。\n情感：",
        "分析以下评论的情感：\n评论:我觉得虽然没什么意思，但整体还可以吧\n情感："
    ]
    model.eval()                                  # 设置为推理模式
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)  # 编码输入
        with torch.no_grad():                     # 关闭梯度计算
            out = model.generate(                 # 生成文本
                **inputs, max_new_tokens=64, do_sample=True,
                temperature=0.7, top_p=0.9,
                repetition_penalty = 1.2  # 惩罚重复词 (1.0表示无惩罚，1.2表示轻微惩罚)
            )
        print("\n Prompt:", p)                   # 打印输入
        print(" Output:", tokenizer.decode(out[0], skip_special_tokens=True))  # 输出结果

if __name__ == "__main__":                        # 程序入口
    main()                                        # 调用主函数执行
