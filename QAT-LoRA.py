# -*- coding: utf-8 -*-
"""
lora_w4a8_qalora.py
----------------------
W4A8 全量化感知训练 (Full QAT) + QA-LoRA
- Stage 1: FP16 LoRA Warmup
- Stage 2: W4A8 QA-LoRA (Weight INT4 + Activation INT8 + Group-wise Adaptation)

适配 Qwen-1.8B 等模型，支持 AMD ROCm / CUDA。
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random, argparse, json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from sklearn.metrics import precision_recall_fscore_support
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)

from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora.layer import Linear as LoRALinear
import gc


# ======================================================
# 1. 核心量化组件 (W4A8 + STE)
# ======================================================

class FakeQuantSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for INT8 Activation Quantization
    """

    @staticmethod
    def forward(ctx, x, scale):
        scale = torch.clamp(scale, min=1e-8)
        # 量化到 INT8 (-127, 127)
        q = torch.clamp(torch.round(x / scale), -127, 127)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output):
        # 直通估计：梯度直接穿透，忽略 Round 操作的不可导性
        return grad_output, None


def calc_scale_tensorwise(x, eps=1e-8):
    """计算激活值的动态量化 Scale (Tensor-wise)"""
    max_abs = x.detach().abs().amax()
    return torch.clamp(max_abs / 127.0, min=eps)

def calc_scale_tokenwise(x, eps=1e-8):
    """
    [改进] 计算激活值的动态量化 Scale (Token-wise / Per-Row)
    
    Args:
        x: 输入张量，形状通常为 [Batch, Seq, Dim]
    Returns:
        scale: 形状为 [Batch, Seq, 1] 的张量，保持维度以便广播
    """
    # dim=-1 表示在最后一个维度（特征维度）上找最大值
    # keepdim=True 保持维度，结果形状从 [B, S, D] 变为 [B, S, 1]
    max_abs = x.detach().abs().amax(dim=-1, keepdim=True)
    
    return torch.clamp(max_abs / 127.0, min=eps)


class QALoRALayer(nn.Module):
    def __init__(self, base_layer: nn.Module, r: int = 8, lora_alpha: int = 16, group_size: int = 32):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.group_size = group_size

        if self.in_features % group_size != 0:
            raise ValueError(f"in_features ({self.in_features}) must be divisible by group_size ({group_size})")

        self.weight = base_layer.weight
        self.weight.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.reduced_dim = self.in_features // group_size

        # LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros(r, self.reduced_dim))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)
        
        # [新增] 用于存储量化参数，以便合并时使用 (训练时动态计算，推理/合并时固定)
        self.register_buffer('quant_scale', None)
        self.register_buffer('quant_zero', None)

    def fake_quant_activation(self, x):
        """修改为 Token-wise 量化"""
        scale = calc_scale_tokenwise(x)
        return FakeQuantSTE.apply(x, scale)

    def fake_quant_weight_asym(self, w):
        """
        [修改重点] 论文公式 (1): Group-wise Asymmetric Min-Max Quantization
        w shape: [Out, In]
        """
        out_dim, in_dim = w.shape
        
        # 1. Reshape to [Out, Num_Groups, Group_Size]
        w_reshaped = w.reshape(out_dim, in_dim // self.group_size, self.group_size)
        
        # 2. 计算 Min/Max (Group-wise)
        # shape: [Out, Num_Groups, 1]
        max_val = w_reshaped.amax(dim=-1, keepdim=True)
        min_val = w_reshaped.amin(dim=-1, keepdim=True)
        
        # 3. 计算 Alpha (Scale) 和 Beta (Zero Point)
        # INT4 range: 0 to 15 (2^4 - 1)
        # 避免除以0，加上 eps
        alpha = (max_val - min_val) / 15.0
        alpha = torch.clamp(alpha, min=1e-5)
        beta = min_val
        
        # 4. Quantize (公式 1)
        # W_int = Round((W - Beta) / Alpha)
        w_int = ((w_reshaped - beta) / alpha).round().clamp(0, 15)
        
        # 5. Dequantize (用于前向传播)
        # W_recon = W_int * Alpha + Beta
        w_recon = w_int * alpha + beta
        
        # Reshape 回原始形状
        w_recon = w_recon.reshape(out_dim, in_dim)
        
        # [可选] 保存当前的统计数据以便后续分析或合并
        if self.training:
            self.quant_scale = alpha.detach() # [Out, Groups, 1]
            self.quant_zero = beta.detach()   # [Out, Groups, 1]
            
        return w_recon, alpha, beta

    def forward(self, x):
        # 1. Activation Quantization (A8)
        x_q = self.fake_quant_activation(x)

        # 2. Weight Quantization (W4 Asymmetric) [修改点]
        w_q, _, _ = self.fake_quant_weight_asym(self.weight)

        # 3. Base Computation
        base_out = F.linear(x_q, w_q)

        # 4. QA-LoRA Path (保持不变)
        b, s, d = x_q.shape
        x_reshaped = x_q.reshape(b, s, d // self.group_size, self.group_size)
        x_pooled = x_reshaped.sum(dim=-1) # Group-wise Sum

        lora_input = x_pooled.to(self.lora_A.dtype)
        lora_out = (lora_input @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return base_out + lora_out.to(base_out.dtype)

    def merge(self):
        """
        [新增功能] 实现论文 Appendix B.3 的无损合并逻辑
        无需重新量化权重，只需更新 Zero Point (Beta)。
        """
        with torch.no_grad():
            out_dim, in_dim = self.weight.shape
            
            # 1. 重新获取当前的量化参数 (Alpha, Beta)
            w_reshaped = self.weight.reshape(out_dim, in_dim // self.group_size, self.group_size)
            max_val = w_reshaped.amax(dim=-1, keepdim=True)
            min_val = w_reshaped.amin(dim=-1, keepdim=True)
            alpha = (max_val - min_val) / 15.0
            alpha = torch.clamp(alpha, min=1e-5)
            beta = min_val # 原始 Zero Point
            
            # 2. 计算 LoRA 的贡献
            # LoRA 实际上是在每个 Group 上加了一个 Bias
            # shape: [Out, r] @ [r, Num_Groups] -> [Out, Num_Groups]
            lora_delta = (self.lora_B @ self.lora_A) * self.scaling
            
            # 调整形状以匹配 Beta [Out, Num_Groups, 1]
            lora_delta = lora_delta.unsqueeze(-1)
            
            # 3. 合并到 Zero Point
            # 论文公式 (7) 的推导逻辑：新的 Zero Point = 旧 Zero Point + LoRA部分
            # 注意：这里我们做的是加法，因为 W_recon = Q*alpha + beta + LoRA
            # => W_recon = Q*alpha + (beta + LoRA)
            new_beta = beta + lora_delta
            
            # 4. 获取 INT4 权重
            w_int = ((w_reshaped - beta) / alpha).round().clamp(0, 15)
            
            return w_int, alpha, new_beta


def convert_to_qalora_w4a8(model, r=8, lora_alpha=16, group_size=32, target_modules=["q_proj", "v_proj"]):
    """
    将模型转换为 W4A8 QA-LoRA 架构
    """
    print(f"🔄 正在将模型转换为 W4A8 QA-LoRA (Group Size={group_size})...")

    def replace_module(module, current_name=""):
        for name, child in module.named_children():
            full_name = f"{current_name}.{name}" if current_name else name

            # 找到目标 Linear 层 (可能是 Peft 的 LoRALinear 或 普通 Linear)
            # 我们只替换在 target_modules 里的层
            if isinstance(child, (LoRALinear, nn.Linear)):
                # 判断名字匹配
                is_target = any(t in name for t in target_modules)  # 简单匹配 key

                # 如果是 Peft LoRALinear，我们需要提取 base_layer
                if isinstance(child, LoRALinear) and is_target:
                    print(f"  -> 替换层 (Peft): {full_name}")
                    base_layer = child.base_layer
                    new_layer = QALoRALayer(base_layer, r=r, lora_alpha=lora_alpha, group_size=group_size)
                    
                    # 先转到目标设备和 dtype (FP16)
                    new_layer = new_layer.to(base_layer.weight.device).to(base_layer.weight.dtype)
                    
                    # 关键修正：强制将可训练参数转回 FP32
                    new_layer.lora_A.data = new_layer.lora_A.data.float()
                    new_layer.lora_B.data = new_layer.lora_B.data.float()

                    setattr(module, name, new_layer)

                # 如果已经是普通 Linear (Stage 2 重新加载 raw model 时)
                elif isinstance(child, nn.Linear) and is_target:
                    print(f"  -> 替换层 (Linear): {full_name}")
                    new_layer = QALoRALayer(child, r=r, lora_alpha=lora_alpha, group_size=group_size)
                    
                    # 先转到目标设备和 dtype (FP16)
                    new_layer = new_layer.to(child.weight.device).to(child.weight.dtype)
                    
                    # 关键修正：强制将可训练参数转回 FP32
                    new_layer.lora_A.data = new_layer.lora_A.data.float()
                    new_layer.lora_B.data = new_layer.lora_B.data.float()
                    
                    setattr(module, name, new_layer)
            else:
                replace_module(child, full_name)

    replace_module(model)
    return model


def save_merged_model(model, output_dir):
    """
    [新增功能] 实现 QA-LoRA 无损合并并保存
    """
    print(">>> 正在进行 QA-LoRA 无损合并...")
    os.makedirs(output_dir, exist_ok=True)
    
    quantized_state = {}
    
    for name, module in model.named_modules():
        if isinstance(module, QALoRALayer):
            # 获取合并后的参数
            w_int, alpha, new_beta = module.merge()
            
            # 保存到字典 (模拟保存为量化模型格式，如 GPTQ/AWQ 格式)
            quantized_state[f"{name}.w_int"] = w_int.cpu()      # INT4 权重
            quantized_state[f"{name}.scale"] = alpha.cpu()      # FP16 Scale
            quantized_state[f"{name}.zero"] = new_beta.cpu()    # FP16 Zero Point (已融合 LoRA)
            
    torch.save(quantized_state, f"{output_dir}/merged_qalora_w4.pt")
    print(f"合并完成，模型已保存至 {output_dir}")


# ======================================================
# 2. 辅助工具
# ======================================================

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrainingMetricsCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.metrics = {'step': [], 'loss': [], 'learning_rate': []}

    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history and state.log_history[-1].get('loss') is not None:
            log = state.log_history[-1]
            self.metrics['step'].append(state.global_step)
            self.metrics['loss'].append(log['loss'])
            self.metrics['learning_rate'].append(log.get('learning_rate', 0.0))

            # [新增] 实时打印显存占用
            if torch.cuda.is_available():
                mem_used = torch.cuda.max_memory_allocated() / 1024**3
                # 打印到控制台，使用 \r 覆盖当前行，避免刷屏太快
                # 注意：Trainer 自身的进度条可能会覆盖这个输出，所以也可以选择每隔 N 步打印一次
                if state.global_step % 10 == 0:
                    print(f" [Step {state.global_step}] Loss: {log['loss']:.4f} | Max Mem: {mem_used:.2f} GB")
                    # 重置峰值，以便观察下一个区间的峰值
                    torch.cuda.reset_peak_memory_stats()

    def save_metrics(self, save_path):
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)


def plot_loss_curve(metrics_dict, save_path=None):
    plt.figure(figsize=(10, 5))
    for name, data in metrics_dict.items():
        if 'step' in data:
            plt.plot(data['step'], data['loss'], label=name)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    if save_path:
        plt.savefig(save_path)
    plt.close()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    
    # Only calculate accuracy where labels are not -100
    mask = labels != -100
    
    # Filter predictions and labels
    filtered_preds = predictions[mask]
    filtered_labels = labels[mask]
    
    correct = (filtered_preds == filtered_labels).sum()
    total = mask.sum()
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Calculate Precision, Recall, F1 (weighted average for multi-class/token level)
    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_labels, filtered_preds, average='weighted', zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ======================================================
# 3. 数据预处理 (Correct ChatML + Mask)
# ======================================================

def preprocess_senti(example, tokenizer):
    # 1. 修正拼写错误
    label_text = "positive" if example["label"] == 1 else "negative"

    # 构建提示词 (prompt) 和 响应 (response)
    prompt = f"分析以下评论的情感：\n评论：{example['text']}\n情感："
    response = f"{label_text}{tokenizer.eos_token}" 

    full_text = prompt + response

    # 编码完整文本
    tokenized = tokenizer(full_text, truncation=True, max_length=512, padding="max_length")

    # 编码 prompt 以获取长度
    # 注意：这里为了保险，最好不要加 padding，直接算长度
    prompt_tokenized = tokenizer(prompt, truncation=True, max_length=512)
    prompt_len = len(prompt_tokenized["input_ids"])

    # 创建 labels
    labels = tokenized["input_ids"][:]

    # === 关键修正开始 ===
    for i in range(len(labels)):
        # 情况1: 如果是 Padding (attention_mask 为 0)，设置为 -100
        if tokenized["attention_mask"][i] == 0:
            labels[i] = -100
        # 情况2: 如果是 Prompt 部分，设置为 -100
        elif i < prompt_len:
            labels[i] = -100
        # 情况3: 剩下的就是 Response 部分，保留原 ID
    # === 关键修正结束 ===

    tokenized["labels"] = labels
    return tokenized


def benchmark_inference(model, tokenizer, dataset, num_samples=50):
    """
    测试模型推理速度
    """
    print(f"\n>>> 开始推理性能测试 (Samples={num_samples})...")
    model.eval()
    
    # Select a subset of data
    subset = dataset.select(range(min(num_samples, len(dataset))))
    
    times = []
    
    for i, example in enumerate(subset):
        # Prepare input (only prompt)
        prompt = f"分析以下评论的情感：\n评论：{example['text']}\n情感："
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            # Generate only a few new tokens
            _ = model.generate(
                **inputs, 
                max_new_tokens=5, 
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        
        # Skip first warmup sample
        if i > 0:
            times.append(end_time - start_time)
            
    if times:
        avg_time = np.mean(times)
        print(f"Average Inference Time: {avg_time*1000:.2f} ms/sample")
        print(f"Throughput: {1.0/avg_time:.2f} samples/sec")
        return avg_time
    else:
        return 0.0


def evaluate_generative(model, tokenizer, dataset):
    """
    使用生成式评估计算准确率 (Generative Evaluation)
    """
    print("\n>>> Starting Generative Evaluation...")
    from tqdm import tqdm

    total = 0
    correct = 0
    
    print(f"Total test samples: {len(dataset)}")
    
    model.eval()

    for i, example in enumerate(tqdm(dataset)):
        label = example["label"] # 0 or 1
        target_text = "positive" if label == 1 else "negative"
        
        prompt = f"分析以下评论的情感：\n评论：{example['text']}\n情感："
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            # 生成
            outputs = model.generate(
                **inputs, 
                max_new_tokens=5, # 只需要生成几个 token
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False # 确定性生成
            )
            
        # 解码
        # 只解码新生成的部分
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # 简单的包含匹配
        is_correct = False
        # 检查生成的文本是否以目标标签开头 (更严格一点)
        if generated_text.startswith(target_text):
            is_correct = True
        # 或者包含目标标签
        elif target_text in generated_text:
            is_correct = True
            
        if is_correct:
            correct += 1
        total += 1
        
        # 打印前几个错误案例用于调试
        if not is_correct and total <= 5:
             print(f"\n[Fail] Label: {target_text}, Pred: '{generated_text}'")

    accuracy = correct / total
    print("\n" + "="*30)
    print("✅ Test Results (Generative):")
    print("="*30)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print("="*30)
    return accuracy


# ======================================================
# 4. 主程序
# ======================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen1.5-1.8B-Chat")
    parser.add_argument("--output_dir", default="./lora_w4a8_out_2stage")
    parser.add_argument("--batch_size", type=int, default=1)  # 显存优化
    parser.add_argument("--grad_accum", type=int, default=16)  # 梯度累积
    parser.add_argument("--lr", type=float, default=2e-4)  # Stage 1 LR
    parser.add_argument("--lr_qat", type=float, default=2e-5)  # Stage 2 LR (Lower)
    parser.add_argument("--group_size", type=int, default=32)  # QA-LoRA Group Size
    parser.add_argument("--lora_rank", type=int, default=8)  # LoRA Rank
    args = parser.parse_args()

    set_seed(42)
    
    # Update output_dir to include configuration
    args.output_dir = f"{args.output_dir}/r{args.lora_rank}_g{args.group_size}"
    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------
    # A. 准备 Tokenizer & Data
    # ----------------------------------------------------
    print(">>> 正在加载 Tokenizer 和 数据集...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("lansinuote/ChnSentiCorp")
    
    # 预处理 (使用多进程加速)
    proc_kwargs = {"num_proc": 4, "remove_columns": ["text", "label"]}
    tokenized_train = dataset["train"].map(lambda x: preprocess_senti(x, tokenizer), **proc_kwargs)
    tokenized_test = dataset["test"].map(lambda x: preprocess_senti(x, tokenizer), **proc_kwargs)

    # Debug Check
    print("Debug Label Example:", [l for l in tokenized_train[0]['labels'] if l != -100])

    # ----------------------------------------------------
    # B. Stage 1: Standard FP16 LoRA Warmup
    # ----------------------------------------------------
    print("\n" + "=" * 50)
    print("🚀 Stage 1: Standard FP16 LoRA Training")
    print("=" * 50)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    # 必须开启梯度检查点以节省显存
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 配置 LoRA (Target 全部线性层)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=16,
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Callback
    metrics_cb1 = TrainingMetricsCallback()

    trainer1 = Trainer(
        model=model,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        args=TrainingArguments(
            output_dir=f"{args.output_dir}/stage1",
            num_train_epochs=1,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            fp16=True,
            save_strategy="epoch",
            evaluation_strategy="no",
            
            logging_steps=10,
            report_to="none",
            # dataloader_num_workers=0
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[metrics_cb1]
    )

    print(">>> 开始 Stage 1 训练...")
    trainer1.train()

    with torch.no_grad():
        # 保存 Stage 1 LoRA 参数
        torch.save(model.state_dict(), f"{args.output_dir}/stage1/lora_state_dict.pt")
        evaluate_generative(model, tokenizer, dataset["test"])



    # # 保存 Stage 1 指标
    # metrics_cb1.save_metrics(f"{args.output_dir}/stage1_metrics.json")

    # 清理显存 (彻底删除 model 和 trainer)
    del trainer1, model, metrics_cb1
    torch.cuda.empty_cache()
    gc.collect()

    # ----------------------------------------------------
    # C. Stage 2: W4A8 QA-LoRA Training
    # ----------------------------------------------------
    print("\n" + "=" * 50)
    print("🚀 Stage 2: W4A8 QA-LoRA (Group-wise Quant + Adaptation)")
    print("=" * 50)

    # 1. 重新加载干净的 Base Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 2. 原地转换为 W4A8 QA-LoRA 结构
    # 注意：Stage 1 只是为了让模型熟悉任务。Stage 2 我们使用新的 QA-LoRA 结构从头(Warmup)开始微调，
    # 或者你可以尝试加载 Stage 1 的参数，但因为 A 矩阵形状不匹配，重新初始化通常更简单且有效。
    model = convert_to_qalora_w4a8(
        model,
        r=args.lora_rank,
        lora_alpha=16,
        group_size=args.group_size,
        target_modules=target_modules
    )

    # 3. 冻结 Base，只训练 QA-LoRA 参数
    print(">>> 配置参数冻结...")
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # 打印可训练参数确认
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"QA-LoRA Trainable Params: {trainable_params} / {all_params} ({trainable_params / all_params:.2%})")

    metrics_cb2 = TrainingMetricsCallback()

    trainer2 = Trainer(
        model=model,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        args=TrainingArguments(
            output_dir=f"{args.output_dir}/stage2_w4a8",
            num_train_epochs=1,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr_qat,  # 使用较小的学习率
            fp16=True,
            logging_steps=10,
            report_to="none",
            save_strategy="epoch",
            evaluation_strategy="no"
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[metrics_cb2]
    )

    # 手动创建优化器，确保在访问trainer2.optimizer之前已初始化
    trainer2.create_optimizer()
    
    # 打印优化器参数组，确认lora_A和lora_B被正确包含
    print(">>> 检查优化器参数组...")
    # 先获取所有需要训练的lora参数
    lora_params = {p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad}
    
    # 检查这些参数是否在trainer的优化器中
    optimizer = trainer2.optimizer
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"  Param Group {i}:")
        print(f"    - LR: {param_group['lr']}")
        # 检查该组包含多少lora参数
        lora_count = sum(1 for param in param_group['params'] if param in lora_params)
        print(f"    - LoRA params in group: {lora_count}")
    
    # 打印一些具体的lora参数信息
    print(">>> 可训练的LoRA参数列表：")
    for n, p in model.named_parameters():
        if 'lora_' in n and p.requires_grad:
            print(f"  - {n}: {p.shape}, requires_grad={p.requires_grad}")
    
    print(">>> 开始 Stage 2 (W4A8 QAT) 训练...")
    trainer2.train()

    # 保存 Stage 2 指标和模型状态
    metrics_cb2.save_metrics(f"{args.output_dir}/stage2_metrics.json")
    torch.save(model.state_dict(), f"{args.output_dir}/stage2_w4a8/qalora_state_dict.pt")
    
    # 保存合并后的模型
    save_merged_model(model, f"{args.output_dir}/stage2_w4a8")

    # ----------------------------------------------------
    # D. 可视化
    # ----------------------------------------------------
    print(">>> 生成对比图表...")
    all_metrics = {
        "Stage1_FP16_LoRA": metrics_cb1.metrics,
        "Stage2_W4A8_QALoRA": metrics_cb2.metrics
    }
    plot_loss_curve(all_metrics, save_path=f"{args.output_dir}/loss_curve_w4a8.png")

    # ----------------------------------------------------
    # E. 推理测试
    # ----------------------------------------------------
    benchmark_inference(model, tokenizer, dataset["test"], num_samples=50)

    # ----------------------------------------------------
    # F. 生成式评估
    # ----------------------------------------------------
    evaluate_generative(model, tokenizer, dataset["test"])

    print(f"\n✅ 全部完成！输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()