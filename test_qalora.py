# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from sklearn.metrics import precision_recall_fscore_support
from peft.tuners.lora.layer import Linear as LoRALinear

# ======================================================
# 0. 显存监控工具 (新增)
# ======================================================
def print_gpu_memory(tag=""):
    """打印当前和峰值显存占用"""
    if torch.cuda.is_available():
        # 确保同步，获取准确值
        torch.cuda.synchronize()
        max_mem = torch.cuda.max_memory_allocated() / 1024**3
        current_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"\n📊 [{tag}] Max GPU Memory: {max_mem:.2f} GB | Current: {current_mem:.2f} GB")
        # 重置峰值统计，以便测量下一阶段的独立峰值
        torch.cuda.reset_peak_memory_stats()

# ======================================================
# 1. 核心量化组件 (复制自 QAT-LoRA.py)
# ======================================================

class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        scale = torch.clamp(scale, min=1e-8)
        q = torch.clamp(torch.round(x / scale), -127, 127)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def calc_scale_tokenwise(x, eps=1e-8):
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
        # 注意：测试时不需要梯度
        self.weight.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.reduced_dim = self.in_features // group_size

        self.lora_A = nn.Parameter(torch.zeros(r, self.reduced_dim))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        
        self.register_buffer('quant_scale', None)
        self.register_buffer('quant_zero', None)

    def fake_quant_activation(self, x):
        scale = calc_scale_tokenwise(x)
        return FakeQuantSTE.apply(x, scale)

    def fake_quant_weight_asym(self, w):
        out_dim, in_dim = w.shape
        w_reshaped = w.reshape(out_dim, in_dim // self.group_size, self.group_size)
        max_val = w_reshaped.amax(dim=-1, keepdim=True)
        min_val = w_reshaped.amin(dim=-1, keepdim=True)
        alpha = (max_val - min_val) / 15.0
        alpha = torch.clamp(alpha, min=1e-5)
        beta = min_val
        w_int = ((w_reshaped - beta) / alpha).round().clamp(0, 15)
        w_recon = w_int * alpha + beta
        w_recon = w_recon.reshape(out_dim, in_dim)
        return w_recon, alpha, beta

    def forward(self, x):
        x_q = self.fake_quant_activation(x)
        w_q, _, _ = self.fake_quant_weight_asym(self.weight)
        base_out = F.linear(x_q, w_q)
        b, s, d = x_q.shape
        x_reshaped = x_q.reshape(b, s, d // self.group_size, self.group_size)
        x_pooled = x_reshaped.sum(dim=-1)
        lora_input = x_pooled.to(self.lora_A.dtype)
        lora_out = (lora_input @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out.to(base_out.dtype)

def convert_to_qalora_w4a8(model, r=8, lora_alpha=16, group_size=32, target_modules=None):
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    print(f"🔄 正在将模型转换为 W4A8 QA-LoRA (Group Size={group_size})...")

    def replace_module(module, current_name=""):
        for name, child in module.named_children():
            full_name = f"{current_name}.{name}" if current_name else name
            if isinstance(child, (LoRALinear, nn.Linear)):
                is_target = any(t in name for t in target_modules)
                if isinstance(child, LoRALinear) and is_target:
                    base_layer = child.base_layer
                    new_layer = QALoRALayer(base_layer, r=r, lora_alpha=lora_alpha, group_size=group_size)
                    new_layer = new_layer.to(base_layer.weight.device).to(base_layer.weight.dtype)
                    setattr(module, name, new_layer)
                elif isinstance(child, nn.Linear) and is_target:
                    new_layer = QALoRALayer(child, r=r, lora_alpha=lora_alpha, group_size=group_size)
                    new_layer = new_layer.to(child.weight.device).to(child.weight.dtype)
                    setattr(module, name, new_layer)
            else:
                replace_module(child, full_name)

    replace_module(model)
    return model

# ======================================================
# 2. 数据处理与评估 (复制自 QAT-LoRA.py)
# ======================================================

def preprocess_senti(example, tokenizer):
    label_text = "positive" if example["label"] == 1 else "negative"
    prompt = f"分析以下评论的情感：\n评论：{example['text']}\n情感："
    response = f"{label_text}{tokenizer.eos_token}" 
    full_text = prompt + response
    tokenized = tokenizer(full_text, truncation=True, max_length=512, padding="max_length")
    prompt_tokenized = tokenizer(prompt, truncation=True, max_length=512)
    prompt_len = len(prompt_tokenized["input_ids"])
    labels = tokenized["input_ids"][:]
    for i in range(len(labels)):
        if tokenized["attention_mask"][i] == 0:
            labels[i] = -100
        elif i < prompt_len:
            labels[i] = -100
    tokenized["labels"] = labels
    return tokenized

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    filtered_preds = predictions[mask]
    filtered_labels = labels[mask]
    correct = (filtered_preds == filtered_labels).sum()
    total = mask.sum()
    accuracy = correct / total if total > 0 else 0.0
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
# 3. 主程序
# ======================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen1.5-1.8B-Chat")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to qalora_state_dict.pt")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    print_gpu_memory("Start")

    print(f">>> Loading Tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(">>> Loading Dataset: lansinuote/ChnSentiCorp")
    dataset = load_dataset("lansinuote/ChnSentiCorp")
    
    # 预处理测试集
    print(">>> Preprocessing test set...")
    test_dataset = dataset["test"].map(
        lambda x: preprocess_senti(x, tokenizer), 
        num_proc=4,
        remove_columns=["text", "label"]
    )

    print(f">>> Loading Base Model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print_gpu_memory("Base Model Loaded")

    # 转换结构
    print(f">>> Converting to QA-LoRA (r={args.lora_rank}, g={args.group_size})...")
    model = convert_to_qalora_w4a8(model, r=args.lora_rank, group_size=args.group_size)
    print_gpu_memory("Converted to QA-LoRA")

    # 加载权重
    print(f">>> Loading weights from {args.checkpoint_path}")
    
    if args.checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(args.checkpoint_path)
    else:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    
    # 加载权重 (strict=False 因为 state_dict 可能包含一些多余的键，或者缺少 base model 的冻结键)
    # 关键是确保 lora_A, lora_B 等被正确加载
    print(">>> Loading state dict...")
    load_result = model.load_state_dict(state_dict, strict=False)
    
    print(f"Missing keys: {len(load_result.missing_keys)}")
    if len(load_result.missing_keys) > 0:
        print(f"Sample missing keys: {load_result.missing_keys[:5]}")
        
    print(f"Unexpected keys: {len(load_result.unexpected_keys)}")
    if len(load_result.unexpected_keys) > 0:
        print(f"Sample unexpected keys: {load_result.unexpected_keys[:5]}")

    print_gpu_memory("Weights Loaded")

    # 检查 LoRA 权重是否成功加载 (检查第一个 QALoRALayer 的 lora_B 是否全为 0)
    for name, module in model.named_modules():
        if isinstance(module, QALoRALayer):
            if module.lora_B.sum() == 0:
                print(f"⚠️ WARNING: {name}.lora_B is all zeros! LoRA weights might not be loaded correctly.")
            else:
                print(f"✅ {name}.lora_B loaded successfully (sum={module.lora_B.sum().item():.4f}).")
            break # 只检查第一个

    # 确保模型在评估模式
    model.eval()
    torch.cuda.empty_cache()

    # 评估 (使用生成式评估 Generative Evaluation)
    print(">>> Starting Generative Evaluation...")
    from tqdm import tqdm

    total = 0
    correct = 0
    
    # 使用原始数据集的 test split，而不是预处理后的 tokenized dataset
    # 因为我们需要原始文本来构建 prompt
    test_subset = dataset["test"]
    
    print(f"Total test samples: {len(test_subset)}")

    for i, example in enumerate(tqdm(test_subset)):
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
        if not is_correct and total <= 10:
             print(f"\n[Fail] Label: {target_text}, Pred: '{generated_text}'")

    accuracy = correct / total
    
    print_gpu_memory("Inference Finished")

    print("\n" + "="*30)
    print("✅ Test Results (Generative):")
    print("="*30)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print("="*30)

if __name__ == "__main__":
    main()
