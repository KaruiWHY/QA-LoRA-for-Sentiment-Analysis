# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

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
# 0. 环境配置
# ======================================================
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 强制设置临时目录，防止 No usable temporary directory found
# _custom_temp_dir = os.path.join(os.getcwd(), "tmp_cache")
# os.makedirs(_custom_temp_dir, exist_ok=True)
# os.environ["TMPDIR"] = _custom_temp_dir
# os.environ["TEMP"] = _custom_temp_dir
# os.environ["TMP"] = _custom_temp_dir
# print(f">>> Temporary directory set to: {_custom_temp_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen1.5-1.8B-Chat")
    # 默认指向 Stage 1 的 checkpoint
    parser.add_argument("--lora_path", type=str, default="./lora_w4a8_out_2stage/r8_g32/stage1/checkpoint-600")
    args = parser.parse_args()

    print_gpu_memory("Start")

    print(f"🚀 Testing Stage 1 LoRA Model")
    print(f"Base Model: {args.base_model}")
    print(f"LoRA Path:  {args.lora_path}")

    # ----------------------------------------------------
    # 1. 加载 Tokenizer & Dataset
    # ----------------------------------------------------
    print(">>> Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(">>> Loading Dataset: lansinuote/ChnSentiCorp")
    dataset = load_dataset("lansinuote/ChnSentiCorp")
    test_subset = dataset["test"]
    print(f"Total test samples: {len(test_subset)}")

    # ----------------------------------------------------
    # 2. 加载模型 (Base + LoRA)
    # ----------------------------------------------------
    print(">>> Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print_gpu_memory("Base Model Loaded")

    print(f">>> Loading LoRA Adapter from {args.lora_path}...")
    try:
        model = PeftModel.from_pretrained(model, args.lora_path)
        print("✅ LoRA Adapter loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LoRA adapter: {e}")
        return

    print_gpu_memory("LoRA Adapter Loaded")

    model.eval()
    
    # ----------------------------------------------------
    # 3. 生成式评估 (Generative Evaluation)
    # ----------------------------------------------------
    print("\n>>> Starting Generative Evaluation...")
    
    total = 0
    correct = 0
    
    for i, example in enumerate(tqdm(test_subset)):
        label = example["label"] # 0 or 1
        target_text = "positive" if label == 1 else "negative"
        
        # 构建 Prompt (与训练时保持一致)
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
            
        # 解码 (只解码新生成的部分)
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # 简单的包含匹配
        is_correct = False
        # 检查生成的文本是否以目标标签开头
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
    
    print_gpu_memory("Inference Finished")

    print("\n" + "="*30)
    print("✅ Test Results (Stage 1 LoRA):")
    print("="*30)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print("="*30)

if __name__ == "__main__":
    main()
