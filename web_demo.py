import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.tuners.lora.layer import Linear as LoRALinear

# ==========================================
# 1. 复制必要的模型定义 (必须与训练代码一致)
# ==========================================

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
        self.weight = base_layer.weight
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.reduced_dim = self.in_features // group_size
        
        self.lora_A = nn.Parameter(torch.zeros(r, self.reduced_dim))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        
        # 注册 buffer 以便加载权重
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

def convert_to_qalora_structure(model, r=8, group_size=32, target_modules=["q_proj", "v_proj"]):
    """仅用于推理的结构转换"""
    print(f"🔄 转换模型结构: r={r}, group_size={group_size}")
    
    def replace_module(module, current_name=""):
        for name, child in module.named_children():
            full_name = f"{current_name}.{name}" if current_name else name
            if isinstance(child, (LoRALinear, nn.Linear)):
                is_target = any(t in name for t in target_modules)
                if isinstance(child, LoRALinear) and is_target:
                    base_layer = child.base_layer
                    new_layer = QALoRALayer(base_layer, r=r, group_size=group_size)
                    new_layer = new_layer.to(base_layer.weight.device).to(base_layer.weight.dtype)
                    setattr(module, name, new_layer)
                elif isinstance(child, nn.Linear) and is_target:
                    new_layer = QALoRALayer(child, r=r, group_size=group_size)
                    new_layer = new_layer.to(child.weight.device).to(child.weight.dtype)
                    setattr(module, name, new_layer)
            else:
                replace_module(child, full_name)

    replace_module(model)
    return model

# ==========================================
# 2. 加载模型与配置
# ==========================================

# --- 配置区域 ---
MODEL_PATH = "Qwen/Qwen1.5-1.8B-Chat"
# 使用您刚刚测试通过的 checkpoint 路径
CHECKPOINT_PATH = "/root/QALoRA/lora_w4a8_out/r8_g32/stage2_w4a8/checkpoint-600/model.safetensors" 
LORA_RANK = 8
GROUP_SIZE = 32
# ----------------

print(">>> 正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print(">>> 正在加载 Base Model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    trust_remote_code=True
)

# 转换结构
model = convert_to_qalora_structure(
    model, 
    r=LORA_RANK, 
    group_size=GROUP_SIZE, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

print(f">>> 正在加载训练好的权重: {CHECKPOINT_PATH}")
if CHECKPOINT_PATH.endswith(".safetensors"):
    from safetensors.torch import load_file
    state_dict = load_file(CHECKPOINT_PATH)
else:
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")

# 加载权重
keys = model.load_state_dict(state_dict, strict=False)
print(f"Load keys info: {keys}")

model.eval()
print(">>> 模型加载完成！")

# ==========================================
# 3. 定义推理逻辑
# ==========================================

def predict(text, history=None):
    if history is None:
        history = []
        
    # 构造 Prompt (针对情感分析任务微调的格式)
    prompt = f"分析以下评论的情感：\n评论：{text}\n情感："
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=10, 
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False # 情感分析通常不需要采样
        )
    
    # 解码并提取结果
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取 "情感：" 后面的部分
    if "情感：" in full_response:
        response = full_response.split("情感：")[-1].strip()
    else:
        response = full_response
        
    return response

# ==========================================
# 4. 启动 Web 界面 (美化版)
# ==========================================

# 自定义 CSS
custom_css = """
.container { max-width: 900px; margin: auto; padding-top: 20px; }
.header-text { text-align: center; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
.header-title { font-size: 2.5em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
.header-subtitle { font-size: 1.2em; color: #7f8c8d; margin-bottom: 20px; }
.team-info { text-align: center; margin-bottom: 30px; color: #34495e; font-weight: 500; font-size: 1.1em; }
.footer { text-align: center; margin-top: 40px; color: #95a5a6; font-size: 0.8em; }
"""

# 使用 Soft 主题
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=["sans-serif"]
)

with gr.Blocks(theme=theme, css=custom_css, title="QA-LoRA 情感分析") as demo:
    with gr.Column(elem_classes="container"):
        # Header
        gr.Markdown(
            """
            <div class="header-text">
                <div class="header-title">中文评论情感分析系统</div>
                <div class="header-subtitle">基于 Qwen1.5-1.8B 的 W4A8 量化微调模型</div>
            </div>
            """
        )
        
        # Team Info
        gr.Markdown(
            """
            <div class="team-info">
                👨‍💻 小组成员： 石晨霡 · 吴昊阳 · 孟令儒
            </div>
            """
        )

        # Main Content
        with gr.Group():
            with gr.Row():
                with gr.Column(scale=1):
                    input_text = gr.Textbox(
                        label="输入评论", 
                        placeholder="请输入您想分析的中文评论...", 
                        lines=5,
                        show_copy_button=True
                    )
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ 清空", variant="secondary")
                        submit_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")

                with gr.Column(scale=1):
                    output_text = gr.Textbox(
                        label="分析结果", 
                        lines=5,
                        interactive=False,
                        show_copy_button=True
                    )
                    
                    # 技术参数折叠面板
                    with gr.Accordion("ℹ️ 模型技术参数", open=True):
                        gr.Markdown(
                            f"""
                            - **基础模型**: Qwen/Qwen1.5-1.8B-Chat
                            - **量化方法**: QA-LoRA (W4A8)
                            - **LoRA Rank**: {LORA_RANK}
                            - **Group Size**: {GROUP_SIZE}
                            - **数据集**: ChnSentiCorp
                            """
                        )

        # Examples
        gr.Markdown("### 📝 测试样例")
        gr.Examples(
            examples=[
                ["唐老师的课程太有意思啦，学到了很多实用的知识！"],
                ["房间很干净，服务也很周到，下次还会来。"],
                ["隔音效果太差了，一晚上没睡好。"],
                ["虽然位置有点偏，但是性价比很高。"],
                ["快递太慢了，包装也破损了，差评！"],
                ["这本书的内容非常精彩，值得一读。"]
            ],
            inputs=input_text,
            outputs=output_text,
            fn=predict,
            cache_examples=False,
        )

        # Footer
        # gr.Markdown(
        #     """
        #     <div class="footer">
        #         Powered by QA-LoRA & Gradio | 2025
        #     </div>
        #     """
        # )

    # Event Handlers
    submit_btn.click(fn=predict, inputs=input_text, outputs=output_text)
    clear_btn.click(lambda: ("", ""), outputs=[input_text, output_text])

if __name__ == "__main__":
    # share=True 会生成一个公共链接，方便您在课堂上展示
    # demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    demo.launch(server_name="0.0.0.0", share=False)
