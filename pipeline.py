import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = 512 // 8
LATENT_HEIGHT = 512 // 8


# 用来生成图像
def generate(
    prompt:str,
    uncond_prompt:str,
    input_image=None,
    strength=0.8,  # strength 当我们由一个输入图像且希望从一个图像生成另一个图像时候，希望对初始的图像给予多少关注
    do_cfg=True,
    cfg_scale=7.5,
    sample_name="ddpm",
    n_inference_step=50,
    models={},   # model 使用预训练模型
    seed=None,  # seed 初始化随机数生成器
    device=None,
    idle_device=None,
    tokenizer=None
):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")

        if idle_device: # 将东西移动到CPU 必须在0 1 之间
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)  # 用来生成噪声的随机数生成器
        if seed is not None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # 将prompt转化为token ： tokenizer
            cond_token = tokenizer.batch_encoder_plus([prompt],padding="max_length",max_length=77).input_ids
            conda_token = torch.Tensor(cond_token).to(device=device,dtype=torch.long)   # batch_size,seq_len
            conda_context = clip(conda_token)  # batch_size,seq_len -> batch_size,seq_len,dim


