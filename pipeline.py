import torch
import numpy as np
from sympy import false
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
    sampler_name="ddpm",
    n_inference_steps=50,
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

        '''__________________________________ 随机数生成器 __________________________________'''
        generator = torch.Generator(device=device)
        if seed:
            generator.seed()
        else:
            generator.manual_seed(seed)

        '''__________________________________CLIP 将文本转化为可作为模型输入的多维向量__________________________'''
        clip = models["clip"]
        clip.to(device)
        if do_cfg:  # 如果想做无分类引导，那么就创造两个batch，一个有条件、一个没条件，这两个都要经过Unet的推导
            # 将prompt转化为token ： tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            conda_tokens = torch.Tensor(cond_tokens).to(dtype=torch.long,device=device)  # batch_size,seq_len
            conda_context = clip(conda_tokens)  # batch_size,seq_len -> batch_size,seq_len,dim

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.Tensor(uncond_tokens).to(device=device, dtype=torch.long)
            uncond_context = clip(uncond_tokens)  # batch_size,seq_len -> batch_size,seq_len,dim

            # 2,seq_len,dim -> 2,77,768  (batch_size == 2)
            context = torch.cat([conda_context, uncond_context])
        else:
            # 将prompt转化为token ： tokenizer
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.Tensor(tokens).to(device=device, dtype=torch.long)  # batch_size,seq_len
            context = clip(tokens)  # 1,77,768
            to_idle(clip)

        if sampler_name == "ddpm":  # 噪声采样器  DDPM\DDIM\基于几何微分方程
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"unknown sampler{sampler_name}")

        '''__________________________________ENCODER 获取潜在向量__________________________'''
        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)
        if input_image:
            ### 从图像生成图像
            encoder = models["encoder"]
            encoder.to(device=device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor).to(device=device, dtype=torch.float32) # height,width,channel

            input_image_tensor = rescale(input_image_tensor,(0,255),(-1,1))  # 归一化
            input_image_tensor = input_image_tensor.unsqueeze(0)   # height,width,channel -> batch_size,height,width,channel
            input_image_tensor = input_image_tensor.permute(0,3,1,2)  # height,width,channel -> batch_size,channel,height,width

            encoder_noise = torch.randn(latents_shape,generator=generator,device=device)

            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents,sampler.timesteps[0]) # 初始值

            to_idle(encoder)
        else:
            ### 无条件，从随机噪声中生成图像
            latents = torch.randn(latents_shape, generator=generator, device=device)  # 随机噪声 ~N(0,1)

        '''__________________________________ DIFFUSION __________________________________'''
        diffusion = models["diffusion"]
        diffusion.to(device=device)

        #### 执行噪声预测，并从图像中去除噪声
        timesteps = tqdm(sampler.timesteps)
        for i,timestep in enumerate(timesteps):
            # 1,320
            time_embedding = get_time_embedding(timestep).to(device=device)

            model_input = latents  # batch_size,4,LATENT_HEIGHT,LATENT_WIDTH

            if do_cfg:
                # batch_size,4,LATENT_HEIGHT,LATENT_WIDTH -> 2*batch_size,4,LATENT_HEIGHT,LATENT_WIDTH
                model_input = model_input.repeat(2,1,1,1)

            '''执行模型预测'''
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunl(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(timestep,latents,model_output)

        to_idle(diffusion)

        '''__________________________________ DECODER 从潜在向量中还原图像 __________________________'''
        decoder = models["decoder"]
        decoder.to(device=device)

        image = decoder(latents)
        to_idle(decoder)

        image = rescale(image,(-1,1),(0,255))
        image = image.permute(0,2,3,1)  # batch_size,channel,height,width -> batch_size,height,width,channel
        image = image.to("cpu",torch.uint8).numpy()
        return image


def rescale(x, old_range, new_range,clamp=false):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x =x.clamp(min=new_min, max=new_max)
    return x

# 将一个整数转化为一个320的向量 类似于一种位置嵌入
def get_time_embedding(timestep):
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0,end=160,dtype=torch.float32)/160)
    # (1,160)
    x = torch.tensor([timestep],dtype=torch.float32)[:,None] * freqs[None]
    # (1,320)
    return torch.cat([torch.cos(x),torch.sin(x)],dim=-1)

