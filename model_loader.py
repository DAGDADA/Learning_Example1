# 使用预训练权重
from email.policy import strict

from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter

def preload_model_from_standard_weights(ckpt_path,device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path,device) # 加载预训练权重

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict["encoder"],strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict["decoder"],strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=False)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)

    return {
        "clip":clip,
        "encoder":encoder,
        "decoder":decoder,
        "diffusion":diffusion
    }
