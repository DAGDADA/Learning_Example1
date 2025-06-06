import torch
import numpy as np

class DDPMSampler:
    def __init__(self,generator:torch.Generator,num_training_steps=1000,beta_start:float = 0.00085,beta_end:float = 0.0120):
        self.betas = torch.linspace(beta_start ** 0.5,beta_end ** 0.5,num_training_steps,dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas,0) # alpha0,alpha0*alpha1,alpha0*alpha1*alpha2,...
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
        self.strength = 1.0

    def set_inference_timesteps(self,num_inference_steps):
        self.num_inference_steps = num_inference_steps
        # 999,998,997,996...0 -> 999,999-20,999-40...0
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0,num_inference_steps)*step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    # def add_noise(self,latent:torch.tensor,timesteps):
