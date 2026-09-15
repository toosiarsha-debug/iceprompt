import os
import torch
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from config import AppConfig

class AudioGenerator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(config.MODEL_NAME)
        self.model = MusicgenForConditionalGeneration.from_pretrained(config.MODEL_NAME).to(self.device)
        self.sampling_rate = self.model.config.audio_encoder.sampling_rate

    def generate(self, prompt: str):
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        max_new_tokens = int(self.config.DURATION * 50)
        
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                guidance_scale=self.config.GUIDANCE_SCALE
            )
            
        audio_data = audio_values[0, 0].cpu().numpy()
        
        # اصلاح نویز و کلیپینگ ایمن
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        max_val = np.max(np.abs(audio_data))
        if max_val > 1e-4:
            audio_data = audio_data / max_val
            
        audio_int16 = (audio_data * 32767).astype(np.int16)
        return (self.sampling_rate, audio_int16)
