import torch
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from config import MODEL_CONFIG


class MusicGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = MODEL_CONFIG["musicgen_model"]
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            self.model_name
        ).to(self.device)
        self.sampling_rate = self.model.config.audio_encoder.sampling_rate
        self.duration = MODEL_CONFIG["generation_duration"]

    def generate(self, prompt_text):
        inputs = self.processor(
            text=[prompt_text],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        max_new_tokens = int(self.duration * 50)
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                guidance_scale=1.0 if self.device == "cpu" else 3.0,
                temperature=1.0
            )

        audio_data = audio_values[0, 0].cpu().numpy()
        audio_data = audio_data / np.max(np.abs(audio_data))
        return self.sampling_rate, audio_data
