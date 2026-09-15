import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import numpy as np
import tempfile

class MusicGenerator:
    def __init__(self, config=None, model_id="facebook/musicgen-small"):
        # اگر ورودی رشته نباشد و شیء یا دیکشنری کانفیگ باشد
        if config is not None:
            if isinstance(config, str):
                model_id = config
            elif hasattr(config, "MODEL_CONFIG"):
                model_id = config.MODEL_CONFIG.get("musicgen_model", model_id)
            elif isinstance(config, dict):
                model_id = config.get("musicgen_model", model_id)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"Loading MusicGen ({model_id}) on {self.device} with {self.dtype}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=self.dtype
        ).to(self.device)
        self.sampling_rate = self.model.config.audio_encoder.sampling_rate

    def generate(self, prompt, duration_seconds=5):
        """تولید تکی"""
        return self.generate_batch([prompt], duration_seconds=duration_seconds)[0]

    def generate_batch(self, prompts, duration_seconds=5):
        """تولید همزمان چند پرامپت با هم روی کارت گرافیک"""
        inputs = self.processor(
            text=prompts,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # ۵ ثانیه = حدود ۲۵۰ توکن صوتی
        max_tokens = int(duration_seconds * 50)

        with torch.inference_mode():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                guidance_scale=3.0
            )

        output_files = []
        audio_data = audio_values.detach().cpu().float().numpy()

        for audio in audio_data:
            audio_arr = audio[0]
            max_val = np.max(np.abs(audio_arr))
            if max_val > 0:
                audio_arr = audio_arr / max_val
            audio_int16 = (audio_arr * 32767).astype(np.int16)

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            scipy.io.wavfile.write(tmp.name, self.sampling_rate, audio_int16)
            output_files.append(tmp.name)

        return output_files

AudioGenerator = MusicGenerator
