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

        # "eager" فقط روی CPU لازم است (برای رفع باگ NaN مخصوص Apple
        # Accelerate/vecLib روی مک). روی GPU عمدا هیچ attn_implementation
        # را اجبار نمی‌کنیم: زیرمدل صوتی (EncodecModel) اصلا از "sdpa"
        # پشتیبانی نمی‌کند و اجبار آن باعث کرش می‌شود؛ با نگفتن چیزی،
        # خود transformers برای هر زیرمدل بهترین پیاده‌سازی ممکن را
        # انتخاب می‌کند.
        model_kwargs = {"torch_dtype": self.dtype}
        if self.device == "cpu":
            model_kwargs["attn_implementation"] = "eager"

        print(f"Loading MusicGen ({model_id}) on {self.device} with {self.dtype}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            model_id, **model_kwargs
        ).to(self.device)
        self.sampling_rate = self.model.config.audio_encoder.sampling_rate
        self.guidance_scale = 3.0

    def generate(self, prompt, duration_seconds=5, skip_seconds=0, seed=None):
        """تولید تکی"""
        return self.generate_batch(
            [prompt], duration_seconds=duration_seconds,
            skip_seconds=skip_seconds, seed=seed
        )[0]

    def _ensure_device(self):
        """
        روی Hugging Face ZeroGPU، کارت گرافیک فقط داخل تابعی که با
        @spaces.GPU دکوریت شده واقعا وصل می‌شود - موقع import/راه‌اندازی
        اولیه، torch.cuda.is_available() ممکن است هنوز False برگرداند و
        مدل روی CPU لود شده باشد. این متد همین‌جا (داخل فراخوانی واقعی
        تولید) دوباره چک می‌کند و اگر GPU حالا در دسترس است ولی مدل هنوز
        روی CPU است، یک‌بار مدل را به GPU منتقل می‌کند.
        """
        if self.device != "cuda" and torch.cuda.is_available():
            print("[MusicGenerator] GPU just became available, moving model to cuda...")
            self.device = "cuda"
            self.dtype = torch.float16
            self.model = self.model.to(self.device, dtype=self.dtype)

    def generate_batch(self, prompts, duration_seconds=5, skip_seconds=0, seed=None):
        """
        تولید همزمان چند پرامپت با هم روی کارت گرافیک.

        skip_seconds: چون MusicGen همیشه از ثانیه‌ی صفر تولید می‌کند (نه
        از وسط آهنگ)، یک نمونه‌ی کوتاه همیشه دقیقا ابتدای قطعه است که
        اغلب فقط یک intro ساکت/ساده است، نه بخشی که واقعا مضمون ژانر یا
        حس را نشان دهد. با تولید duration_seconds کامل ولی دور ریختن
        skip_seconds اول از خروجی پخش‌شده، بخشی که می‌شنوی به توسعه‌ی
        واقعی قطعه نزدیک‌تر است.
        """
        self._ensure_device()

        # seed ثابت: چون نمونه‌گیری MusicGen تصادفی است، یک پرامپت
        # یکسان هر بار صدای متفاوتی می‌دهد و بخشی از نمره‌ی کاربر عملا
        # برای «شانس همان تولید» می‌شود، نه خود پرامپت. با seed یکسان برای
        # همه‌ی نمونه‌های یک جلسه، تفاوت بین نمونه‌ها فقط از تفاوت پرامپت‌ها
        # می‌آید و یک پرامپت همیشه همان صدا را می‌دهد (مثلا نمونه‌ی الیت
        # در نسل بعد دقیقا همان چیزی است که نمره‌اش را دادی).
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

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
                guidance_scale=self.guidance_scale
            )

        output_files = []
        audio_data = audio_values.detach().cpu().float().numpy()

        # آزاد کردن حافظه‌ی GPU بین فراخوانی‌ها
        if self.device == "cuda":
            del audio_values
            torch.cuda.empty_cache()

        skip_samples = int(skip_seconds * self.sampling_rate)

        for audio in audio_data:
            audio_arr = audio[0]

            if skip_samples > 0 and skip_samples < len(audio_arr):
                audio_arr = audio_arr[skip_samples:]

            max_val = np.max(np.abs(audio_arr))
            if max_val > 0:
                audio_arr = audio_arr / max_val
            audio_int16 = (audio_arr * 32767).astype(np.int16)

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            scipy.io.wavfile.write(tmp.name, self.sampling_rate, audio_int16)
            output_files.append(tmp.name)

        return output_files


AudioGenerator = MusicGenerator
