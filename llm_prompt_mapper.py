import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMCategoryMapper:
    """
    پرامپت آزاد کاربر رو می‌گیره و برای هر دسته (Atmosphere/Genre/
    Instrument/Emotion) با کمک یک LLM محلی، بهترین گزینه رو از کاتالوگ
    ثابت (CATEGORIES) انتخاب می‌کنه. این همون لایه‌ای‌ست که پرامپت آزاد
    رو به فرمت مورد نیاز روش IEC تبدیل می‌کنه.

    مدل پیش‌فرض: HuggingFaceTB/SmolLM3-3B — کاملا باز (Apache 2.0)،
    بدون gate و بدون نیاز به توکن/لایسنس، و برای این نوع دستورالعمل‌های
    محدودکننده (انتخاب از یک لیست کوتاه) به‌طور محسوسی بهتر از مدل‌های
    قبلی (flan-t5-base) عمل می‌کند.

    اگر مدل لود نشود یا خروجی قابل‌تشخیص نباشد، برای همان دسته مقدار
    None برمی‌گردد تا فراخوان (optimizer) بتواند به روش embedding فالبک
    کند - یعنی هیچ‌وقت کل سیستم به این LLM وابسته‌ی صرف نمی‌ماند. علاوه
    بر این، optimizer.py حتی وقتی LLM جواب معتبری بدهد، آن را با بهترین
    گزینه‌ی embedding مقایسه می‌کند و فقط اگر واقعا بهتر یا هم‌ارز باشد
    قبولش می‌کند (نگاه کنید به _resolve_category_word).
    """

    def __init__(self, categories, model_name="HuggingFaceTB/SmolLM3-3B"):
        self.categories = categories
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.is_loaded = True
        except Exception as e:
            print(f"[LLMCategoryMapper] Could not load {model_name}, will fallback to embeddings: {e}")
            self.is_loaded = False

    def _ask_llm_for_category(self, user_prompt, cat, words):
        options_str = ", ".join(words)
        user_msg = (
            f"Music description: \"{user_prompt}\"\n"
            f"Which one of these {cat.lower()} options best fits this description? "
            f"Options: {options_str}. "
            f"If one of these options fits well, reply with exactly that word or phrase. "
            f"If none of them really fit, reply with a single better word or short phrase "
            f"of your own instead. Reply with only the word or phrase, nothing else."
        )
        # "/no_think" حالت استدلال طولانی SmolLM3 رو خاموش می‌کنه تا
        # جواب کوتاه و مستقیم بدیم (برای این کار ساده لازم نیست فکر کنه)
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": user_msg},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=15, do_sample=False
                )
            output_ids = generated_ids[0][len(inputs.input_ids[0]):]
            raw = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip().lower()
        except Exception:
            return None

        if not raw:
            return None

        # اول تلاش می‌کنیم ببینیم آیا خروجی دقیقاً یکی از گزینه‌های
        # موجود کاتالوگ است (برای جلوگیری از تکثیر مترادف‌های نزدیک)
        for w in sorted(words, key=len, reverse=True):
            if re.search(rf"\b{re.escape(w.lower())}\b", raw):
                return w

        # اگر هیچ‌کدام از گزینه‌های کاتالوگ مطابقت نداشت، خروجی خام مدل
        # را به‌عنوان یک کلمه/عبارت جدید برمی‌گردانیم (نه رد کردنش).
        # optimizer.py مسئول اضافه کردن این کلمه‌ی جدید به کاتالوگ است.
        cleaned = re.sub(r'["\'.!?]', '', raw).strip()
        word_count = len(cleaned.split())
        if 0 < word_count <= 3:
            return cleaned
        return None

    def map_to_categories(self, user_prompt):
        """
        خروجی: دیکشنری {category: word_or_None}
        اگر مدل لود نشده باشد، دیکشنری با همه‌ی مقادیر None برمی‌گردد
        و فراخوان باید به‌طور کامل به embedding فالبک کند.
        """
        if not self.is_loaded:
            return {cat: None for cat in self.categories}

        result = {}
        for cat, words in self.categories.items():
            result[cat] = self._ask_llm_for_category(user_prompt, cat, words)
        return result
