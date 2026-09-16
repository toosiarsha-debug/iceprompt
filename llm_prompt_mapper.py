import re
import torch
from transformers import pipeline


class LLMCategoryMapper:
    """
    پرامپت آزاد کاربر رو می‌گیره و برای هر دسته (Atmosphere/Genre/
    Instrument/Emotion) با کمک یک LLM محلی، بهترین گزینه رو از کاتالوگ
    ثابت (CATEGORIES) انتخاب می‌کنه. این همون لایه‌ای‌ست که پرامپت آزاد
    رو به فرمت مورد نیاز روش IEC تبدیل می‌کنه.

    اگر مدل لود نشود یا خروجی قابل‌تشخیص نباشد، برای همان دسته مقدار
    None برمی‌گردد تا فراخوان (optimizer) بتواند به روش embedding فالبک
    کند - یعنی هیچ‌وقت کل سیستم گیر نمی‌کند.
    """

    def __init__(self, categories, model_name="google/flan-t5-base"):
        self.categories = categories
        device = 0 if torch.cuda.is_available() else -1
        try:
            self.generator = pipeline(
                "text2text-generation",
                model=model_name,
                device=device
            )
            self.is_loaded = True
        except Exception as e:
            print(f"[LLMCategoryMapper] Could not load {model_name}, will fallback to embeddings: {e}")
            self.is_loaded = False

    def _ask_llm_for_category(self, user_prompt, cat, words):
        options_str = ", ".join(words)
        input_text = (
            f"Music description: \"{user_prompt}\"\n"
            f"Which one of these {cat.lower()} options best fits this description? "
            f"Options: {options_str}. "
            f"Answer with exactly one word or phrase from the list, nothing else."
        )
        try:
            res = self.generator(input_text, max_new_tokens=10, do_sample=False)
            raw = res[0]["generated_text"].strip().lower()
        except Exception:
            return None

        # تلاش برای پیدا کردن دقیق‌ترین گزینه‌ی معتبر داخل خروجی مدل
        # (مدل‌های کوچک گاهی متن اضافه یا حروف‌بزرگ/کوچک متفاوت برمی‌گردانند)
        for w in sorted(words, key=len, reverse=True):
            if re.search(rf"\b{re.escape(w.lower())}\b", raw):
                return w
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
